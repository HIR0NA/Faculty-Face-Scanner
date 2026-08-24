#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║  J.A.R.V.I.S. — Faculty Face Recognition Web Interface     ║
║  ระบบค้นหาใบหน้าอัตโนมัติผ่านหน้าเว็บ สไตล์ Iron Man       ║
╚══════════════════════════════════════════════════════════════╝

Flask Web Application ที่ให้ผู้ใช้:
- อัปโหลดรูปตัวอย่างอาจารย์
- ตั้งค่า URL / Threshold / Depth
- สั่งรัน Scanner แบบ real-time
- ดูผลลัพธ์พร้อมที่มา (กิจกรรม/ข่าว)
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request, send_from_directory

# ══════════════════════════════════════════════════════════════
#  Flask App Setup
# ══════════════════════════════════════════════════════════════
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max upload

BASE_DIR = Path(__file__).parent
SAMPLES_DIR = BASE_DIR / "teacher_samples"
OUTPUT_DIR = BASE_DIR / "matched_results"
REPORT_FILE = BASE_DIR / "scan_report.csv"
SAMPLES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def sanitize_folder_name(title: str) -> str:
    """แปลง page title เป็นชื่อโฟลเดอร์ที่ใช้ได้บน Windows/Linux"""
    if not title or title == "ไม่ระบุ":
        return "ไม่ระบุกิจกรรม"
    # ลบ characters ที่ใช้เป็นชื่อไฟล์/โฟลเดอร์ไม่ได้
    name = re.sub(r'[<>:"/\\|?*]', '', title)
    # ลบช่องว่างหัวท้าย และจุดท้าย (Windows ไม่ชอบ)
    name = name.strip().rstrip('.')
    # จำกัดความยาว (Windows max path component = 255)
    if len(name) > 120:
        name = name[:120].rstrip()
    return name or "ไม่ระบุกิจกรรม"

# ══════════════════════════════════════════════════════════════
#  Global Scanner State
# ══════════════════════════════════════════════════════════════
scanner_state = {
    "status": "idle",           # idle, loading_model, building_profile, crawling, scanning, done, error
    "message": "",
    "progress": 0,
    "total": 0,
    "pages_crawled": 0,
    "images_found": 0,
    "images_processed": 0,
    "faces_detected": 0,
    "matches_found": 0,
    "errors": 0,
    "start_time": None,
    "elapsed": "00:00",
    "matches": [],              # list of match records
    "log_lines": [],            # recent log lines for live feed
}
scanner_lock = threading.Lock()
scanner_thread: Optional[threading.Thread] = None
stop_event = threading.Event()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("JARVIS")


def add_log(msg: str):
    """เพิ่ม log line เข้า state สำหรับแสดงใน UI"""
    with scanner_lock:
        scanner_state["log_lines"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "msg": msg,
        })
        # เก็บแค่ 200 บรรทัดล่าสุด
        if len(scanner_state["log_lines"]) > 200:
            scanner_state["log_lines"] = scanner_state["log_lines"][-200:]


def update_state(**kwargs):
    with scanner_lock:
        scanner_state.update(kwargs)
        if scanner_state["start_time"]:
            delta = time.time() - scanner_state["start_time"]
            m, s = divmod(int(delta), 60)
            scanner_state["elapsed"] = f"{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════
#  Scanner Engine (runs in background thread)
# ══════════════════════════════════════════════════════════════
def run_scanner(start_url: str, allowed_domain: str, max_depth: int,
                similarity_threshold: float, det_size: int):
    """Background scanner thread — ใช้ logic เดียวกับ faculty_scanner.py"""
    try:
        update_state(
            status="loading_model",
            message="กำลังโหลดโมเดล InsightFace (buffalo_l)...",
            start_time=time.time(),
            matches=[], log_lines=[], errors=0,
            pages_crawled=0, images_found=0, images_processed=0,
            faces_detected=0, matches_found=0, progress=0, total=0,
        )
        add_log("JARVIS: กำลังเริ่มระบบ...")

        # ── Load InsightFace ──
        try:
            from insightface.app import FaceAnalysis
            face_app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "recognition"],
            )
            face_app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=0.5)
            add_log(f"JARVIS: โมเดลพร้อมใช้งาน (det_size={det_size})")
        except Exception as e:
            err_msg = f"โหลดโมเดลไม่ได้: {str(e)[:100]}"
            update_state(status="error", message=err_msg)
            add_log(f"ERROR: {err_msg}")
            import traceback; traceback.print_exc()
            return

        if stop_event.is_set():
            update_state(status="idle", message="ถูกยกเลิก")
            return

        # ── Build Target Profile ──
        update_state(status="building_profile", message="กำลังสร้าง Target Profile...")
        add_log("JARVIS: วิเคราะห์รูปตัวอย่าง...")

        image_files = [
            f for f in sorted(SAMPLES_DIR.iterdir())
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        add_log(f"JARVIS: พบรูปตัวอย่าง {len(image_files)} ไฟล์ ใน {SAMPLES_DIR}")
        if not image_files:
            update_state(status="error", message="ไม่พบรูปตัวอย่างใน teacher_samples/")
            add_log(f"ERROR: ไม่พบรูปตัวอย่าง! path={SAMPLES_DIR.resolve()}")
            return

        embeddings = []
        for img_path in image_files:
            try:
                add_log(f"JARVIS: กำลังอ่าน {img_path.name}...")
                
                # อ่านไฟล์รองรับภาษาไทยใน Path
                img_array = np.fromfile(str(img_path), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    add_log(f"WARNING: โหลดรูปภาพล้มเหลว: {img_path.name}")
                    continue
                add_log(f"JARVIS: {img_path.name} — size={img.shape[1]}x{img.shape[0]}")
                faces = face_app.get(img)
                add_log(f"JARVIS: {img_path.name} — พบ {len(faces)} ใบหน้า")
                if not faces:
                    add_log(f"WARNING: ตรวจจับใบหน้าไม่ได้ใน {img_path.name}")
                    continue
                largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                embeddings.append(largest.normed_embedding)
                add_log(f"JARVIS: {img_path.name} — score={largest.det_score:.3f} ✓")
            except Exception as e:
                add_log(f"ERROR: ประมวลผล {img_path.name} ล้มเหลว: {str(e)[:80]}")
                import traceback; traceback.print_exc()
                continue

        if not embeddings:
            update_state(status="error", message="ไม่สามารถสร้าง Target Profile ได้")
            return

        mean_emb = np.mean(embeddings, axis=0)
        target = mean_emb / np.linalg.norm(mean_emb)
        add_log(f"JARVIS: Target Profile สร้างเสร็จ ({len(embeddings)} รูป)")

        if stop_event.is_set():
            update_state(status="idle", message="ถูกยกเลิก")
            return

        # ── Crawling ──
        update_state(status="crawling", message=f"กำลัง Crawl {allowed_domain}...")
        add_log(f"JARVIS: เริ่ม Crawl จาก {start_url[:60]}...")

        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
        })

        visited = set()
        image_urls = {}  # img_url -> {"source_page": url, "page_title": title}

        def crawl(url, depth=0):
            if stop_event.is_set():
                return
            normalized = url.rstrip("/").split("#")[0]
            if normalized in visited or depth > max_depth:
                return
            visited.add(normalized)

            try:
                r = session.get(url, timeout=15, allow_redirects=True)
                r.raise_for_status()
                ct = r.headers.get("Content-Type", "")
                if "text/html" not in ct and "application/xhtml" not in ct:
                    return
            except Exception:
                return

            try:
                soup = BeautifulSoup(r.text, "lxml")
            except Exception:
                soup = BeautifulSoup(r.text, "html.parser")

            # ดึง page title สำหรับระบุที่มา
            title_tag = soup.find("title")
            page_title = title_tag.text.strip() if title_tag else url

            # ดึง <img>
            new_count = 0
            for img_tag in soup.find_all("img"):
                for attr in ("src", "data-src", "data-original", "data-lazy-src"):
                    raw = img_tag.get(attr, "")
                    if raw and not raw.startswith("data:"):
                        abs_url = urljoin(url, raw.strip())
                        p = urlparse(abs_url).path.lower()
                        if any(p.endswith(e) for e in SUPPORTED_IMAGE_EXTENSIONS):
                            if abs_url not in image_urls:
                                image_urls[abs_url] = {
                                    "source_page": url,
                                    "page_title": page_title,
                                }
                                new_count += 1
                srcset = img_tag.get("srcset", "")
                if srcset:
                    for entry in srcset.split(","):
                        parts = entry.strip().split()
                        if parts and not parts[0].startswith("data:"):
                            abs_url = urljoin(url, parts[0].strip())
                            p = urlparse(abs_url).path.lower()
                            if any(p.endswith(e) for e in SUPPORTED_IMAGE_EXTENSIONS):
                                if abs_url not in image_urls:
                                    image_urls[abs_url] = {
                                        "source_page": url,
                                        "page_title": page_title,
                                    }
                                    new_count += 1

            update_state(pages_crawled=len(visited), images_found=len(image_urls))
            add_log(f"CRAWL [d={depth}] {url[:60]}... +{new_count} imgs (total: {len(image_urls)})")
            time.sleep(0.3)

            # Follow links
            if depth < max_depth:
                links = set()
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    if not href or href.startswith("#"):
                        continue
                    abs_link = urljoin(url, href)
                    parsed = urlparse(abs_link)
                    if parsed.netloc == allowed_domain:
                        skip_exts = {".pdf", ".doc", ".docx", ".xls", ".xlsx",
                                     ".zip", ".mp4", ".mp3"}
                        if not any(parsed.path.lower().endswith(e) for e in skip_exts):
                            links.add(abs_link)
                for link in sorted(links):
                    if stop_event.is_set():
                        return
                    crawl(link, depth + 1)

        crawl(start_url, 0)
        add_log(f"JARVIS: Crawl เสร็จ — {len(visited)} หน้า, {len(image_urls)} รูป")

        if not image_urls or stop_event.is_set():
            update_state(status="done" if not stop_event.is_set() else "idle",
                         message="ไม่พบรูปภาพ" if not image_urls else "ถูกยกเลิก")
            return

        # ── Download + Detect + Match ──
        update_state(status="scanning", message="กำลังสแกนใบหน้า...",
                     total=len(image_urls), progress=0)
        add_log(f"JARVIS: เริ่มสแกน {len(image_urls)} รูป...")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def download_image(url_str):
            try:
                resp = session.get(url_str, timeout=15)
                if len(resp.content) < 500 or len(resp.content) > 20*1024*1024:
                    return None
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    return None
                h, w = img.shape[:2]
                if h < 50 or w < 50:
                    return None
                return img
            except Exception:
                return None

        processed = 0
        with ThreadPoolExecutor(max_workers=12) as executor:
            future_to_url = {
                executor.submit(download_image, u): (u, info)
                for u, info in image_urls.items()
            }
            for future in as_completed(future_to_url):
                if stop_event.is_set():
                    break
                img_url, img_info = future_to_url[future]
                processed += 1

                try:
                    img = future.result()
                except Exception:
                    with scanner_lock:
                        scanner_state["errors"] += 1
                    continue

                if img is None:
                    update_state(progress=processed)
                    continue

                with scanner_lock:
                    scanner_state["images_processed"] += 1

                try:
                    faces = face_app.get(img)
                except Exception:
                    with scanner_lock:
                        scanner_state["errors"] += 1
                    update_state(progress=processed)
                    continue

                if not faces:
                    update_state(progress=processed)
                    continue

                with scanner_lock:
                    scanner_state["faces_detected"] += len(faces)

                for fi, face in enumerate(faces):
                    sim = float(np.dot(face.normed_embedding, target))
                    if sim >= similarity_threshold:
                        # Annotate (JARVIS Style)
                        annotated = img.copy()
                        h, w = annotated.shape[:2]
                        x1, y1, x2, y2 = face.bbox.astype(int)

                        # ขยายกรอบออกด้านนอก 10px
                        pad = 10
                        bx1 = max(0, x1 - pad)
                        by1 = max(0, y1 - pad)
                        bx2 = min(w, x2 + pad)
                        by2 = min(h, y2 + pad)

                        color = (0, 255, 200)  # cyan-green
                        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2)

                        # มุม HUD
                        cl = min(20, (bx2 - bx1) // 4, (by2 - by1) // 4)
                        cv2.line(annotated, (bx1, by1), (bx1 + cl, by1), color, 3)
                        cv2.line(annotated, (bx1, by1), (bx1, by1 + cl), color, 3)
                        cv2.line(annotated, (bx2, by1), (bx2 - cl, by1), color, 3)
                        cv2.line(annotated, (bx2, by1), (bx2, by1 + cl), color, 3)
                        cv2.line(annotated, (bx1, by2), (bx1 + cl, by2), color, 3)
                        cv2.line(annotated, (bx1, by2), (bx1, by2 - cl), color, 3)
                        cv2.line(annotated, (bx2, by2), (bx2 - cl, by2), color, 3)
                        cv2.line(annotated, (bx2, by2), (bx2, by2 - cl), color, 3)

                        label = f"{sim:.1%}"
                        face_w = bx2 - bx1
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fs = max(0.5, min(0.9, face_w / 150))
                        th = 1 if fs < 0.7 else 2
                        (tw, th_), _ = cv2.getTextSize(label, font, fs, th)

                        ly = by1 - 8
                        if ly - th_ < 5: ly = by2 + th_ + 8
                        
                        overlay = annotated.copy()
                        cv2.rectangle(overlay, (bx1, ly - th_ - 4), (bx1 + tw + 8, ly + 4), (0, 0, 0), cv2.FILLED)
                        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
                        cv2.putText(annotated, label, (bx1 + 4, ly), font, fs, color, th)

                        url_hash = hashlib.md5(img_url.encode()).hexdigest()[:10]
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"match_{ts}_{url_hash}_f{fi}.jpg"

                        # Clean page title
                        page_title = img_info.get("page_title", "")
                        # Remove site name suffix
                        page_title = re.sub(r'\s*[-–|]\s*มหาวิทยาลัย.*$', '', page_title).strip()
                        if not page_title:
                            page_title = "ไม่ระบุ"

                        # สร้างโฟลเดอร์ย่อยตามชื่องาน/กิจกรรม
                        folder_name = sanitize_folder_name(page_title)
                        event_dir = OUTPUT_DIR / folder_name
                        event_dir.mkdir(parents=True, exist_ok=True)

                        # Save file (รองรับโฟลเดอร์ภาษาไทยบน Windows)
                        is_success, buffer = cv2.imencode(".jpg", annotated)
                        if is_success:
                            with open(event_dir / filename, "wb") as f:
                                f.write(buffer)

                        # เก็บ relative path เป็น folder/filename
                        rel_path = f"{folder_name}/{filename}"

                        match_record = {
                            "filename": filename,
                            "folder": folder_name,
                            "rel_path": rel_path,
                            "similarity": round(sim, 4),
                            "image_url": img_url,
                            "source_page": img_info.get("source_page", ""),
                            "page_title": page_title,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        with scanner_lock:
                            scanner_state["matches"].append(match_record)
                            scanner_state["matches_found"] += 1
                        add_log(f"MATCH! sim={sim:.4f} | {page_title[:40]}")

                update_state(progress=processed)

                if processed % 50 == 0:
                    add_log(f"SCAN: {processed}/{len(image_urls)} | faces={scanner_state['faces_detected']} | match={scanner_state['matches_found']}")

        # ── Export CSV ──
        if scanner_state["matches"]:
            with open(REPORT_FILE, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["Filename", "Similarity", "Image_URL", "Source_Page", "Page_Title", "Timestamp"])
                for m in sorted(scanner_state["matches"], key=lambda x: x["similarity"], reverse=True):
                    writer.writerow([m["filename"], m["similarity"], m["image_url"],
                                     m["source_page"], m["page_title"], m["timestamp"]])

        update_state(status="done",
                     message=f"สแกนเสร็จ! พบ {scanner_state['matches_found']} รูปที่ตรงกัน")
        add_log(f"JARVIS: สแกนเสร็จสมบูรณ์ — Match: {scanner_state['matches_found']}")

    except Exception as e:
        update_state(status="error", message=f"เกิดข้อผิดพลาด: {str(e)[:100]}")
        add_log(f"ERROR: {str(e)[:200]}")
        import traceback
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════
#  Flask Routes — API
# ══════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with scanner_lock:
        return jsonify(scanner_state)


@app.route("/api/samples", methods=["GET"])
def api_get_samples():
    """รายการรูปตัวอย่างที่มีอยู่"""
    samples = []
    for f in sorted(SAMPLES_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            samples.append({"name": f.name, "size": f.stat().st_size})
    return jsonify({"samples": samples})


@app.route("/api/samples/upload", methods=["POST"])
def api_upload_sample():
    """อัปโหลดรูปตัวอย่าง"""
    if "files" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์"}), 400

    uploaded = []
    for f in request.files.getlist("files"):
        if f.filename:
            ext = Path(f.filename).suffix.lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                save_path = SAMPLES_DIR / f.filename
                f.save(str(save_path))
                uploaded.append(f.filename)

    return jsonify({"uploaded": uploaded, "count": len(uploaded)})


@app.route("/api/samples/delete/<filename>", methods=["DELETE"])
def api_delete_sample(filename):
    """ลบรูปตัวอย่าง"""
    fp = SAMPLES_DIR / filename
    if fp.exists():
        fp.unlink()
        return jsonify({"deleted": filename})
    return jsonify({"error": "ไม่พบไฟล์"}), 404


@app.route("/api/samples/image/<filename>")
def api_sample_image(filename):
    """ส่งรูปตัวอย่าง"""
    return send_from_directory(str(SAMPLES_DIR), filename)


@app.route("/api/results/image/<path:folder>/<filename>")
def api_result_image(folder, filename):
    """ส่งรูปผลลัพธ์ (รองรับโฟลเดอร์ย่อย)"""
    return send_from_directory(str(OUTPUT_DIR / folder), filename)

@app.route("/api/results/image_legacy/<filename>")
def api_result_image_legacy(filename):
    """ส่งรูปผลลัพธ์แบบเก่า (เผื่อมีรูปเก่าค้างในโฟลเดอร์หลัก)"""
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route("/api/scan/start", methods=["POST"])
def api_start_scan():
    """เริ่มสแกน"""
    global scanner_thread
    if scanner_state["status"] not in ("idle", "done", "error"):
        return jsonify({"error": "Scanner กำลังทำงานอยู่"}), 400

    data = request.get_json(force=True)
    start_url = data.get("start_url", "https://www.dusit.ac.th")
    allowed_domain = data.get("allowed_domain", "")
    if not allowed_domain:
        allowed_domain = urlparse(start_url).netloc
    max_depth = int(data.get("max_depth", 1))
    threshold = float(data.get("threshold", 0.40))
    det_size_val = int(data.get("det_size", 1280))

    # Clear old results
    for item in OUTPUT_DIR.iterdir():
        if item.name == ".gitkeep":
            continue
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    stop_event.clear()
    scanner_thread = threading.Thread(
        target=run_scanner,
        args=(start_url, allowed_domain, max_depth, threshold, det_size_val),
        daemon=True,
    )
    scanner_thread.start()
    return jsonify({"message": "Scanner started"})


@app.route("/api/scan/stop", methods=["POST"])
def api_stop_scan():
    """หยุดสแกน"""
    stop_event.set()
    update_state(status="idle", message="กำลังหยุด...")
    return jsonify({"message": "กำลังหยุด Scanner..."})


@app.route("/api/results")
def api_results():
    """ส่งรายการผลลัพธ์"""
    with scanner_lock:
        matches = sorted(scanner_state["matches"], key=lambda x: x["similarity"], reverse=True)
    return jsonify({"matches": matches, "total": len(matches)})


@app.route("/api/results/clear", methods=["POST"])
def api_clear_results():
    """ล้างผลลัพธ์"""
    for item in OUTPUT_DIR.iterdir():
        if item.name == ".gitkeep":
            continue
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    with scanner_lock:
        scanner_state["matches"] = []
        scanner_state["matches_found"] = 0
    return jsonify({"message": "ล้างผลลัพธ์เรียบร้อย"})


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print()
    print("=" * 56)
    print("   J.A.R.V.I.S. — Faculty Face Scanner Web UI")
    print("   Open browser: http://localhost:5000")
    print("=" * 56)
    print()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
