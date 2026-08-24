#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  Faculty Face Scanner — End-to-End Face Recognition Web Crawler    ║
║                                                                    ║
║  ระบบสแกนใบหน้าอัตโนมัติสำหรับค้นหาภาพอาจารย์จากเว็บไซต์มหาวิทยาลัย   ║
║  ใช้ InsightFace (ArcFace + SCRFD) บน NVIDIA GPU                    ║
╚══════════════════════════════════════════════════════════════════════╝

Pipeline 4 ขั้นตอน:
  1. Target Embedding  — สร้างโปรไฟล์ใบหน้าจากรูปตัวอย่าง
  2. Domain Crawling   — ไล่เก็บ URL รูปภาพจากเว็บไซต์เป้าหมาย
  3. Image Processing  — ดาวน์โหลด + ตรวจจับใบหน้าแบบขนาน (GPU)
  4. Matching & Output — เทียบความคล้าย + บันทึกผลลัพธ์

วิธีใช้:
  1. ใส่รูปอาจารย์ 1-3 รูป ไว้ใน ./teacher_samples/
  2. แก้ START_URL ให้ตรงกับเว็บไซต์เป้าหมาย
  3. รัน: python faculty_scanner.py
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup

# ══════════════════════════════════════════════════════════════════════
#  ค่าคอนฟิกหลัก — แก้ตรงนี้ตามต้องการ
# ══════════════════════════════════════════════════════════════════════
START_URL: str = "https://www.dusit.ac.th/home/?s=%E0%B8%A7%E0%B8%B4%E0%B8%8A%E0%B8%8A%E0%B8%B2+%E0%B8%89%E0%B8%B4%E0%B8%A1%E0%B8%9E%E0%B8%A5%E0%B8%B5"
ALLOWED_DOMAIN: str = "www.dusit.ac.th"
MAX_CRAWL_DEPTH: int = 1            # ความลึกของการ crawl (1 = เข้าไปดูในข่าวด้วย)
CRAWL_DELAY: float = 0.5            # หน่วงระหว่างดึงแต่ละหน้า (วินาที)
SIMILARITY_THRESHOLD: float = 0.40  # เกณฑ์ความคล้ายขั้นต่ำ (0.0 - 1.0) - ลดลงเพื่อจับหน้าเล็ก/เบลอ
MAX_WORKERS: int = 16               # จำนวน thread สำหรับดาวน์โหลดรูป
DET_SIZE: tuple[int, int] = (1280, 1280)  # ขนาด input ของ face detector
SAMPLES_DIR: str = "./teacher_samples"
OUTPUT_DIR: str = "./matched_results"
REPORT_FILE: str = "scan_report.csv"
REQUEST_TIMEOUT: int = 15           # timeout สำหรับ HTTP request (วินาที)
MIN_IMAGE_SIZE: int = 50            # ขนาดรูปต่ำสุดที่จะประมวลผล (พิกเซล)
USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# นามสกุลไฟล์รูปที่รองรับ
SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
)

# ══════════════════════════════════════════════════════════════════════
#  ตั้งค่า Logging
# ══════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("FacultyScanner")


# ══════════════════════════════════════════════════════════════════════
#  Data Classes
# ══════════════════════════════════════════════════════════════════════
@dataclass
class MatchRecord:
    """เก็บข้อมูลของแต่ละ match ที่เจอ"""
    filename: str
    similarity: float
    image_url: str
    source_page: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class CrawlStats:
    """สถิติการทำงานทั้งหมด"""
    pages_crawled: int = 0
    images_found: int = 0
    images_processed: int = 0
    faces_detected: int = 0
    matches_found: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        """คำนวณเวลาที่ผ่านไป"""
        delta = time.time() - self.start_time
        minutes, seconds = divmod(int(delta), 60)
        return f"{minutes:02d}:{seconds:02d}"


# ══════════════════════════════════════════════════════════════════════
#  Phase 1 — สร้าง Target Embedding จากรูปตัวอย่าง
# ══════════════════════════════════════════════════════════════════════
def initialize_face_engine():
    """
    โหลดโมเดล InsightFace (buffalo_l) พร้อม CUDA acceleration
    
    ใช้เฉพาะโมดูล detection + recognition เพื่อประหยัด VRAM
    กรณี GPU ไม่พร้อม จะ fallback เป็น CPU พร้อมแจ้งเตือน
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        log.error("❌ ไม่พบ insightface — กรุณารัน: pip install insightface")
        sys.exit(1)

    # ตรวจสอบ CUDA provider
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            log.info("🟢 GPU พร้อมใช้งาน — CUDAExecutionProvider")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            log.warning(
                "⚠️  CUDAExecutionProvider ไม่พร้อม — จะใช้ CPU แทน (ช้ากว่ามาก)\n"
                "   Providers ที่มี: %s", available
            )
            providers = ["CPUExecutionProvider"]
    except ImportError:
        log.warning("⚠️  ไม่พบ onnxruntime — ใช้ค่า default ของ InsightFace")
        providers = None

    # โหลด FaceAnalysis เฉพาะ detection + recognition
    log.info("📦 กำลังโหลดโมเดล InsightFace (buffalo_l)...")
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    app.prepare(ctx_id=0, det_size=DET_SIZE, det_thresh=0.5)
    log.info("✅ โมเดลพร้อมใช้งาน — det_size=%s", DET_SIZE)
    return app


def build_target_profile(face_app) -> np.ndarray:
    """
    สร้าง target embedding จากรูปตัวอย่างใน SAMPLES_DIR
    
    ขั้นตอน:
    1. โหลดรูปทั้งหมดจาก teacher_samples/
    2. ตรวจจับใบหน้าในแต่ละรูป → เลือกใบหน้าที่ใหญ่ที่สุด
    3. ดึง embedding 512 มิติ จากแต่ละรูป
    4. คำนวณค่าเฉลี่ยแล้ว L2-normalize → target_profile
    
    Returns:
        np.ndarray: L2-normalized mean embedding (512,)
    """
    samples_path = Path(SAMPLES_DIR)
    if not samples_path.exists():
        log.error("❌ ไม่พบโฟลเดอร์ %s — กรุณาสร้างแล้วใส่รูปอาจารย์", SAMPLES_DIR)
        sys.exit(1)

    # หาไฟล์รูปทั้งหมด
    image_files = [
        f for f in sorted(samples_path.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not image_files:
        log.error(
            "❌ ไม่พบรูปภาพใน %s\n"
            "   รองรับนามสกุล: %s\n"
            "   กรุณาใส่รูปอาจารย์ 1-3 รูป",
            SAMPLES_DIR, ", ".join(SUPPORTED_IMAGE_EXTENSIONS),
        )
        sys.exit(1)

    log.info("📸 พบรูปตัวอย่าง %d รูป: %s", len(image_files), [f.name for f in image_files])
    embeddings: list[np.ndarray] = []

    for img_path in image_files:
        # อ่านรูปเป็น BGR (ตามที่ InsightFace ต้องการ)
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("⚠️  อ่านรูปไม่ได้: %s — ข้าม", img_path.name)
            continue

        faces = face_app.get(img)
        if not faces:
            log.warning("⚠️  ตรวจจับใบหน้าไม่ได้ใน: %s — ข้าม", img_path.name)
            continue

        # เลือกใบหน้าที่ใหญ่ที่สุด (กรณีรูปหมู่)
        largest_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        emb = largest_face.normed_embedding  # (512,) L2-normalized แล้ว
        embeddings.append(emb)
        log.info(
            "   ✓ %s — det_score=%.3f, bbox_area=%d px²",
            img_path.name,
            largest_face.det_score,
            int((largest_face.bbox[2] - largest_face.bbox[0]) *
                (largest_face.bbox[3] - largest_face.bbox[1])),
        )

    if not embeddings:
        log.error("❌ ไม่สามารถสร้าง target profile ได้ — ไม่มีใบหน้าที่ตรวจจับได้เลย")
        sys.exit(1)

    # คำนวณค่าเฉลี่ย → L2-normalize
    mean_embedding = np.mean(embeddings, axis=0)
    target_profile = mean_embedding / np.linalg.norm(mean_embedding)

    log.info(
        "🎯 Target profile สร้างเสร็จ — ใช้ %d/%d รูป, embedding norm=%.6f",
        len(embeddings), len(image_files), np.linalg.norm(target_profile),
    )
    return target_profile


# ══════════════════════════════════════════════════════════════════════
#  Phase 2 — Recursive Domain Crawling
# ══════════════════════════════════════════════════════════════════════
class DomainCrawler:
    """
    ไล่เก็บ URL รูปภาพจากเว็บไซต์เป้าหมายแบบ recursive
    
    หลักการทำงาน:
    - เริ่มจาก START_URL → parse HTML → เก็บ <img> src
    - ไล่ตาม <a href> ที่อยู่ใน domain เดียวกัน
    - จำกัดความลึกด้วย MAX_CRAWL_DEPTH
    - ใช้ visited_urls set ป้องกัน infinite loop
    """

    def __init__(self):
        self.visited_urls: set[str] = set()
        self.image_urls: dict[str, str] = {}  # image_url → source_page
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/*;q=0.8",
            "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
        })

    def _normalize_url(self, url: str) -> str:
        """ทำให้ URL เป็นรูปแบบมาตรฐาน — ตัด fragment, trailing slash"""
        parsed = urlparse(url)
        # ตัด fragment (#...) ออก
        clean = parsed._replace(fragment="")
        normalized = clean.geturl().rstrip("/")
        return normalized

    def _is_same_domain(self, url: str) -> bool:
        """ตรวจว่า URL อยู่ใน domain เป้าหมายหรือไม่"""
        try:
            parsed = urlparse(url)
            return parsed.netloc == ALLOWED_DOMAIN or parsed.netloc == ""
        except Exception:
            return False

    def _is_valid_page_url(self, url: str) -> bool:
        """กรองเฉพาะ URL ที่น่าจะเป็นหน้าเว็บ HTML"""
        parsed = urlparse(url)
        # ข้ามลิงก์พิเศษ
        if parsed.scheme in ("mailto", "tel", "javascript", "data"):
            return False
        # ข้ามไฟล์ที่ไม่ใช่ HTML
        path_lower = parsed.path.lower()
        skip_exts = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
                     ".zip", ".rar", ".mp4", ".mp3", ".avi", ".mov", ".wmv"}
        if any(path_lower.endswith(ext) for ext in skip_exts):
            return False
        return True

    def _extract_image_urls(self, soup: BeautifulSoup, page_url: str) -> list[str]:
        """
        ดึง URL รูปภาพจาก HTML — รองรับหลายแหล่ง:
        - <img src="...">
        - <img data-src="..."> (lazy loading)
        - <img srcset="..."> (responsive images)
        - <source srcset="..."> ใน <picture>
        """
        found: list[str] = []

        for img_tag in soup.find_all("img"):
            # ดึงจาก src, data-src, data-original (lazy load patterns)
            for attr in ("src", "data-src", "data-original", "data-lazy-src"):
                raw = img_tag.get(attr, "")
                if raw and not raw.startswith("data:"):
                    abs_url = urljoin(page_url, raw.strip())
                    found.append(abs_url)

            # ดึงจาก srcset (เอา URL ตัวแรก / ใหญ่สุด)
            srcset = img_tag.get("srcset", "")
            if srcset:
                for entry in srcset.split(","):
                    parts = entry.strip().split()
                    if parts and not parts[0].startswith("data:"):
                        abs_url = urljoin(page_url, parts[0].strip())
                        found.append(abs_url)

        # <source> ใน <picture>
        for source_tag in soup.find_all("source"):
            srcset = source_tag.get("srcset", "")
            if srcset:
                for entry in srcset.split(","):
                    parts = entry.strip().split()
                    if parts and not parts[0].startswith("data:"):
                        abs_url = urljoin(page_url, parts[0].strip())
                        found.append(abs_url)

        # กรองเฉพาะนามสกุลรูปที่รองรับ
        valid: list[str] = []
        for url in found:
            parsed_path = urlparse(url).path.lower()
            # ตรวจนามสกุลไฟล์ หรือ URL ที่มี image ใน path
            if any(parsed_path.endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS):
                valid.append(url)
            elif re.search(r"\.(jpg|jpeg|png|webp)", url, re.IGNORECASE):
                valid.append(url)

        return valid

    def crawl(self, url: str, depth: int = 0) -> None:
        """
        Recursive crawling — ดึงรูปภาพและลิงก์จากหน้าเว็บ
        
        Args:
            url: URL ของหน้าเว็บที่จะ crawl
            depth: ระดับความลึกปัจจุบัน
        """
        normalized = self._normalize_url(url)
        if normalized in self.visited_urls:
            return
        if depth > MAX_CRAWL_DEPTH:
            return

        self.visited_urls.add(normalized)

        try:
            response = self.session.get(
                url,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            response.raise_for_status()

            # ตรวจว่าเป็น HTML จริง
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return

        except requests.RequestException as e:
            log.debug("   ⚠️  ดึงหน้า %s ไม่ได้: %s", url[:80], str(e)[:60])
            return

        try:
            soup = BeautifulSoup(response.text, "lxml")
        except Exception:
            soup = BeautifulSoup(response.text, "html.parser")

        # ดึง URL รูปภาพ
        page_images = self._extract_image_urls(soup, url)
        new_images = 0
        for img_url in page_images:
            if img_url not in self.image_urls:
                self.image_urls[img_url] = url
                new_images += 1

        log.info(
            "🌐 [depth=%d] %s — พบรูป %d (ใหม่ %d) | รวมทั้งหมด %d รูป",
            depth, url[:70], len(page_images), new_images, len(self.image_urls),
        )

        # หน่วงเพื่อไม่ให้โหลดเซิร์ฟเวอร์หนัก
        time.sleep(CRAWL_DELAY)

        # ไล่ตามลิงก์ภายใน domain เดียวกัน
        if depth < MAX_CRAWL_DEPTH:
            links: list[str] = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                if not href or href.startswith("#"):
                    continue
                abs_link = urljoin(url, href)
                if self._is_same_domain(abs_link) and self._is_valid_page_url(abs_link):
                    links.append(abs_link)

            # เรียง + deduplicate ก่อน crawl
            for link in sorted(set(links)):
                self.crawl(link, depth + 1)


# ══════════════════════════════════════════════════════════════════════
#  Phase 3 & 4 — ดาวน์โหลดรูป, ตรวจจับใบหน้า, เทียบความคล้าย
# ══════════════════════════════════════════════════════════════════════
class FaceMatchProcessor:
    """
    ดาวน์โหลดรูปภาพแบบขนาน → ตรวจจับใบหน้า → เทียบกับ target profile
    
    ใช้ ThreadPoolExecutor สำหรับ download I/O
    การ inference บน GPU ทำใน main thread เพื่อหลีกเลี่ยง CUDA context issues
    """

    def __init__(self, face_app, target_profile: np.ndarray, stats: CrawlStats):
        self.face_app = face_app
        self.target_profile = target_profile
        self.stats = stats
        self.matches: list[MatchRecord] = []
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _download_image(self, url: str) -> Optional[tuple[str, np.ndarray]]:
        """
        ดาวน์โหลดรูปภาพจาก URL → decode เป็น numpy array (BGR)
        
        Returns:
            tuple(url, image_array) หรือ None ถ้าดาวน์โหลด/decode ไม่ได้
        """
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()

            # ตรวจขนาดไฟล์ (ข้ามไฟล์ > 20MB)
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > 20 * 1024 * 1024:
                return None

            # อ่านข้อมูลทั้งหมด
            img_bytes = resp.content
            if len(img_bytes) < 100:  # ไฟล์เล็กเกินไป
                return None

            # Decode เป็น numpy array
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                return None

            # ตรวจขนาดขั้นต่ำ
            h, w = img.shape[:2]
            if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
                return None

            return (url, img)

        except Exception:
            return None

    def _compute_similarity(self, embedding: np.ndarray) -> float:
        """
        คำนวณ cosine similarity ระหว่าง embedding กับ target profile
        
        เนื่องจากทั้งสอง vector ผ่าน L2-normalize แล้ว
        cosine similarity = dot product
        """
        return float(np.dot(embedding, self.target_profile))

    def _generate_filename(self, url: str, face_idx: int) -> str:
        """สร้างชื่อไฟล์ unique จาก URL hash"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"match_{timestamp}_{url_hash}_face{face_idx}.jpg"

    def _annotate_image(
        self,
        img: np.ndarray,
        bbox: np.ndarray,
        similarity: float,
    ) -> np.ndarray:
        """
        วาด bounding box รอบใบหน้า — ขยายกรอบออกด้านนอก
        เพื่อไม่ให้บดบังใบหน้า + แสดงคะแนนเหนือกรอบ
        """
        annotated = img.copy()
        h, w = annotated.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)

        # ขยายกรอบออกด้านนอก 10px (ไม่ให้เส้นทับหน้า)
        pad = 10
        bx1 = max(0, x1 - pad)
        by1 = max(0, y1 - pad)
        bx2 = min(w, x2 + pad)
        by2 = min(h, y2 + pad)

        # เส้นกรอบบาง สีฟ้าเรืองแสง (สไตล์ JARVIS)
        color = (0, 255, 200)  # cyan-green
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2)

        # วาดมุมเน้นสไตล์ HUD (เส้นมุมยาวขึ้น)
        corner_len = min(20, (bx2 - bx1) // 4, (by2 - by1) // 4)
        cv2.line(annotated, (bx1, by1), (bx1 + corner_len, by1), color, 3)
        cv2.line(annotated, (bx1, by1), (bx1, by1 + corner_len), color, 3)
        cv2.line(annotated, (bx2, by1), (bx2 - corner_len, by1), color, 3)
        cv2.line(annotated, (bx2, by1), (bx2, by1 + corner_len), color, 3)
        cv2.line(annotated, (bx1, by2), (bx1 + corner_len, by2), color, 3)
        cv2.line(annotated, (bx1, by2), (bx1, by2 - corner_len), color, 3)
        cv2.line(annotated, (bx2, by2), (bx2 - corner_len, by2), color, 3)
        cv2.line(annotated, (bx2, by2), (bx2, by2 - corner_len), color, 3)

        # ป้ายคะแนนอยู่เหนือกรอบ (ไม่ทับหน้า)
        label = f"{similarity:.1%}"
        face_w = bx2 - bx1
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(0.9, face_w / 150))
        thickness = 1 if font_scale < 0.7 else 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = by1 - 8
        if label_y - th < 5:
            label_y = by2 + th + 8  # ถ้าไม่มีที่ด้านบน ใส่ด้านล่างแทน

        # พื้นหลังป้าย (โปร่งแสง)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (bx1, label_y - th - 4), (bx1 + tw + 8, label_y + 4), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        cv2.putText(annotated, label, (bx1 + 4, label_y), font, font_scale, color, thickness)

        return annotated

    def process_batch(self, image_urls: dict[str, str]) -> None:
        """
        ประมวลผลรูปภาพทั้งหมดแบบขนาน
        
        ขั้นตอน:
        1. ThreadPoolExecutor ดาวน์โหลดรูปหลาย thread พร้อมกัน
        2. ส่งรูปที่ decode ได้ไปยัง InsightFace (GPU) ทีละรูป
        3. เทียบใบหน้าแต่ละหน้ากับ target profile
        4. ถ้า similarity >= threshold → บันทึกรูป + CSV
        
        Args:
            image_urls: dict ของ {image_url: source_page_url}
        """
        total = len(image_urls)
        log.info("⬇️  เริ่มดาวน์โหลดและประมวลผลรูปภาพ %d รูป (workers=%d)...", total, MAX_WORKERS)

        url_list = list(image_urls.items())
        processed = 0

        # ดาวน์โหลดเป็น batch แล้ว process ทีละรูปบน GPU
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit ทุก download task พร้อมกัน
            future_to_info = {
                executor.submit(self._download_image, img_url): (img_url, src_page)
                for img_url, src_page in url_list
            }

            for future in as_completed(future_to_info):
                img_url, source_page = future_to_info[future]
                processed += 1

                try:
                    result = future.result()
                except Exception:
                    self.stats.errors += 1
                    continue

                if result is None:
                    continue

                _, img = result
                self.stats.images_processed += 1

                # ── Face Detection (GPU) ──
                try:
                    faces = self.face_app.get(img)
                except Exception as e:
                    log.debug("   ⚠️  ตรวจจับใบหน้าล้มเหลว: %s — %s", img_url[:50], e)
                    self.stats.errors += 1
                    continue

                if not faces:
                    continue

                self.stats.faces_detected += len(faces)

                # ── เทียบแต่ละใบหน้า ──
                for face_idx, face in enumerate(faces):
                    emb = face.normed_embedding
                    sim = self._compute_similarity(emb)

                    if sim >= SIMILARITY_THRESHOLD:
                        self.stats.matches_found += 1

                        # วาด bounding box + บันทึก
                        annotated = self._annotate_image(img, face.bbox, sim)
                        filename = self._generate_filename(img_url, face_idx)
                        save_path = self.output_dir / filename
                        cv2.imwrite(str(save_path), annotated)

                        record = MatchRecord(
                            filename=filename,
                            similarity=round(sim, 4),
                            image_url=img_url,
                            source_page=source_page,
                        )
                        self.matches.append(record)

                        log.info(
                            "   🎯 MATCH! sim=%.4f | %s | face#%d",
                            sim, img_url[:60], face_idx,
                        )

                # แสดง progress ทุก 50 รูป
                if processed % 50 == 0 or processed == total:
                    log.info(
                        "   📊 Progress: %d/%d รูป | ใบหน้า: %d | Match: %d | เวลา: %s",
                        processed, total, self.stats.faces_detected,
                        self.stats.matches_found, self.stats.elapsed(),
                    )

    def export_report(self) -> None:
        """บันทึกผลลัพธ์ลง CSV"""
        if not self.matches:
            log.info("📝 ไม่มี match — ข้าม CSV report")
            return

        report_path = Path(REPORT_FILE)
        with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Filename", "Similarity", "Image_URL", "Source_Page", "Timestamp"])
            for m in sorted(self.matches, key=lambda x: x.similarity, reverse=True):
                writer.writerow([m.filename, m.similarity, m.image_url, m.source_page, m.timestamp])

        log.info("📝 บันทึก CSV report เสร็จ: %s (%d records)", report_path, len(self.matches))


# ══════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════════════════
def print_banner() -> None:
    """แสดง banner เริ่มต้น"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║         🔍  Faculty Face Scanner v1.0                       ║
║         ระบบค้นหาใบหน้าอาจารย์อัตโนมัติ                         ║
╚══════════════════════════════════════════════════════════════╝"""
    print(banner)
    print(f"  🌐 Target URL       : {START_URL}")
    print(f"  📏 Crawl Depth      : {MAX_CRAWL_DEPTH}")
    print(f"  🎯 Threshold        : {SIMILARITY_THRESHOLD}")
    print(f"  🖥️  Detection Size   : {DET_SIZE}")
    print(f"  👥 Workers          : {MAX_WORKERS}")
    print(f"  📂 Samples Dir      : {SAMPLES_DIR}")
    print(f"  💾 Output Dir       : {OUTPUT_DIR}")
    print()


def print_summary(stats: CrawlStats, matches: list[MatchRecord]) -> None:
    """แสดงสรุปผลลัพธ์สุดท้าย"""
    print()
    print("═" * 60)
    print("  📊 สรุปผลการสแกน")
    print("═" * 60)
    print(f"  ⏱️  เวลาทั้งหมด      : {stats.elapsed()}")
    print(f"  🌐 หน้าเว็บที่ crawl  : {stats.pages_crawled}")
    print(f"  🖼️  รูปภาพที่พบ       : {stats.images_found}")
    print(f"  🔬 รูปที่ประมวลผล     : {stats.images_processed}")
    print(f"  👤 ใบหน้าที่ตรวจจับ   : {stats.faces_detected}")
    print(f"  🎯 Match ที่พบ       : {stats.matches_found}")
    print(f"  ❌ Errors           : {stats.errors}")
    print("═" * 60)

    if matches:
        print("\n  🏆 Top Matches:")
        for i, m in enumerate(sorted(matches, key=lambda x: x.similarity, reverse=True)[:10], 1):
            print(f"     {i}. sim={m.similarity:.4f} | {m.filename}")
        print(f"\n  📁 ผลลัพธ์อยู่ที่: {OUTPUT_DIR}/")
        print(f"  📝 รายงาน CSV: {REPORT_FILE}")
    else:
        print("\n  ℹ️  ไม่พบ match — ลองลด SIMILARITY_THRESHOLD หรือเพิ่มรูปตัวอย่าง")

    print()


def main() -> None:
    """จุดเริ่มต้นของ pipeline ทั้งหมด"""
    print_banner()
    stats = CrawlStats()

    # ── Phase 1: สร้าง Target Profile ──
    log.info("━" * 50)
    log.info("📌 Phase 1: สร้าง Target Embedding...")
    log.info("━" * 50)
    face_app = initialize_face_engine()
    target_profile = build_target_profile(face_app)

    # ── Phase 2: Crawl เว็บไซต์ ──
    log.info("")
    log.info("━" * 50)
    log.info("📌 Phase 2: Crawling เว็บไซต์ %s...", ALLOWED_DOMAIN)
    log.info("━" * 50)
    crawler = DomainCrawler()
    crawler.crawl(START_URL, depth=0)
    stats.pages_crawled = len(crawler.visited_urls)
    stats.images_found = len(crawler.image_urls)
    log.info(
        "✅ Crawl เสร็จ — หน้าเว็บ: %d | รูปภาพ: %d",
        stats.pages_crawled, stats.images_found,
    )

    if not crawler.image_urls:
        log.warning("⚠️  ไม่พบรูปภาพเลย — ตรวจสอบ URL และ domain ให้ถูกต้อง")
        print_summary(stats, [])
        return

    # ── Phase 3 & 4: ดาวน์โหลด + ตรวจจับ + เทียบ ──
    log.info("")
    log.info("━" * 50)
    log.info("📌 Phase 3-4: ดาวน์โหลด & ตรวจจับใบหน้า & เทียบความคล้าย...")
    log.info("━" * 50)
    processor = FaceMatchProcessor(face_app, target_profile, stats)
    processor.process_batch(crawler.image_urls)

    # ── Export Results ──
    processor.export_report()
    print_summary(stats, processor.matches)


if __name__ == "__main__":
    main()
