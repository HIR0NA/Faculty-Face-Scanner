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

import argparse
import csv
from collections import Counter
import hashlib
import logging
import os
import re
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
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
GALLERY_FILE: str = "scan_gallery.html"
CACHE_DIR: str = ".embedding_cache"
DEDUP_THRESHOLD: float = 0.80       # เกณฑ์สำหรับจับรูปซ้ำ (similarity ระหว่าง match)
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
    activity_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    is_duplicate: bool = False


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


def _compute_samples_hash(samples_dir: str) -> str:
    """คำนวณ hash จากไฟล์ตัวอย่างทั้งหมดเพื่อใช้ตรวจสอบ cache"""
    samples_path = Path(samples_dir)
    if not samples_path.exists():
        return ""
    image_files = sorted(
        f for f in samples_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    h = hashlib.sha256()
    for f in image_files:
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
        h.update(str(int(f.stat().st_mtime)).encode())
    return h.hexdigest()[:16]


def _load_cached_profile(samples_dir: str) -> Optional[np.ndarray]:
    """โหลด target profile จาก cache ถ้ายังใช้ได้"""
    cache_dir = Path(CACHE_DIR)
    if not cache_dir.exists():
        return None
    cache_hash = _compute_samples_hash(samples_dir)
    cache_file = cache_dir / f"profile_{cache_hash}.npy"
    if cache_file.exists():
        profile = np.load(str(cache_file))
        log.info("⚡ โหลด target profile จาก cache: %s", cache_file.name)
        return profile
    return None


def _save_profile_cache(profile: np.ndarray, samples_dir: str) -> None:
    """บันทึก target profile ลง cache"""
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_hash = _compute_samples_hash(samples_dir)
    cache_file = cache_dir / f"profile_{cache_hash}.npy"
    np.save(str(cache_file), profile)
    log.info("💾 บันทึก target profile ลง cache: %s", cache_file.name)


def build_target_profile(face_app, samples_dir: str = SAMPLES_DIR, use_cache: bool = True) -> np.ndarray:
    """
    สร้าง target embedding จากรูปตัวอย่าง — ใช้ cache ถ้ามี

    Returns:
        np.ndarray: L2-normalized mean embedding (512,)
    """
    if use_cache:
        cached = _load_cached_profile(samples_dir)
        if cached is not None:
            return cached

    samples_path = Path(samples_dir)
    if not samples_path.exists():
        log.error("❌ ไม่พบโฟลเดอร์ %s — กรุณาสร้างแล้วใส่รูปอาจารย์", samples_dir)
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
            samples_dir, ", ".join(SUPPORTED_IMAGE_EXTENSIONS),
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
    _save_profile_cache(target_profile, samples_dir)
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
        self.page_titles: dict[str, str] = {}  # page_url → page_title
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

    def _is_pagination_link(self, url: str) -> bool:
        """ตรวจว่าเป็นลิงก์ pagination ของหน้าค้นหา/archive"""
        return bool(re.search(r'[?&]paged=\d+', url) or
                     re.search(r'/page/\d+/', url))

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

        return list(dict.fromkeys(valid))

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

        # ดึงชื่อหน้าเว็บ (ใช้เป็นชื่อโฟลเดอร์)
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            raw_title = title_tag.string.strip()
            for sep in (" – ", " - ", " | "):
                if sep in raw_title:
                    raw_title = raw_title.rsplit(sep, 1)[0].strip()
                    break
            self.page_titles[url] = raw_title

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
        if depth <= MAX_CRAWL_DEPTH:
            page_links: list[str] = []
            next_links: list[str] = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                if not href or href.startswith("#"):
                    continue
                abs_link = urljoin(url, href)
                if not (self._is_same_domain(abs_link) and self._is_valid_page_url(abs_link)):
                    continue
                # Pagination → crawl ที่ depth เดิม (ถือเป็นหน้าเดียวกัน)
                if self._is_pagination_link(abs_link):
                    page_links.append(abs_link)
                elif depth < MAX_CRAWL_DEPTH:
                    next_links.append(abs_link)

            for link in sorted(set(page_links)):
                self.crawl(link, depth)
            for link in sorted(set(next_links)):
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

    def __init__(self, face_app, target_profile: np.ndarray, stats: CrawlStats,
                 page_titles: Optional[dict[str, str]] = None):
        self.face_app = face_app
        self.target_profile = target_profile
        self.stats = stats
        self.page_titles = page_titles or {}
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

    @staticmethod
    def _sanitize_dirname(name: str, max_len: int = 120) -> str:
        """ทำให้ชื่อโฟลเดอร์ปลอดภัยสำหรับ filesystem"""
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        name = re.sub(r'\s+', ' ', name).strip().rstrip('.')
        if len(name) > max_len:
            name = name[:max_len].rsplit(' ', 1)[0].strip()
        return name or "unknown_page"

    def _get_activity_dir(self, source_page: str) -> Path:
        """สร้างและคืนค่า path ของโฟลเดอร์กิจกรรมจาก source page URL"""
        title = self.page_titles.get(source_page, "")
        if not title:
            parsed = urlparse(source_page)
            path_parts = [p for p in parsed.path.strip("/").split("/") if p]
            title = path_parts[-1] if path_parts else "unknown"
            title = title.replace("-", " ").replace("_", " ")
        dirname = self._sanitize_dirname(title)
        activity_dir = self.output_dir / dirname
        activity_dir.mkdir(parents=True, exist_ok=True)
        return activity_dir

    def _generate_filename(self, url: str, face_idx: int) -> str:
        """สร้างชื่อไฟล์ unique จาก URL hash"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"match_{timestamp}_{url_hash}_f{face_idx}.jpg"

    def deduplicate_matches(self) -> int:
        """ตรวจจับรูปซ้ำ — ใบหน้าเดียวกันจากมุมต่าง/ขนาดต่าง แล้วทำเครื่องหมาย"""
        valid = [m for m in self.matches if m.embedding is not None]
        if len(valid) < 2:
            return 0

        marked = 0
        n = len(valid)
        for i in range(n):
            if valid[i].is_duplicate:
                continue
            for j in range(i + 1, n):
                if valid[j].is_duplicate:
                    continue
                sim = float(np.dot(valid[i].embedding, valid[j].embedding))
                if sim >= DEDUP_THRESHOLD:
                    if valid[j].similarity < valid[i].similarity:
                        valid[j].is_duplicate = True
                    else:
                        valid[i].is_duplicate = True
                    marked += 1

        if marked > 0:
            log.info("🔄 พบรูปซ้ำ %d รูป (threshold=%.2f)", marked, DEDUP_THRESHOLD)
        return marked

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

        # เส้นกรอบบาง สีม่วงนีออน (สไตล์ JARVIS)
        color = (255, 0, 213)  # neon purple
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

                        # วาด bounding box + บันทึกลงโฟลเดอร์ของกิจกรรม
                        annotated = self._annotate_image(img, face.bbox, sim)
                        filename = self._generate_filename(img_url, face_idx)
                        activity_dir = self._get_activity_dir(source_page)
                        save_path = activity_dir / filename
                        ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if ok:
                            save_path.write_bytes(buf.tobytes())

                        record = MatchRecord(
                            filename=filename,
                            similarity=round(sim, 4),
                            image_url=img_url,
                            source_page=source_page,
                            activity_name=activity_dir.name,
                            embedding=emb.copy(),
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
            writer.writerow(["Activity", "Filename", "Similarity", "Image_URL", "Source_Page", "Timestamp"])
            for m in sorted(self.matches, key=lambda x: (-x.similarity,)):
                writer.writerow([m.activity_name, m.filename, m.similarity, m.image_url, m.source_page, m.timestamp])

        log.info("📝 บันทึก CSV report เสร็จ: %s (%d records)", report_path, len(self.matches))

    def export_gallery(self, stats: CrawlStats) -> None:
        """สร้าง HTML gallery แสดงผลลัพธ์แยกตามกิจกรรม"""
        if not self.matches:
            log.info("🌐 ไม่มี match — ข้าม gallery")
            return

        import html as _html
        from collections import defaultdict

        esc = _html.escape
        active_matches = [m for m in self.matches if not m.is_duplicate]
        groups: dict[str, list[MatchRecord]] = defaultdict(list)
        for m in active_matches:
            groups[m.activity_name or "unknown"].append(m)
        for g in groups.values():
            g.sort(key=lambda x: -x.similarity)
        sorted_acts = sorted(groups.items(), key=lambda x: -len(x[1]))

        total = len(active_matches)
        n_acts = len(sorted_acts)
        high = sum(1 for m in active_matches if m.similarity >= 0.55)
        med = sum(1 for m in active_matches if 0.45 <= m.similarity < 0.55)
        low = sum(1 for m in active_matches if m.similarity < 0.45)

        secs: list[str] = []
        for act, ms in sorted_acts:
            bst = max(x.similarity for x in ms)
            avg = sum(x.similarity for x in ms) / len(ms)
            cards = []
            for m in ms:
                src = esc(f"matched_results/{act}/{m.filename}")
                pct = f"{m.similarity:.1%}"
                c = "var(--green)" if m.similarity >= 0.55 else "var(--amber)" if m.similarity >= 0.45 else "var(--red)"
                cards.append(
                    f'<div class="card" data-sim="{m.similarity:.4f}">'
                    f'<img src="{src}" loading="lazy" onclick="openLightbox(this.src)">'
                    f'<div class="card-body"><span class="sim-badge" style="color:{c}">{pct}</span>'
                    f'<div class="card-meta">{esc(act[:80])}</div>'
                    f'<a class="card-link" href="{esc(m.source_page)}" target="_blank" rel="noopener">'
                    f'&#128279; ดูหน้าต้นฉบับ</a>'
                    f'<div class="card-ts">&#128336; {esc(m.timestamp)}</div>'
                    f'</div></div>'
                )
            cards_html = "\n".join(cards)
            secs.append(
                f'<div class="activity" data-count="{len(ms)}" data-best="{bst:.4f}">'
                f'<div class="act-head" onclick="toggle(this)">'
                f'<div class="act-icon">&#128194;</div>'
                f'<div class="act-title">{esc(act)}</div>'
                f'<div class="act-stats">{len(ms)} รูป | best {bst:.1%} | avg {avg:.1%}</div>'
                f'<span class="chevron">&#9660;</span></div>'
                f'<div class="grid">{cards_html}</div></div>'
            )
        body_sections = "\n".join(secs)

        html_out = _build_gallery_html(total, n_acts, high, med, low, body_sections)
        Path(GALLERY_FILE).write_text(html_out, encoding="utf-8")
        log.info(
            "🌐 สร้าง gallery เสร็จ: %s (%d กิจกรรม, %d matches)",
            GALLERY_FILE, n_acts, total,
        )


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
        # สรุปแยกตามกิจกรรม
        activity_counts = Counter(m.activity_name for m in matches)
        print(f"\n  📂 แยกตามกิจกรรม ({len(activity_counts)} กิจกรรม):")
        for i, (activity, count) in enumerate(activity_counts.most_common(), 1):
            sims = [m.similarity for m in matches if m.activity_name == activity]
            best = max(sims)
            avg = sum(sims) / len(sims)
            print(f"     {i:2d}. [{count:3d} รูป | best={best:.2f} avg={avg:.2f}] {activity}")

        print(f"\n  🏆 Top 10 Matches:")
        for i, m in enumerate(sorted(matches, key=lambda x: x.similarity, reverse=True)[:10], 1):
            act = m.activity_name[:35] + "…" if len(m.activity_name) > 35 else m.activity_name
            print(f"     {i:2d}. sim={m.similarity:.4f} | {act}")
        print(f"\n  📁 ผลลัพธ์อยู่ที่: {OUTPUT_DIR}/")
        print(f"  📝 รายงาน CSV: {REPORT_FILE}")
    else:
        print("\n  ℹ️  ไม่พบ match — ลองลด SIMILARITY_THRESHOLD หรือเพิ่มรูปตัวอย่าง")

    print()


def _generate_gallery_from_csv() -> None:
    """สร้าง gallery HTML จากข้อมูล CSV เดิม (ไม่ต้อง crawl ใหม่)"""
    import html as _html
    from collections import defaultdict

    csv_path = Path(REPORT_FILE)
    if not csv_path.exists():
        log.error("❌ ไม่พบไฟล์ %s", REPORT_FILE)
        return

    esc = _html.escape
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        act = r.get("Activity") or r.get("Page_Title", "").strip()
        if not act:
            fn = r["Filename"]
            act = fn.split("/")[0] if "/" in fn else "unknown"
        groups[act].append(r)

    for g in groups.values():
        g.sort(key=lambda x: -float(x["Similarity"]))
    sorted_acts = sorted(groups.items(), key=lambda x: -len(x[1]))
    total = len(rows)
    n_acts = len(sorted_acts)
    high = sum(1 for r in rows if float(r["Similarity"]) >= 0.55)
    med = sum(1 for r in rows if 0.45 <= float(r["Similarity"]) < 0.55)
    low = sum(1 for r in rows if float(r["Similarity"]) < 0.45)

    sections = []
    for act, ms in sorted_acts:
        bst = max(float(m["Similarity"]) for m in ms)
        avg = sum(float(m["Similarity"]) for m in ms) / len(ms)
        cards = []
        for m in ms:
            sim = float(m["Similarity"])
            fn = m["Filename"]
            if "/" not in fn:
                fn = f"{act}/{fn}"
            src = esc(f"matched_results/{fn}")
            pct = f"{sim:.1%}"
            c = "#00e676" if sim >= 0.55 else "#ffab00" if sim >= 0.45 else "#ff5252"
            sp = esc(m.get("Source_Page", ""))
            ts = esc(m.get("Timestamp", ""))
            cards.append(
                f'<div class="card" data-sim="{sim:.4f}">'
                f'<img src="{src}" loading="lazy" onclick="openLightbox(this.src)">'
                f'<div class="info"><span class="sim" style="color:{c}">{pct}</span>'
                f'<div class="meta">{esc(act[:80])}</div>'
                f'<a href="{sp}" target="_blank" rel="noopener">&#128279; ดูหน้าต้นฉบับ</a>'
                f'<div class="ts">&#128336; {ts}</div></div></div>'
            )
        cards_html = "\n".join(cards)
        sections.append(
            f'<div class="activity" data-count="{len(ms)}" data-best="{bst:.4f}">'
            f'<div class="ah" onclick="toggle(this)">'
            f'<div class="at">&#128194; {esc(act)}</div>'
            f'<div class="as">{len(ms)} รูป | best {bst:.1%} | avg {avg:.1%}</div>'
            f'<span class="chv">&#9660;</span></div>'
            f'<div class="grid">{cards_html}</div></div>'
        )

    body = "\n".join(sections)
    html_content = _build_gallery_html(total, n_acts, high, med, low, body)
    Path(GALLERY_FILE).write_text(html_content, encoding="utf-8")
    log.info("🌐 สร้าง gallery เสร็จ: %s (%d กิจกรรม, %d matches)", GALLERY_FILE, n_acts, total)


def _build_gallery_html(total: int, n_acts: int, high: int, med: int, low: int, body: str) -> str:
    """สร้าง HTML gallery ที่สวยงามทันสมัย"""
    return f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Faculty Face Scanner — Gallery</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');
:root {{
  --bg: #06060f;
  --surface: #0d0d1a;
  --surface2: #13132a;
  --border: rgba(139,92,246,.15);
  --border-hover: rgba(139,92,246,.4);
  --purple: #8b5cf6;
  --purple-glow: rgba(139,92,246,.3);
  --cyan: #06b6d4;
  --cyan-glow: rgba(6,182,212,.25);
  --green: #10b981;
  --amber: #f59e0b;
  --red: #ef4444;
  --text: #e2e8f0;
  --text-muted: #64748b;
  --text-dim: #475569;
  --mono: 'JetBrains Mono', 'Courier New', monospace;
  --sans: 'Inter', 'Segoe UI', system-ui, sans-serif;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;overflow-x:hidden}}

/* ── Animated grid background ── */
body::before {{
  content:'';position:fixed;top:0;left:0;width:100%;height:100%;
  background:
    linear-gradient(rgba(139,92,246,.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(139,92,246,.03) 1px,transparent 1px);
  background-size:60px 60px;pointer-events:none;z-index:0;
}}

/* ── Header ── */
.header {{
  position:relative;z-index:1;
  background:linear-gradient(180deg,rgba(139,92,246,.08) 0%,transparent 100%);
  border-bottom:1px solid var(--border);
  padding:32px 24px 28px;
}}
.header::after {{
  content:'';position:absolute;bottom:-1px;left:50%;transform:translateX(-50%);
  width:200px;height:2px;background:linear-gradient(90deg,transparent,var(--purple),transparent);
}}
.brand {{display:flex;align-items:center;justify-content:center;gap:12px;margin-bottom:24px}}
.brand-icon {{
  width:44px;height:44px;border-radius:12px;
  background:linear-gradient(135deg,var(--purple),var(--cyan));
  display:flex;align-items:center;justify-content:center;
  font-size:22px;box-shadow:0 0 20px var(--purple-glow);
}}
.brand h1 {{
  font-family:var(--mono);font-size:15px;font-weight:700;
  letter-spacing:3px;text-transform:uppercase;color:var(--purple);
}}
.brand span {{font-size:11px;color:var(--text-muted);letter-spacing:1px;display:block;margin-top:2px}}

/* ── Stats ── */
.stats {{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;max-width:700px;margin:0 auto}}
.stat {{
  flex:1;min-width:100px;max-width:140px;
  background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:14px 10px;text-align:center;
  transition:.2s;position:relative;overflow:hidden;
}}
.stat:hover {{border-color:var(--border-hover);transform:translateY(-1px)}}
.stat-val {{
  font-family:var(--mono);font-size:28px;font-weight:700;
  background:linear-gradient(135deg,var(--cyan),var(--purple));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}}
.stat-label {{font-size:9px;text-transform:uppercase;letter-spacing:2px;color:var(--text-muted);margin-top:4px}}
.stat.green .stat-val {{background:linear-gradient(135deg,var(--green),var(--cyan));-webkit-background-clip:text;background-clip:text}}
.stat.amber .stat-val {{background:linear-gradient(135deg,var(--amber),#fbbf24);-webkit-background-clip:text;background-clip:text}}
.stat.red .stat-val {{background:linear-gradient(135deg,var(--red),#fb7185);-webkit-background-clip:text;background-clip:text}}

/* ── Progress ── */
.progress {{margin:16px 24px;height:4px;background:var(--surface2);border-radius:2px;overflow:hidden}}
.progress-bar {{
  height:100%;border-radius:2px;
  background:linear-gradient(90deg,var(--purple),var(--cyan));
  box-shadow:0 0 8px var(--purple-glow);
}}

/* ── Container ── */
.container {{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:24px}}

/* ── Toolbar ── */
.toolbar {{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;flex-wrap:wrap;gap:12px}}
.toolbar h2 {{
  font-family:var(--mono);font-size:13px;font-weight:600;
  letter-spacing:2px;text-transform:uppercase;color:var(--purple);
}}
.toolbar .count {{font-family:var(--mono);font-size:13px;color:var(--cyan)}}

/* ── Filters ── */
.filters {{display:flex;gap:6px;margin-bottom:24px;flex-wrap:wrap}}
.fbtn {{
  background:var(--surface);border:1px solid var(--border);
  color:var(--text-muted);padding:7px 16px;border-radius:20px;
  cursor:pointer;font-size:12px;font-family:var(--sans);
  transition:.2s;user-select:none;
}}
.fbtn:hover {{border-color:var(--border-hover);color:var(--text)}}
.fbtn.on {{
  background:linear-gradient(135deg,rgba(139,92,246,.15),rgba(6,182,212,.1));
  border-color:var(--purple);color:#fff;
  box-shadow:0 0 12px var(--purple-glow);
}}

/* ── Activity sections ── */
.activity {{
  margin-bottom:16px;border:1px solid var(--border);border-radius:14px;
  overflow:hidden;background:var(--surface);
  transition:.2s;
}}
.activity:hover {{border-color:rgba(139,92,246,.25)}}
.act-head {{
  display:flex;align-items:center;padding:14px 20px;gap:12px;
  cursor:pointer;user-select:none;
  background:linear-gradient(90deg,rgba(139,92,246,.04),transparent);
  transition:.15s;
}}
.act-head:hover {{background:linear-gradient(90deg,rgba(139,92,246,.08),transparent)}}
.act-icon {{
  width:32px;height:32px;border-radius:8px;
  background:linear-gradient(135deg,rgba(139,92,246,.15),rgba(6,182,212,.1));
  border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;
  font-size:14px;flex-shrink:0;
}}
.act-title {{flex:1;font-size:14px;font-weight:600;color:var(--text);line-height:1.3}}
.act-stats {{
  font-family:var(--mono);font-size:11px;color:var(--cyan);
  white-space:nowrap;letter-spacing:.5px;
}}
.chevron {{
  color:var(--text-dim);transition:transform .3s;font-size:12px;
  width:20px;text-align:center;
}}
.activity.collapsed .chevron {{transform:rotate(-90deg)}}
.activity.collapsed .grid {{display:none}}

/* ── Card grid ── */
.grid {{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
  gap:12px;padding:4px 16px 16px;
}}
.card {{
  background:var(--surface2);border-radius:10px;overflow:hidden;
  border:1px solid transparent;transition:.25s;
}}
.card:hover {{
  border-color:var(--border-hover);
  transform:translateY(-3px);
  box-shadow:0 8px 24px rgba(0,0,0,.3),0 0 16px var(--purple-glow);
}}
.card img {{
  width:100%;aspect-ratio:4/3;object-fit:cover;display:block;
  background:var(--bg);cursor:pointer;transition:.2s;
}}
.card:hover img {{filter:brightness(1.05)}}
.card-body {{padding:12px 14px}}
.sim-badge {{
  font-family:var(--mono);font-size:20px;font-weight:700;
  display:block;margin-bottom:4px;
}}
.card-meta {{font-size:11px;color:var(--text-muted);margin-bottom:6px;line-height:1.4;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}}
.card-link {{color:var(--cyan);text-decoration:none;font-size:11px;display:inline-flex;align-items:center;gap:4px}}
.card-link:hover {{text-decoration:underline}}
.card-ts {{font-size:10px;color:var(--text-dim);margin-top:4px}}

/* ── Lightbox ── */
.lightbox {{
  display:none;position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,.95);z-index:999;
  justify-content:center;align-items:center;cursor:pointer;
  backdrop-filter:blur(8px);
}}
.lightbox.show {{display:flex}}
.lightbox img {{
  max-width:94vw;max-height:94vh;border-radius:12px;
  border:1px solid var(--border-hover);
  box-shadow:0 0 40px var(--purple-glow);
}}

/* ── Duplicate badge ── */
.dup-badge {{
  position:absolute;top:8px;right:8px;
  background:rgba(239,68,68,.9);color:#fff;
  font-size:9px;font-weight:700;padding:2px 6px;border-radius:4px;
  letter-spacing:.5px;text-transform:uppercase;
}}

/* ── Responsive ── */
@media(max-width:768px){{
  .grid{{grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}}
  .act-stats{{display:none}}
  .stat{{min-width:70px;padding:10px 6px}}
  .stat-val{{font-size:22px}}
}}
@media(max-width:480px){{
  .grid{{grid-template-columns:1fr 1fr;gap:6px;padding:4px 10px 10px}}
  .header{{padding:20px 16px}}
}}
</style>
</head>
<body>

<div class="header">
  <div class="brand">
    <div class="brand-icon">&#9889;</div>
    <div>
      <h1>Faculty Face Scanner</h1>
      <span>Automated Recognition Results</span>
    </div>
  </div>
  <div class="stats">
    <div class="stat"><div class="stat-val">{n_acts}</div><div class="stat-label">Activities</div></div>
    <div class="stat"><div class="stat-val">{total}</div><div class="stat-label">Matches</div></div>
    <div class="stat green"><div class="stat-val">{high}</div><div class="stat-label">High &ge;55%</div></div>
    <div class="stat amber"><div class="stat-val">{med}</div><div class="stat-label">Medium</div></div>
    <div class="stat red"><div class="stat-val">{low}</div><div class="stat-label">Low</div></div>
  </div>
</div>

<div class="progress"><div class="progress-bar" style="width:100%"></div></div>

<div class="container">
  <div class="toolbar">
    <h2>&#128270; Identified Matches</h2>
    <span class="count">{total} found &middot; {n_acts} activities</span>
  </div>
  <div class="filters">
    <div class="fbtn on" onclick="filterAll(this)">ALL ({total})</div>
    <div class="fbtn" onclick="filterTier(this,0.55,9)">HIGH &ge;55% ({high})</div>
    <div class="fbtn" onclick="filterTier(this,0.45,0.55)">MEDIUM ({med})</div>
    <div class="fbtn" onclick="filterTier(this,0,0.45)">LOW ({low})</div>
  </div>
  {body}
</div>

<div class="lightbox" id="lb"><img src="" alt=""></div>

<script>
function toggle(el){{el.closest('.activity').classList.toggle('collapsed')}}
function filterAll(btn){{
  document.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  document.querySelectorAll('.card').forEach(c=>c.style.display='');
  document.querySelectorAll('.activity').forEach(a=>a.style.display='');
}}
function filterTier(btn,lo,hi){{
  document.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  document.querySelectorAll('.card').forEach(c=>{{
    var s=parseFloat(c.dataset.sim);
    c.style.display=(s>=lo&&s<hi)?'':'none';
  }});
  document.querySelectorAll('.activity').forEach(a=>{{
    var vis=a.querySelectorAll('.card:not([style*="none"])').length;
    a.style.display=vis?'':'none';
  }});
}}
function openLightbox(src){{
  var lb=document.getElementById('lb');
  lb.querySelector('img').src=src;
  lb.classList.add('show');
}}
document.getElementById('lb').onclick=function(){{this.classList.remove('show')}};
document.addEventListener('keydown',function(e){{
  if(e.key==='Escape') document.getElementById('lb').classList.remove('show');
}});
</script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    """แปลง command line arguments"""
    parser = argparse.ArgumentParser(
        description="Faculty Face Scanner — ค้นหาใบหน้าอาจารย์จากเว็บไซต์",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-u", "--url", default=START_URL, help="URL เริ่มต้นสำหรับ crawl")
    parser.add_argument("-d", "--domain", default=ALLOWED_DOMAIN, help="domain ที่อนุญาต")
    parser.add_argument("-t", "--threshold", type=float, default=SIMILARITY_THRESHOLD,
                        help="เกณฑ์ความคล้ายขั้นต่ำ (default: %.2f)" % SIMILARITY_THRESHOLD)
    parser.add_argument("--depth", type=int, default=MAX_CRAWL_DEPTH,
                        help="ความลึกของ crawl (default: %d)" % MAX_CRAWL_DEPTH)
    parser.add_argument("-s", "--samples", default=SAMPLES_DIR, help="โฟลเดอร์รูปตัวอย่าง")
    parser.add_argument("-o", "--output", default=OUTPUT_DIR, help="โฟลเดอร์ผลลัพธ์")
    parser.add_argument("-w", "--workers", type=int, default=MAX_WORKERS,
                        help="จำนวน download workers (default: %d)" % MAX_WORKERS)
    parser.add_argument("--no-cache", action="store_true", help="ไม่ใช้ embedding cache")
    parser.add_argument("--no-dedup", action="store_true", help="ไม่ตรวจจับรูปซ้ำ")
    parser.add_argument("--dedup-threshold", type=float, default=DEDUP_THRESHOLD,
                        help="เกณฑ์จับรูปซ้ำ (default: %.2f)" % DEDUP_THRESHOLD)
    parser.add_argument("--gallery-only", action="store_true",
                        help="สร้างเฉพาะ gallery จากข้อมูลเดิม (ไม่ crawl)")
    return parser.parse_args()


def main() -> None:
    """จุดเริ่มต้นของ pipeline ทั้งหมด"""
    args = parse_args()

    global START_URL, ALLOWED_DOMAIN, SIMILARITY_THRESHOLD, MAX_CRAWL_DEPTH
    global SAMPLES_DIR, OUTPUT_DIR, MAX_WORKERS, DEDUP_THRESHOLD
    START_URL = args.url
    ALLOWED_DOMAIN = args.domain
    SIMILARITY_THRESHOLD = args.threshold
    MAX_CRAWL_DEPTH = args.depth
    SAMPLES_DIR = args.samples
    OUTPUT_DIR = args.output
    MAX_WORKERS = args.workers
    DEDUP_THRESHOLD = args.dedup_threshold

    print_banner()

    # ── Gallery-only mode ──
    if args.gallery_only:
        log.info("🌐 Gallery-only mode — สร้าง gallery จาก CSV เดิม")
        _generate_gallery_from_csv()
        return

    stats = CrawlStats()

    # ── Phase 1: สร้าง Target Profile ──
    log.info("━" * 50)
    log.info("📌 Phase 1: สร้าง Target Embedding...")
    log.info("━" * 50)
    face_app = initialize_face_engine()
    target_profile = build_target_profile(face_app, SAMPLES_DIR, use_cache=not args.no_cache)

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
    processor = FaceMatchProcessor(face_app, target_profile, stats, crawler.page_titles)
    processor.process_batch(crawler.image_urls)

    # ── Phase 5: Deduplicate ──
    if not args.no_dedup:
        log.info("")
        log.info("━" * 50)
        log.info("📌 Phase 5: ตรวจจับรูปซ้ำ...")
        log.info("━" * 50)
        n_dupes = processor.deduplicate_matches()
        if n_dupes:
            unique = [m for m in processor.matches if not m.is_duplicate]
            log.info("✅ เหลือ %d รูปไม่ซ้ำ (จาก %d)", len(unique), len(processor.matches))

    # ── Export Results ──
    processor.export_report()
    processor.export_gallery(stats)
    print_summary(stats, processor.matches)


if __name__ == "__main__":
    main()
