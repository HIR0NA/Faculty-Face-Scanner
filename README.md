# 🔍 Faculty Face Scanner

**ระบบค้นหาใบหน้าอาจารย์อัตโนมัติจากเว็บไซต์มหาวิทยาลัย**

ใช้เทคโนโลยี Deep Metric Learning (InsightFace ArcFace) บน NVIDIA GPU เพื่อ crawl เว็บไซต์, ตรวจจับใบหน้าจากรูปภาพทุกรูป (รวมถึงรูปหมู่), แล้วเทียบกับใบหน้าเป้าหมายที่กำหนดไว้

---

## 📋 สิ่งที่ต้องมี (Prerequisites)

| ส่วนประกอบ | เวอร์ชันขั้นต่ำ | หมายเหตุ |
|---|---|---|
| Python | 3.10+ | แนะนำ 3.11 |
| NVIDIA GPU | — | ต้องมี CUDA cores (GTX 1060 ขึ้นไป) |
| CUDA Toolkit | 11.8+ หรือ 12.x | ตรวจด้วย `nvcc --version` |
| cuDNN | 8.x+ | ต้องตรงกับ CUDA version |
| Visual C++ Build Tools | 14.0+ | จำเป็นสำหรับ compile insightface บน Windows |

### ตรวจสอบ GPU

```bash
# ตรวจว่ามี NVIDIA GPU
nvidia-smi

# ตรวจ CUDA version
nvcc --version
```

---

## 🚀 วิธีติดตั้ง

### 1. สร้าง Virtual Environment (แนะนำ)

```bash
python -m venv venv
venv\Scripts\activate    # Windows
```

### 2. ถอน onnxruntime ตัวเก่า (ถ้ามี)

**สำคัญมาก:** ถ้ามี `onnxruntime` (CPU) ติดตั้งอยู่ ต้องถอนออกก่อน ไม่งั้น GPU จะไม่ทำงาน

```bash
pip uninstall onnxruntime onnxruntime-gpu -y
```

### 3. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 4. ตรวจสอบ CUDA Provider

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

ต้องเห็น `CUDAExecutionProvider` ในรายการ ถ้าไม่เห็น ดูส่วน **แก้ปัญหา** ด้านล่าง

---

## 📸 วิธีใช้งาน

### ขั้นตอนที่ 1 — ใส่รูปอาจารย์ตัวอย่าง

ใส่รูปอาจารย์ **1-3 รูป** ไว้ในโฟลเดอร์ `teacher_samples/`

**เคล็ดลับเลือกรูป:**
- ✅ รูปหน้าตรง, ชัด, แสงดี
- ✅ รูปจากหลายมุม/สภาพแสงจะช่วยเพิ่มความแม่นยำ
- ✅ รูปเดี่ยวดีที่สุด แต่รูปหมู่ก็ได้ (ระบบจะเลือกหน้าใหญ่สุดเอง)
- ❌ หลีกเลี่ยงรูปที่ใส่แว่นกันแดด/หน้ากาก
- ❌ หลีกเลี่ยงรูปเบลอหรือมืดมาก

**นามสกุลที่รองรับ:** `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`

### ขั้นตอนที่ 2 — ตั้งค่า URL เป้าหมาย

เปิดไฟล์ `faculty_scanner.py` แล้วแก้ค่าตรงส่วนบน:

```python
START_URL      = "https://www.sdu.ac.th"    # URL เริ่มต้น
ALLOWED_DOMAIN = "www.sdu.ac.th"            # จำกัด domain
MAX_CRAWL_DEPTH = 2                         # ความลึก (0=หน้าเดียว, 2=3 ชั้น)
SIMILARITY_THRESHOLD = 0.55                 # เกณฑ์ความคล้าย (0.0-1.0)
```

### ขั้นตอนที่ 3 — รัน!

```bash
python faculty_scanner.py
```

---

## 📁 โครงสร้างไฟล์

```
Faculty Face Scanner/
├── faculty_scanner.py      ← สคริปต์หลัก
├── requirements.txt        ← รายการ dependencies
├── README.md               ← คู่มือ (ไฟล์นี้)
├── teacher_samples/        ← ใส่รูปอาจารย์ตัวอย่างที่นี่
│   ├── photo1.jpg
│   └── photo2.jpg
├── matched_results/        ← ผลลัพธ์ (รูปที่ match + bounding box)
│   ├── match_20260823_1430_abc123_face0.jpg
│   └── ...
└── scan_report.csv         ← รายงาน CSV (สร้างอัตโนมัติ)
```

---

## 📊 ผลลัพธ์

### matched_results/
รูปภาพที่ตรวจจับว่าตรงกับอาจารย์ — มี **กรอบสีเขียว** รอบใบหน้าที่ match พร้อมคะแนนความคล้าย

### scan_report.csv
ตาราง CSV ที่มีคอลัมน์:

| คอลัมน์ | คำอธิบาย |
|---|---|
| Filename | ชื่อไฟล์รูปที่บันทึกไว้ |
| Similarity | คะแนนความคล้าย (0.0-1.0) |
| Image_URL | URL ต้นฉบับของรูปภาพ |
| Source_Page | หน้าเว็บที่พบรูปภาพนี้ |
| Timestamp | เวลาที่ตรวจจับ |

---

## ⚙️ ปรับแต่งค่า

| ค่า | ค่าเริ่มต้น | คำอธิบาย |
|---|---|---|
| `START_URL` | `https://www.sdu.ac.th` | URL เริ่มต้นของ crawler |
| `ALLOWED_DOMAIN` | `www.sdu.ac.th` | จำกัดให้ crawl เฉพาะ domain นี้ |
| `MAX_CRAWL_DEPTH` | `2` | ความลึกของการ crawl (0 = หน้าเดียว) |
| `CRAWL_DELAY` | `0.5` | หน่วงระหว่างดึงแต่ละหน้า (วินาที) |
| `SIMILARITY_THRESHOLD` | `0.55` | เกณฑ์ความคล้ายขั้นต่ำ |
| `MAX_WORKERS` | `16` | จำนวน thread สำหรับดาวน์โหลด |
| `DET_SIZE` | `(1280, 1280)` | ขนาด input ของ face detector |
| `REQUEST_TIMEOUT` | `15` | timeout สำหรับ HTTP request (วินาที) |
| `MIN_IMAGE_SIZE` | `50` | ขนาดรูปต่ำสุด (พิกเซล) |

### แนะนำค่า Threshold

| สถานการณ์ | Threshold | หมายเหตุ |
|---|---|---|
| ค้นหาแม่นยำสูง | 0.55 - 0.60 | False positive น้อย |
| สมดุล | 0.45 - 0.55 | ค่าเริ่มต้นที่ดี |
| ค้นหาครอบคลุม | 0.35 - 0.45 | อาจมี false positive |

---

## 🔧 แก้ปัญหาที่พบบ่อย

### ❌ "CUDAExecutionProvider ไม่พร้อม"

1. ตรวจว่ามี NVIDIA GPU: `nvidia-smi`
2. ตรวจว่าติดตั้ง CUDA Toolkit: `nvcc --version`
3. ถอน onnxruntime ทั้งหมดแล้วติดตั้งใหม่:
   ```bash
   pip uninstall onnxruntime onnxruntime-gpu -y
   pip install onnxruntime-gpu
   ```
4. ตรวจว่า CUDA bin อยู่ใน PATH:
   ```bash
   echo %PATH% | findstr CUDA
   ```

### ❌ "Microsoft Visual C++ 14.0 or greater is required"

ติดตั้ง Visual C++ Build Tools:
- ดาวน์โหลด [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- เลือก "Desktop development with C++"

### ❌ "ตรวจจับใบหน้าไม่ได้ในรูปตัวอย่าง"

- ใช้รูปที่ชัดเจน, หน้าตรง
- ตรวจว่ารูปไม่เสียหาย: ลองเปิดดูใน image viewer
- ลองใช้รูปขนาดใหญ่กว่า (อย่างน้อย 200x200 px)

### ❌ "ไม่พบ match เลย"

- ลด `SIMILARITY_THRESHOLD` ลง (เช่น 0.45)
- เพิ่มรูปตัวอย่างจากหลายมุม
- ตรวจว่า URL เป้าหมายมีรูปอาจารย์จริงๆ

### ❌ โปรแกรมช้ามาก

- ตรวจว่าใช้ GPU จริง (ดู log ว่ามี CUDAExecutionProvider)
- ลด `DET_SIZE` เป็น `(960, 960)` หรือ `(640, 640)`
- ลด `MAX_CRAWL_DEPTH` เป็น 1

### ❌ "NumPy _ARRAY_API not found"

NumPy 2.0+ ไม่เข้ากันกับ onnxruntime เก่า:
```bash
pip install "numpy>=1.24.0,<2.0.0"
```

---

## 🏗️ สถาปัตยกรรม

```
┌──────────────────┐
│  teacher_samples │ รูปอาจารย์ 1-3 รูป
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Phase 1         │ InsightFace (SCRFD + ArcFace)
│  Target Embed    │ → L2-normalized mean embedding (512-d)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Phase 2         │ BeautifulSoup + requests
│  Domain Crawl    │ → รวบรวม image URLs ทั้งหมด
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Phase 3         │ ThreadPoolExecutor (16 workers)
│  Download + Det  │ → ดาวน์โหลดรูป + ตรวจจับใบหน้า (GPU)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Phase 4         │ Cosine Similarity >= 0.55
│  Match & Output  │ → matched_results/ + scan_report.csv
└──────────────────┘
```

---

## 📜 License

สร้างสำหรับใช้งานภายในเท่านั้น — กรุณาเคารพ robots.txt และข้อกำหนดการใช้งานของเว็บไซต์เป้าหมาย
