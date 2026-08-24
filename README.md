# 🔍 J.A.R.V.I.S. — Faculty Face Scanner

**ระบบค้นหาใบหน้าอาจารย์อัตโนมัติจากเว็บไซต์มหาวิทยาลัย**

ใช้เทคโนโลยี Deep Metric Learning (InsightFace ArcFace) บน NVIDIA GPU เพื่อ crawl เว็บไซต์, ตรวจจับใบหน้าจากรูปภาพทุกรูป (รวมถึงรูปหมู่), แล้วเทียบกับใบหน้าเป้าหมายที่กำหนดไว้

พร้อม **Web UI** สไตล์ Iron Man HUD สำหรับควบคุมและดูผลลัพธ์แบบ Real-time

---

## ✨ Features

- 🤖 **AI Face Recognition** — ใช้ InsightFace (SCRFD + ArcFace) ตรวจจับและจดจำใบหน้า 512 มิติ
- 🌐 **Smart Web Crawler** — Crawl เว็บไซต์ตาม depth ที่กำหนด พร้อม rate limiting ป้องกันโดนบล็อก
- 🖥️ **Web UI (JARVIS HUD)** — หน้าเว็บสไตล์ Iron Man สำหรับตั้งค่า สแกน และดูผลลัพธ์แบบ real-time
- ⚡ **GPU Accelerated** — ใช้ NVIDIA CUDA เร่งความเร็วการประมวลผล
- 📊 **Export CSV** — ส่งออกผลลัพธ์เป็นไฟล์ CSV พร้อมลิงก์ต้นทาง
- 🖼️ **Matched Gallery** — แสดงรูปที่ match พร้อม bounding box และคะแนนความคล้าย

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

## 🚀 วิธีติดตั้ง (Quick Start)

### 1. Clone โปรเจค

```bash
git clone https://github.com/HIR0NA/Faculty-Face-Scanner.git
cd Faculty-Face-Scanner
```

### 2. สร้าง Virtual Environment (แนะนำ)

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. ถอน onnxruntime ตัวเก่า (ถ้ามี)

**สำคัญมาก:** ถ้ามี `onnxruntime` (CPU) ติดตั้งอยู่ ต้องถอนออกก่อน ไม่งั้น GPU จะไม่ทำงาน

```bash
pip uninstall onnxruntime onnxruntime-gpu -y
```

### 4. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 5. ตรวจสอบ CUDA Provider

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

### ขั้นตอนที่ 2 — เปิด Web UI

```bash
python web_app.py
```

เปิดเบราว์เซอร์ไปที่ **http://localhost:5000**

### ขั้นตอนที่ 3 — ตั้งค่าและสแกน

บน Web UI คุณสามารถ:

1. **TARGET URI** — ใส่ URL ของเว็บไซต์เป้าหมาย
2. **CRAWL DEPTH** — ตั้งความลึกของการ crawl (0 = หน้าเดียว, 3 = 4 ชั้น)
3. **SIMILARITY THRESHOLD** — ตั้งเกณฑ์ความคล้าย (0.0-1.0)
4. **DETECTION SIZE** — ขนาด input ของ face detector
5. กด **▶ INITIATE SCAN** เพื่อเริ่มสแกน

### (ทางเลือก) รัน CLI โดยตรง

```bash
python faculty_scanner.py
```

---

## 📁 โครงสร้างไฟล์

```
Faculty-Face-Scanner/
├── web_app.py              ← Web UI (Flask) — JARVIS HUD
├── faculty_scanner.py      ← Core engine (crawler + face recognition)
├── requirements.txt        ← รายการ dependencies
├── README.md               ← คู่มือ (ไฟล์นี้)
├── .gitignore              ← ไฟล์ที่ไม่ต้องอัพ git
├── templates/
│   └── index.html          ← หน้าเว็บ JARVIS HUD
├── teacher_samples/        ← ใส่รูปอาจารย์ตัวอย่างที่นี่
│   ├── .gitkeep
│   └── (ใส่รูปของคุณที่นี่)
└── matched_results/        ← ผลลัพธ์ (สร้างอัตโนมัติ)
    └── .gitkeep
```

---

## 📊 ผลลัพธ์

### Web UI Dashboard
หน้า Web UI จะแสดงผลลัพธ์แบบ real-time:
- **PAGES** — จำนวนหน้าเว็บที่ crawl แล้ว
- **IMAGES** — จำนวนรูปที่ดาวน์โหลดแล้ว
- **FACES** — จำนวนใบหน้าที่ตรวจจับได้
- **MATCHES** — จำนวนรูปที่ match กับเป้าหมาย

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
| `TARGET URI` | — | URL เริ่มต้นของ crawler |
| `CRAWL DEPTH` | `3` | ความลึกของการ crawl (0 = หน้าเดียว) |
| `SIMILARITY THRESHOLD` | `0.40` | เกณฑ์ความคล้ายขั้นต่ำ |
| `DETECTION SIZE` | `1280` | ขนาด input ของ face detector |

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

- ลด `SIMILARITY THRESHOLD` ลง (เช่น 0.45)
- เพิ่มรูปตัวอย่างจากหลายมุม
- ตรวจว่า URL เป้าหมายมีรูปอาจารย์จริงๆ

### ❌ โปรแกรมช้ามาก

- ตรวจว่าใช้ GPU จริง (ดู log ว่ามี CUDAExecutionProvider)
- ลด `DETECTION SIZE` เป็น `960` หรือ `640`
- ลด `CRAWL DEPTH`

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
│  Phase 4         │ Cosine Similarity >= threshold
│  Match & Output  │ → matched_results/ + scan_report.csv
└──────────────────┘
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **InsightFace** (SCRFD face detection + ArcFace embedding)
- **ONNX Runtime GPU** (CUDA acceleration)
- **Flask** (Web UI server)
- **BeautifulSoup4** (HTML parsing / web crawling)
- **OpenCV** (Image processing)
- **NumPy** (Numerical computing)

---

## 📜 License

MIT License — สร้างสำหรับงานวิจัย/การศึกษา กรุณาเคารพ robots.txt และข้อกำหนดการใช้งานของเว็บไซต์เป้าหมาย
