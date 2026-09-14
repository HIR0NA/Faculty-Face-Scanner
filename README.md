# 🔍 Faculty Face Scanner (Detective Bureau Edition)

**ระบบค้นหาใบหน้าอาจารย์/บุคคลเป้าหมายอัตโนมัติจากเว็บไซต์มหาวิทยาลัย**

สแกนและดึงรูปภาพจากเว็บไซต์ตามระดับความลึก (Crawl Depth) ตรวจจับใบหน้าทุกใบหน้า (รวมถึงรูปหมู่) ด้วยเทคโนโลยี AI Deep Metric Learning (InsightFace ArcFace) พร้อมหน้า Web UI สไตล์ Detective Bureau แยกผลลัพธ์ตามกิจกรรม/ข่าวมหาวิทยาลัย และส่งออกรายงาน CSV ได้ทันที

รองรับทั้งเครื่องที่มี **NVIDIA GPU (CUDA)** เพื่อความเร็วสูงสุด และมีระบบ **Auto-Fallback ทำงานบน CPU** สำหรับโน้ตบุ๊กทั่วไป

---

## ⚡ วิธีเปิดใช้งานแบบง่ายที่สุด (สำหรับ Windows - 1 คลิก)

เหมาะสำหรับอาจารย์หรือผู้ใช้งานทั่วไปที่ไม่ต้องการพิมพ์คำสั่งใน Terminal:

1. **ดาวน์โหลดโปรเจกต์:**
   - กดปุ่มสีเขียว **Code** บน GitHub แล้วเลือก **Download ZIP**
   - แตกไฟล์ ZIP ออกมาไว้ในเครื่อง (เช่น Desktop หรือ Documents)
2. **ดับเบิลคลิกไฟล์ `run.bat`:**
   - ตัวโปรแกรมจะตรวจสอบ Python, สร้าง Virtual Environment และติดตั้งแพ็กเกจที่จำเป็นให้อัตโนมัติ (ครั้งแรกจะใช้เวลาประมาณ 2-3 นาที)
   - จากนั้นหน้าต่างเว็บเบราว์เซอร์จะเด้งเปิดขึ้นมาที่ **http://localhost:5000** ให้อัตโนมัติทันที!

---

## 🖥️ วิธีใช้งานผ่านหน้า Web UI

เมื่อเปิดหน้าเว็บ **http://localhost:5000** ขึ้นมาแล้ว ให้ทำตามขั้นตอนดังนี้:

### 1. อัปโหลดรูปภาพบุคคลเป้าหมาย (Suspect Samples)
- ลากไฟล์รูปอาจารย์หรือบุคคลเป้าหมาย **1-3 รูป** มาวางในช่อง **"Drop evidence photos here"**
- *คำแนะนำ:* ใช้รูปหน้าตรง ชัดเจน ไม่สวมแว่นกันแดด/หน้ากาก ยิ่งมีหลายมุมยิ่งแม่นยำ

### 2. กำหนดค่าการค้นหา (Investigation Config)
- **TARGET URL:** ลิงก์เริ่มต้นที่ต้องการให้บอทค้นหา
  - *ทริคแนะนำ:* ใส่ลิงก์หน้าค้นหาชื่อของเว็บ เช่น `https://www.dusit.ac.th/home/?s=ชื่ออาจารย์` จะทำให้ค้นหาได้ตรงจุดและเจอรูปเยอะที่สุด
- **SEARCH DEPTH:** ระดับความลึกในการคลิกลิงก์ย่อย
  - `1`: ค้นหาเฉพาะในหน้าผลการค้นหาและข่าวที่เกี่ยวข้อง
  - `2 - 3`: มุดเข้าตามลิงก์ย่อยลึกขึ้น (ได้รูปเยอะขึ้น แต่อาจใช้เวลานานขึ้น)
- **MATCH THRESHOLD:** เกณฑ์ความคล้าย (แนะนำ `0.40` - `0.50`)
- **DETECTION SIZE:** ขนาดตรวจจับใบหน้า (`1280` คมชัดสูง / `640` เน้นความเร็ว)

### 3. เริ่มการค้นหา (Begin Investigation)
- กดปุ่ม **`▶ BEGIN INVESTIGATION`**
- ระบบจะดาวน์โหลดรูปภาพ ตรวจจับใบหน้า และแสดงผลลัพธ์แบบ Real-time
- ผลลัพธ์จะถูก**จัดหมวดหมู่แยกตามชื่องาน/ข่าวมหาวิทยาลัย** คลิกดูรูปขนาดเต็มได้ทันที
- กดปุ่ม **📦 Export CSV** เพื่อดาวน์โหลดสรุปผลเป็นตาราง Excel/CSV พร้อมลิงก์ต้นฉบับ

---

## 🛠️ วิธีติดตั้งและรันแบบ Manual (สำหรับ macOS / Linux / นักพัฒนา)

### 1. โคลน Repository
```bash
git clone https://github.com/HIR0NA/Faculty-Face-Scanner.git
cd Faculty-Face-Scanner
```

### 2. สร้างและเปิดใช้งาน Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. ติดตั้ง Dependencies
```bash
pip install --upgrade pip
pip install "numpy>=1.24.0,<2.0.0"
pip install -r requirements.txt
```

*(หมายเหตุ: ต้องใช้ `numpy < 2.0.0` เพื่อความเข้ากันได้กับ InsightFace)*

### 4. รัน Web Application
```bash
python web_app.py
```
เปิดเบราว์เซอร์ไปที่: **http://localhost:5000**

---

## 📁 โครงสร้างโปรเจกต์

```
Faculty-Face-Scanner/
├── run.bat                 # ตัวเปิดโปรแกรมแบบ 1 คลิกสำหรับ Windows
├── web_app.py              # Backend Server (Flask + Crawler + Face Recognition)
├── faculty_scanner.py      # Core Scanner Engine (สำหรับรันแบบ CLI)
├── requirements.txt        # รายการไลบรารีที่จำเป็น
├── templates/
│   └── index.html          # หน้าจอ Web UI (Detective Bureau Theme)
├── teacher_samples/        # โฟลเดอร์เก็บรูปตัวอย่างบุคคลเป้าหมาย
├── matched_results/        # โฟลเดอร์เก็บรูปที่ค้นพบ (แยกโฟลเดอร์ตามชื่องาน)
└── scan_report.csv         # ไฟล์รายงานสรุปผลลัพธ์
```

---

## 🔧 การแก้ไขปัญหาที่พบบ่อย (Troubleshooting)

### 1. ดับเบิลคลิก `run.bat` แล้วแจ้งว่า "ไม่พบ Python ในเครื่อง"
- ให้ดาวน์โหลดและติดตั้ง **Python 3.10 หรือ 3.11** จาก [python.org](https://www.python.org/)
- **สำคัญมาก:** ในขั้นตอนติดตั้ง ให้ติ๊กเลือกช่อง **"Add python.exe to PATH"** ด้านล่างสุดเสมอ

### 2. ติดตั้ง InsightFace ไม่สำเร็จ (แจ้งเตือน Visual C++ 14.0 or greater is required)
- ดาวน์โหลดและติดตั้ง **Visual Studio C++ Build Tools** จาก [Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- เลือกหัวข้อ **"Desktop development with C++"** แล้วกด Install

### 3. เครื่องไม่มีการ์ดจอ NVIDIA รันได้ไหม?
- **รันได้ 100%:** ระบบมีกลไกตรวจจับอัตโนมัติ หากไม่พบการ์ดจอแยกที่มี CUDA ระบบจะปรับไปประมวลผลด้วย CPU โดยอัตโนมัติ

### 4. สแกนแล้วขึ้น Error 403 Forbidden หรือถูกบล็อก
- เกิดจากการดึงข้อมูลเร็วเกินไปจนเซิร์ฟเวอร์มหาลัยป้องกัน
- *วิธีแก้:* ให้สลับไปแชร์เน็ตจาก Hotspot มือถือเพื่อเปลี่ยน IP Address แล้วสแกนต่อได้ทันที
