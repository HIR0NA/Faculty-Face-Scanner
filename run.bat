@echo off
chcp 65001 >nul
title J.A.R.V.I.S. - Faculty Face Scanner
echo ========================================================
echo   J.A.R.V.I.S. - Faculty Face Scanner Launcher
echo ========================================================
echo.

:: 1. ตรวจสอบว่ามี Python หรือไม่
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] ไม่พบ Python ในเครื่อง!
    echo กรุณาติดตั้ง Python 3.10 หรือ 3.11 จาก https://www.python.org/
    echo *อย่าลืมติ๊กถูก "Add Python to PATH" ตอนติดตั้ง*
    echo.
    pause
    exit /b
)

:: 2. สร้าง Virtual Environment ถ้ายังไม่มี
if not exist "venv\Scripts\activate.bat" (
    echo [*] กำลังสร้าง Virtual Environment (venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] สร้าง venv ไม่สำเร็จ!
        pause
        exit /b
    )
)

:: 3. เปิดใช้งาน venv
call venv\Scripts\activate.bat

:: 4. ติดตั้ง dependencies ถ้ายังไม่เคยติดตั้ง
python -c "import flask, cv2, insightface" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] กำลังติดตั้ง Dependencies (ครั้งแรกอาจใช้เวลา 2-5 นาที)...
    python -m pip install --upgrade pip
    pip install "numpy>=1.24.0,<2.0.0"
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo [WARNING] ติดตั้งบางแพ็กเกจไม่สำเร็จ อาจต้องติดตั้ง Visual C++ Build Tools
        echo ดาวน์โหลดได้ที่: https://visualstudio.microsoft.com/visual-cpp-build-tools/
        echo.
    )
)

:: 5. เปิด Browser อัตโนมัติหลังจากเริ่มรัน
start "" http://localhost:5000

:: 6. รัน Web App
echo.
echo [*] กำลังเริ่มระบบ Web Application...
echo [*] เปิดเว็บที่: http://localhost:5000
echo [*] กด Ctrl+C เพื่อปิดโปรแกรม
echo ========================================================
echo.

python web_app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] โปรแกรมหยุดทำงานผิดปกติ (Code: %errorlevel%)
)
echo.
pause
