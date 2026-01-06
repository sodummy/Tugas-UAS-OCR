@echo off
cd /d "C:\Users\User\OneDrive\Documents\Semester 5\Pengolahan Citra Digital"
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
call .venv\Scripts\activate.bat
python scan_from_camera.py
pause
