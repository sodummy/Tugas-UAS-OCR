# Document Scanner (OpenCV + Tesseract)

Quick scanner that captures from webcam, detects document, preprocesses, and runs OCR.

Requirements
- Python 3.8+
- Tesseract OCR installed and on PATH (or set env var `TESSERACT_CMD` to tesseract.exe path)
- Install Python deps:

```bash
pip install -r requirements.txt
```

Run

```bash
python scan_from_camera.py
```

Usage
- Press `s` to scan the current frame (saves `scanned_image.png` and `scanned_output.txt`).
- Press `q` to quit.

Notes
- If Tesseract is not found on Windows, set environment variable before running:

```powershell
$env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
python scan_from_camera.py
```
