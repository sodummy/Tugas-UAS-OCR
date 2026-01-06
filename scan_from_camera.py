import os
import cv2
import numpy as np
import pytesseract
import imutils
from imutils.perspective import four_point_transform
from fpdf import FPDF
from PIL import Image
from datetime import datetime

# Optional: Allow user to set Tesseract executable via env var TESSERACT_CMD
tess_cmd = os.environ.get('TESSERACT_CMD')
if tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = tess_cmd


def find_document_contour(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None


def deskew_image(image):
    """Straighten tilted document"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find contours and get bounding rect
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        angle = rect[2]
        # Rotate if angle is significant
        if -45 < angle < 45:
            h, w = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return image


def preprocess_for_ocr(warped):
    """Enhanced preprocessing for better OCR results"""
    # Deskew the image
    warped = deskew_image(warped)
    
    # Convert to grayscale
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Denoise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Enhance contrast
    alpha = 1.5  # Brightness
    beta = 30    # Contrast
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # Adaptive thresholding for better results
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 21, 10)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return th


def save_text_to_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', size=10)
    pdf.set_auto_page_break(auto=True, margin=10)
    
    # Process text to remove problematic characters
    for line in text.split('\n'):
        if not line.strip():
            pdf.ln(3)
            continue
        try:
            # Remove characters that fpdf doesn't handle well
            # Keep only basic ASCII, Latin-1, and common punctuation
            clean_line = ''.join(c if ord(c) < 256 else ' ' for c in line)
            if clean_line.strip():
                pdf.multi_cell(0, 4, clean_line)
        except Exception as e:
            # Skip lines with problematic encoding
            try:
                # Try with just ASCII
                ascii_line = clean_line.encode('ascii', errors='ignore').decode('ascii')
                if ascii_line.strip():
                    pdf.multi_cell(0, 4, ascii_line)
            except:
                pass
    
    pdf.output(filename)
    return filename


def save_image_to_pdf(image_path, filename):
    # Use Pillow to convert image to PDF (handles various image formats)
    try:
        im = Image.open(image_path)
        if im.mode == 'RGBA':
            im = im.convert('RGB')
        im.save(filename, "PDF", resolution=100.0)
        return filename
    except Exception as e:
        raise


def scan_image(frame):
    ratio = frame.shape[0] / 500.0
    orig = frame.copy()
    image = imutils.resize(frame, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    docCnt = find_document_contour(edged)
    if docCnt is not None:
        pts = docCnt.reshape(4, 2) * ratio
        warped = four_point_transform(orig, pts)
    else:
        warped = orig

    # Create a cleaned, scan-like color image for saving (apply similar enhancement as OCR prep)
    gray_warp = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_warp)
    gray_denoise = cv2.bilateralFilter(gray_clahe, 11, 17, 17)
    scanned_clean = cv2.convertScaleAbs(gray_denoise, alpha=1.5, beta=30)
    scanned_color = cv2.cvtColor(scanned_clean, cv2.COLOR_GRAY2BGR)

    ocr_ready = preprocess_for_ocr(warped)
    return warped, ocr_ready, scanned_color


def main():
    print('Starting camera. Press `s` to scan, `p` to save PDF, `d` for demo, `q` to quit.')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    out_text = 'scanned_output.txt'
    out_img = 'scanned_image.png'
    out_img_processed = 'scanned_image_processed.png'
    last_text = ''

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to grab frame')
            break

        display = frame.copy()
        cv2.putText(display, 'Press s to scan, p to save PDF, d for demo, q to quit', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow('Camera', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            # Demo mode with sample text
            last_text = """DOCUMENT SCANNER DEMO
Ini adalah hasil scan dari dokumen demo.
Tulisan ini adalah contoh OCR text yang bisa disave ke PDF.

Demo mode memungkinkan testing tanpa kamera."""
            print(f'Demo text loaded: {len(last_text)} chars')
            print('Press `p` to save this demo text as PDF.')
        elif key == ord('s'):
            warped, ocr_ready, scanned_color = scan_image(frame)
            cv2.imshow('Scanned (preprocessed)', ocr_ready)

            # Run OCR with better config
            try:
                # Better Tesseract config: --psm 6 = uniform block of text
                # -l eng+ind = English + Indonesian languages
                custom_config = r'--psm 6 -l eng'
                last_text = pytesseract.image_to_string(ocr_ready, config=custom_config)
            except Exception as e:
                last_text = ''
                print('OCR error:', e)

            with open(out_text, 'w', encoding='utf-8') as f:
                f.write(last_text)

            cv2.imwrite(out_img, warped)
            # save the cleaned scanned image (processed)
            cv2.imwrite(out_img_processed, scanned_color)
            print(f'Scan saved: {out_img}, processed image saved: {out_img_processed}, text saved: {out_text}')
            print('Press `p` to save as PDF, or `s` to scan again.')
        elif key == ord('p'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if last_text:
                pdf_filename = f'scanned_output_text_{timestamp}.pdf'
                try:
                    save_text_to_pdf(last_text, pdf_filename)
                    print(f'Text PDF saved: {pdf_filename}')
                except Exception as e:
                    print(f'Error saving text PDF: {e}')
            else:
                # If no OCR text, save the last processed scanned image to PDF
                if os.path.exists(out_img_processed):
                    pdf_filename = f'scanned_image_{timestamp}.pdf'
                    try:
                        save_image_to_pdf(out_img_processed, pdf_filename)
                        print(f'Image PDF saved: {pdf_filename}')
                    except Exception as e:
                        print(f'Error saving image PDF: {e}')
                else:
                    print('No OCR text and no scanned image available. Use `s` to scan first or `d` for demo text.')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
