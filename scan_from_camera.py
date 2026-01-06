import os
import cv2
import numpy as np
import pytesseract
import imutils
from imutils.perspective import four_point_transform
from fpdf import FPDF
from PIL import Image
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image as PILImage, ImageTk
import threading
import time

# Optional: Allow user to set Tesseract executable via env var TESSERACT_CMD
tess_cmd = os.environ.get('TESSERACT_CMD')
if tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = tess_cmd
else:
    # Try common Windows installation paths
    common_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\User\AppData\Local\Tesseract-OCR\tesseract.exe',
    ]
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break


def check_tesseract():
    """Check if Tesseract is properly installed and accessible"""
    try:
        pytesseract.get_tesseract_version()
        return True, "Tesseract is ready"
    except Exception as e:
        return False, str(e)


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


class OCRApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Controls")
        self.root.geometry("700x600")
        self.root.configure(bg='#2c3e50')
        
        self.ocr_text = ""
        self.current_image = None
        self.processing = False
        self.current_language = "eng"
        
        # Check Tesseract on startup
        self.tesseract_available, self.tesseract_msg = check_tesseract()
        
        self.setup_ui()
        
        # Show warning if Tesseract not found
        if not self.tesseract_available:
            self.show_tesseract_installation_guide()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Load Image Button
        self.load_btn = tk.Button(main_frame, text="Load Image", command=self.load_image,
                                   bg='#3498db', fg='white', font=('Arial', 11, 'bold'),
                                   height=2, cursor='hand2')
        self.load_btn.pack(fill=tk.X, pady=5)
        
        # Language Selection
        lang_frame = ttk.Frame(main_frame)
        lang_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(lang_frame, text="Language:", font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        self.lang_var = tk.StringVar(value="eng")
        self.lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var,
                                        values=["eng", "ind", "eng+ind"], state="readonly",
                                        width=15)
        self.lang_combo.pack(side=tk.LEFT, padx=5)
        self.lang_combo.bind("<<ComboboxSelected>>", lambda e: self.update_language())
        
        # Process OCR Button
        self.process_btn = tk.Button(main_frame, text="Process OCR", command=self.process_ocr,
                                      bg='#2ecc71', fg='white', font=('Arial', 11, 'bold'),
                                      height=2, cursor='hand2')
        self.process_btn.pack(fill=tk.X, pady=5)
        
        # Export Results Section
        export_frame = ttk.LabelFrame(main_frame, text="Export Results:", padding=10)
        export_frame.pack(fill=tk.X, pady=10)
        
        button_frame = ttk.Frame(export_frame)
        button_frame.pack(fill=tk.X)
        
        self.read_btn = tk.Button(button_frame, text="Read", command=self.read_text,
                                   bg='#e74c3c', fg='white', font=('Arial', 10),
                                   width=15, cursor='hand2')
        self.read_btn.pack(side=tk.LEFT, padx=5)
        
        self.copy_btn = tk.Button(button_frame, text="Copy", command=self.copy_to_clipboard,
                                   bg='#f39c12', fg='white', font=('Arial', 10),
                                   width=15, cursor='hand2')
        self.copy_btn.pack(side=tk.LEFT, padx=5)
        
        self.export_pdf_btn = tk.Button(button_frame, text="Export PDF", command=self.export_pdf,
                                         bg='#9b59b6', fg='white', font=('Arial', 10),
                                         width=15, cursor='hand2')
        self.export_pdf_btn.pack(side=tk.LEFT, padx=5)
        
        # Statistics Frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics:", padding=10)
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.stats_label = tk.Label(stats_frame, text="• Words: 0\n• Characters: 0\n• Processing Time: 0.00s\n• Language: eng",
                                     font=('Arial', 10), justify=tk.LEFT, bg='#2c3e50', fg='#ecf0f1')
        self.stats_label.pack(fill=tk.X, padx=5, pady=5)
        
        # OCR Result Text Area
        result_frame = ttk.LabelFrame(main_frame, text="OCR Result:", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.text_area = tk.Text(result_frame, height=8, font=('Courier', 10),
                                  bg='#34495e', fg='#ecf0f1', wrap=tk.WORD)
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar for text area
        scrollbar = ttk.Scrollbar(self.text_area, command=self.text_area.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.config(yscrollcommand=scrollbar.set)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file_path:
            self.current_image = cv2.imread(file_path)
            messagebox.showinfo("Success", f"Image loaded: {os.path.basename(file_path)}")
            
    def update_language(self):
        self.current_language = self.lang_var.get()
        
    def process_ocr(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        if not self.tesseract_available:
            messagebox.showerror("Tesseract Not Found", 
                "Tesseract OCR is not installed.\n\n"
                "Please download and install from:\n"
                "https://github.com/UB-Mannheim/tesseract/wiki\n\n"
                "1. Download: tesseract-ocr-w64-setup-v5.3.0.exe\n"
                "2. Run installer (default location is fine)\n"
                "3. Restart this application")
            return
            
        self.process_btn.config(state=tk.DISABLED, text="Processing...")
        self.root.update()
        
        # Run OCR in a separate thread
        threading.Thread(target=self._run_ocr, daemon=True).start()
        
    def _run_ocr(self):
        try:
            start_time = time.time()
            
            # Preprocess image
            warped, ocr_ready, _ = scan_image(self.current_image)
            
            # Run OCR with selected language
            custom_config = f'--psm 6 -l {self.current_language}'
            self.ocr_text = pytesseract.image_to_string(ocr_ready, config=custom_config)
            
            processing_time = time.time() - start_time
            
            # Update UI
            self.text_area.config(state=tk.NORMAL)
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(1.0, self.ocr_text)
            self.text_area.config(state=tk.NORMAL)
            
            # Update statistics
            word_count = len(self.ocr_text.split())
            char_count = len(self.ocr_text)
            
            stats_text = f"• Words: {word_count}\n• Characters: {char_count}\n• Processing Time: {processing_time:.2f}s\n• Language: {self.current_language}"
            self.stats_label.config(text=stats_text)
            
            messagebox.showinfo("Success", "OCR processing completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"OCR processing failed: {str(e)}")
            
        finally:
            self.process_btn.config(state=tk.NORMAL, text="Process OCR")
            
    def read_text(self):
        if not self.ocr_text:
            messagebox.showwarning("Warning", "No OCR text available!")
            return
        messagebox.showinfo("OCR Result", self.ocr_text)
        
    def copy_to_clipboard(self):
        if not self.ocr_text:
            messagebox.showwarning("Warning", "No OCR text to copy!")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.ocr_text)
        messagebox.showinfo("Success", "Text copied to clipboard!")
        
    def export_pdf(self):
        if not self.ocr_text:
            messagebox.showwarning("Warning", "No OCR text to export!")
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f'ocr_result_{timestamp}.pdf'
        
        try:
            save_text_to_pdf(self.ocr_text, pdf_filename)
            messagebox.showinfo("Success", f"PDF exported: {pdf_filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PDF: {str(e)}")
    
    def show_tesseract_installation_guide(self):
        """Show installation guide for Tesseract"""
        guide = """
        === TESSERACT OCR INSTALLATION GUIDE ===
        
        Tesseract is not currently installed on your system.
        
        INSTALLATION STEPS:
        1. Download installer from:
           https://github.com/UB-Mannheim/tesseract/wiki
           
        2. Download file: tesseract-ocr-w64-setup-v5.3.0.exe
           (or latest version)
        
        3. Run the installer with default settings
           (Installation path: C:\\Program Files\\Tesseract-OCR)
        
        4. Restart this application
        
        After installation, the OCR feature will be available.
        """
        messagebox.showinfo("Setup Required", guide)


def run_gui():
    root = tk.Tk()
    app = OCRApplication(root)
    root.mainloop()


if __name__ == '__main__':
    run_gui()
