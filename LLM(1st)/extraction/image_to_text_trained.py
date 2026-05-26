import pytesseract
from PIL import Image
import spacy
import cv2
import numpy as np
import re


# 1. Tesseract Path Set Karein
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\badha\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def get_clean_text(image_path):
    # OpenCV se image read karein
    img = cv2.imread(image_path)
    
    if img is None:
        raise FileNotFoundError(f"Image nahi mili is path par: {image_path}")

    # Grayscale mein convert karein
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding: Image ko pure Black & White banana
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Rescale: Accuracy badhane ke liye image size double karein
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Tesseract config
    custom_config = r'--oem 3 --psm 6' 
    text = pytesseract.image_to_string(gray, config=custom_config)
    
    return text

def clean_ocr_text(text):
    # Sirf A-Z, a-z, numbers aur spaces rehne do
    # Baaki symbols (| [ — _) hata do
    clean = re.sub(r'[^a-zA-Z0-9\s,.]', '', text)
    # Extra spaces aur new lines hatayein
    clean = " ".join(clean.split())
    return clean

# --- MAIN EXECUTION ---
try:
    # 2. Path ko variable mein rakhein (Use 'r' for raw string)
    path = r"C:\onedrive\Desktop\LLM\LLM(1st)\extraction\portfolio.png"
    
    # 3. OCR se text nikaalein
    extracted_text = get_clean_text(path)
    extracted_texts = clean_ocr_text(extracted_text)
    print("--- Refined Extracted Text ---")
    print(extracted_texts)

    # 4. SpaCy Model Load Karein
    # Ensure karein ki aapne model pehle train karke save kiya hua hai
    nlp = spacy.load("./my_tech_model")

    # 5. Model se Entities nikaalein
    doc = nlp(extracted_text)

    print("\n--- Entities Found by Your Model ---")
    if not doc.ents:
        print("Koi Name ya Skill nahi mili. Shayad model ko aur training chahiye ya OCR text noisy hai.")
    else:
        for ent in doc.ents:
            print(f"{ent.text} -> {ent.label_}")

except Exception as e:
    print(f"Error occurred: {e}")