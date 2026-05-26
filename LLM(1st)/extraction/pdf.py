import pdfplumber

with pdfplumber.open("sample.pdf") as pdf:
    for page in pdf.pages():
        print(page.extract_text())