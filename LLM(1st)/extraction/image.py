import pytesseract
from PIL import Image

img = Image.open("sam.jpeg")
text = pytesseract.image_to_string(img)
print(text)