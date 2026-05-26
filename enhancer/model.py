import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

sr = cv2.dnn_superres.DnnSuperResImpl_create()

model_path = BASE_DIR / "EDSR_x4.pb"
sr.readModel(str(model_path))

sr.setModel("edsr", 4)

image_path = BASE_DIR / "low.jpg"
output_path = BASE_DIR / "enhanced.jpg"

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Could not open image: {image_path}")

upscaled = sr.upsample(image)
cv2.imwrite(str(output_path), upscaled)

print(f"Done! Saved: {output_path}")
