import requests
import cv2
import numpy as np
import base64

# Load image
image_path = "frames/frame_0000.png"
img = cv2.imread(image_path)

# Encode image → base64
_, img_encoded = cv2.imencode(".png", img)
img_base64 = base64.b64encode(img_encoded).decode("utf-8")

# Create mask
h, w = img.shape[:2]
mask = np.zeros((h, w), dtype=np.uint8)
cv2.rectangle(mask, (50, 50), (200, 200), 255, -1)

_, mask_encoded = cv2.imencode(".png", mask)
mask_base64 = base64.b64encode(mask_encoded).decode("utf-8")

# Send request
response = requests.post(
    "http://127.0.0.1:8080/api/v1/inpaint",
    json={
        "image": img_base64,
        "mask": mask_base64,
        "model": "lama"
    }
)

print("Status:", response.status_code)

if response.status_code == 200:
    # 🔥 FIX: direct binary decode
    result = np.frombuffer(response.content, np.uint8)
    result_img = cv2.imdecode(result, cv2.IMREAD_COLOR)

    if result_img is not None:
        cv2.imwrite("test_output.png", result_img)
        print("✅ SUCCESS — output saved")
    else:
        print("❌ Failed decoding image")
else:
    print("❌ API FAILED:", response.text)