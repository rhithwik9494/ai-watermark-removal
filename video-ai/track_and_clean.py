import cv2
import os
import requests
import numpy as np
import base64

frames_dir = "frames"
output_dir = "clean_frames"

os.makedirs(output_dir, exist_ok=True)

frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])

for file in frame_files:
    frame_path = os.path.join(frames_dir, file)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"⚠️ Skipping {file}")
        continue

    h, w = frame.shape[:2]

    # 🔥 AUTO MASK (bottom-right watermark area)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Adjust this if watermark position changes
    cv2.rectangle(mask, (w - 220, h - 120), (w - 10, h - 10), 255, -1)

    # Encode image
    _, img_encoded = cv2.imencode(".png", frame)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")

    # Encode mask
    _, mask_encoded = cv2.imencode(".png", mask)
    mask_base64 = base64.b64encode(mask_encoded).decode("utf-8")

    try:
        response = requests.post(
            "http://127.0.0.1:8080/api/v1/inpaint",
            json={
                "image": img_base64,
                "mask": mask_base64,
                "model": "lama"
            },
            timeout=60
        )

        if response.status_code != 200:
            print(f"❌ API error {file}")
            continue

        # Decode result
        result = np.frombuffer(response.content, np.uint8)
        result_img = cv2.imdecode(result, cv2.IMREAD_COLOR)

        if result_img is not None:
            out_path = os.path.join(output_dir, file)
            cv2.imwrite(out_path, result_img)
            print(f"✅ Cleaned {file}")
        else:
            print(f"⚠️ Decode failed {file}")

    except Exception as e:
        print(f"❌ Error {file}: {e}")

print("🔥 ALL FRAMES CLEANED")