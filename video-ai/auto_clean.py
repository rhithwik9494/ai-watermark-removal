import cv2
import os
import requests
import numpy as np

frames_dir = "frames"
output_dir = "clean_frames"

os.makedirs(output_dir, exist_ok=True)

frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])

def detect_watermark(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bright areas (common watermark style)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Remove noise
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(frame)

    found = False

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Ignore tiny areas
        if w * h > 500:
            cv2.rectangle(mask, (x, y), (x+w, y+h), (255,255,255), -1)
            found = True

    return mask, found


for file in frame_files:
    frame_path = os.path.join(frames_dir, file)
    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    mask, found = detect_watermark(frame)

    # If nothing detected → copy frame
    if not found:
        cv2.imwrite(os.path.join(output_dir, file), frame)
        print(f"➡️ No watermark: copied {file}")
        continue

    # Save temp files
    cv2.imwrite("temp.png", frame)
    cv2.imwrite("mask.png", mask)

    try:
        with open("temp.png", "rb") as img_file, open("mask.png", "rb") as mask_file:
            response = requests.post(
                "http://127.0.0.1:8080/api/v1/inpaint",
                files={
                    "image": img_file,
                    "mask": mask_file
                },
                data={"model": "lama"},
                timeout=60
            )

        if response.status_code != 200:
            print(f"❌ API Error {response.status_code}")
            continue

        result = np.frombuffer(response.content, np.uint8)
        result_img = cv2.imdecode(result, cv2.IMREAD_COLOR)

        if result_img is not None:
            cv2.imwrite(os.path.join(output_dir, file), result_img)
            print(f"🔥 Cleaned {file}")
        else:
            print(f"⚠️ Failed decoding {file}")

    except Exception as e:
        print(f"❌ Error: {e}")

print("🚀 AUTO CLEAN COMPLETED")