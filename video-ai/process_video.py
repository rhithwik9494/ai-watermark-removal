import cv2
import os

video_path = "input.mp4"
frames_dir = "frames"
output_dir = "clean_frames"

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 🎬 Extract frames
cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{frames_dir}/frame_{frame_count:04d}.png", frame)
    frame_count += 1

cap.release()

print(f"✅ Extracted {frame_count} frames")

# 🎬 Rebuild video (after cleaning)
print("⚠️ After cleaning frames, rebuilding video...")

os.system(
    "ffmpeg -framerate 30 -i clean_frames/frame_%04d.png -c:v libx264 output.mp4"
)

print("🎉 Output video saved as output.mp4")