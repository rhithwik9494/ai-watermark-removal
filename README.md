# AI Video Watermark Removal System

A production-grade, modular, and automated video watermark removal system powered by OpenCV, FFmpeg, and IOPaint AI inpainting.

## Features

- **Automatic Watermark Detection** — No manual ROI required. Uses low-variance analysis across frames
- **Object Tracking** — OpenCV CSRT tracker follows watermark across all frames
- **AI Inpainting** — IOPaint LaMA model for high-quality watermark removal
- **Audio Preservation** — Original audio stream is preserved using FFmpeg
- **Batch Processing** — Process entire folders of videos
- **GUI Interface** — User-friendly Tkinter application
- **Performance Optimized** — Resizable frames, CPU-optimized, optional GPU support
- **Comprehensive Logging** — Structured logs with rotation
- **Google Colab Ready** — Notebook for cloud execution

---

## Project Architecture

```
ai-removal/
├── app/
│   ├── core/
│   │   ├── detection.py       # Auto watermark detection
│   │   ├── tracking.py          # CSRT tracker
│   │   ├── inpaint.py           # IOPaint API client
│   │   └── video.py             # FFmpeg video processing
│   ├── pipelines/
│   │   ├── process_video.py     # Single video pipeline
│   │   └── process_batch.py     # Batch processor
│   ├── ui/
│   │   └── gui.py               # Tkinter GUI
│   └── utils/
│       ├── logger.py            # Structured logging
│       └── file_manager.py      # File utilities
├── data/
│   ├── frames/                  # Temporary extracted frames
│   ├── clean_frames/            # Temporary clean frames
│   ├── videos/                  # Input video storage
│   └── outputs/                 # Output video storage
├── main.py                      # CLI entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Prerequisites

### Required Software

- **Python 3.9+**
- **FFmpeg** — Must be installed and available in PATH
- **IOPaint Server** — Running locally or remotely

### Installing FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
1. Download from https://ffmpeg.org/download.html
2. Extract and add `bin` folder to PATH
3. Verify: `ffmpeg -version`

### Installing IOPaint

```bash
pip install iopaint
iopaint start --model lama --device cpu --port 8080
```

For GPU:
```bash
iopaint start --model lama --device cuda --port 8080
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/rhithwik9494/ai-watermark-removal.git
cd ai-watermark-removal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Start IOPaint Server

```bash
iopaint start --model lama --device cpu --port 8080
```

### 2. Process a Single Video (CLI)

```bash
python main.py -i data/videos/input.mp4 -o data/outputs/cleaned.mp4
```

### 3. Process a Batch of Videos

```bash
python main.py --batch -i data/videos -o data/outputs --workers 2
```

### 4. Launch GUI

```bash
python main.py --gui
```

---

## CLI Options

```
Options:
  -h, --help            Show help message
  -i, --input           Input video file path
  -o, --output          Output video file path
  --batch               Enable batch processing mode
  --input-dir           Input directory (batch mode)
  --output-dir          Output directory (batch mode)
  --api-url             IOPaint API URL (default: http://127.0.0.1:8080/api/v1/inpaint)
  --model               Model: lama (default), ldm, mat
  --resize W H          Resize frames (default: 640 360)
  --no-resize           Process at original resolution
  --gpu                 Enable GPU acceleration
  --workers N           Parallel workers for batch mode (default: 1)
  --no-cleanup          Keep temporary frame files
  --gui                 Launch GUI
  -v, --verbose         Enable verbose logging
```

---

## How It Works

### 1. Frame Extraction
FFmpeg extracts frames from the input video as PNG sequence.

### 2. Watermark Detection
The system analyzes the first 15 frames:
- Converts frames to grayscale
- Computes variance map across frames
- Static elements (watermarks) have low variance
- Thresholds and morphology operations isolate the watermark
- Returns bounding box of largest low-variance region

### 3. Tracking
OpenCV CSRT tracker is initialized with the detected bounding box:
- Tracks the watermark region across all frames
- Falls back to last known position if tracking fails
- Periodically updates fallback position every 30 frames

### 4. Mask Generation
For each frame, a binary mask is created:
- Black background (0)
- White filled rectangle (255) at tracked location

### 5. AI Inpainting
Each frame + mask pair is sent to IOPaint API:
- Multipart form: `files={"image": ..., "mask": ...}`
- Model parameter: `data={"model": "lama"}`
- Response is raw binary image bytes
- Decoded with `np.frombuffer` + `cv2.imdecode`

### 6. Video Reconstruction
FFmpeg rebuilds video from cleaned frames:
- Preserves original FPS
- Merges original audio stream using `-map 0:v:0 -map 1:a:0 -shortest`
- H.264 encoding with high quality (CRF 18)

---

## Performance Optimization

### Frame Resizing
Resize frames to 640x360 (or your preference) before processing:
```bash
python main.py -i video.mp4 --resize 640 360
```

### CPU-Optimized Model
The default `lama` model is optimized for CPU inference.

### Parallel Processing
Process multiple videos simultaneously:
```bash
python main.py --batch -i ./videos -o ./output --workers 4
```

### In-Memory Processing (Future Enhancement)
For minimal disk I/O, frames can be processed in memory (currently writes to disk).

---

## Google Colab

Use the provided notebook `AI_Watermark_Removal.ipynb` for cloud execution:

1. Upload to Google Colab
2. Install IOPaint in the notebook
3. Upload video via UI or mount Google Drive
4. Run all cells
5. Download result

---

## Troubleshooting

### "FFmpeg not found"
Install FFmpeg and ensure it's in your system PATH.

### "Connection error to IOPaint API"
Start IOPaint server: `iopaint start --model lama --port 8080`

### "No watermark detected automatically"
The video may not have a static watermark, or the watermark blends too well.
Try adjusting detection parameters or processing at original resolution.

### "Tracking failed"
The watermark may move or disappear in some frames. The system falls back to last known position.

### Slow processing
Use `--resize` to reduce resolution, or ensure IOPaint is running on GPU.

---

## Configuration

Detection parameters can be adjusted in `app/core/detection.py`:

```python
detector = WatermarkDetector(
    sample_frames=15,          # Frames analyzed for detection
    variance_threshold=15.0,    # Low variance threshold
    min_watermark_area=500.0,   # Minimum watermark size
    morph_kernel_size=5,        # Morphology kernel size
    dilation_iterations=3       # Dilation strength
)
```

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

---

## License

This project is open source. See LICENSE file for details.

---

## Acknowledgments

- [IOPaint](https://github.com/Sanster/IOPaint) — AI inpainting engine
- [LaMA](https://github.com/advimman/lama) — Large Mask Inpainting model
- OpenCV — Computer vision library
- FFmpeg — Video processing toolkit

