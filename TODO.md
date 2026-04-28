# Fix Plan: AI Video Watermark Removal System

## Steps

- [x] 1. Fix `app/core/detection.py` — Fix double-padding bug and variance threshold logic
- [x] 2. Fix `app/utils/logger.py` — Fix ColoredFormatter mutating record.levelname
- [x] 3. Fix `main.py` — Default to no-resize (original resolution)
- [x] 4. Fix `app/ui/gui.py` — Default resize_enabled to False
- [x] 5. Fix `app/pipelines/process_video.py` — Better fallback when detection fails
- [x] 6. Fix `app/core/tracking.py` — Lower fallback_threshold, apply bbox_margin
- [x] 7. Fix `app/core/inpaint.py` — Reuse encoded buffers in base64 fallback
- [x] 8. Update `README.md` — Reflect new no-resize default
- [x] 9. Update `AI_Watermark_Removal.ipynb` — Make resize optional (None by default)
- [x] 10. Test & verify all imports and CLI work correctly

