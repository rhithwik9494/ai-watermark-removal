# Fix Plan: Watermark Still Visible + Video Quality Degraded

## Root Causes
1. **Default resize to 640x360** permanently destroys video resolution
2. **Detection too narrow** — only variance-based, misses many watermark types
3. **Hard rectangular mask** without padding/dilation misses watermark edges
4. **Tracking skips up to 5 frames** by copying original instead of inpainting
5. **No API format fallback** — only multipart, no base64 JSON fallback
6. **When detection fails, copies original video** instead of trying harder

## Implementation Steps

- [ ] 1. Fix `app/core/detection.py` — Multi-method detection + improved mask generation
- [ ] 2. Fix `app/core/tracking.py` — Lower fallback threshold + bbox margin
- [ ] 3. Fix `app/core/inpaint.py` — Dual API format support + mask dilation + better retries
- [ ] 4. Fix `app/core/video.py` — Better FFmpeg encoding settings
- [ ] 5. Fix `app/pipelines/process_video.py` — Default resize=None, better fallback logic
- [ ] 6. Fix `app/pipelines/process_batch.py` — Default resize=None
- [ ] 7. Fix `main.py` — Default --no-resize behavior
- [ ] 8. Fix `app/ui/gui.py` — Resize unchecked by default
- [ ] 9. Update `README.md` — Reflect new defaults

