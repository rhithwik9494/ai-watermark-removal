"""
End-to-end video watermark removal pipeline.
Orchestrates detection, tracking, inpainting, and video reconstruction.
"""

import os
import time
from pathlib import Path
from typing import Optional, Callable, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from app.utils.logger import get_logger
from app.utils.file_manager import FileManager
from app.core.detection import WatermarkDetector
from app.core.tracking import WatermarkTracker
from app.core.inpaint import IOPaintClient
from app.core.video import VideoProcessor

logger = get_logger(__name__)


class VideoRemovalPipeline:
    """
    Complete pipeline for removing watermarks from a single video.
    
    Pipeline stages:
    1. Extract frames using FFmpeg
    2. Auto-detect watermark region
    3. Track watermark across frames
    4. Generate masks dynamically
    5. Inpaint each frame via IOPaint API
    6. Rebuild video with original audio
    7. Cleanup temporary files
    """
    
    def __init__(self,
                 iopaint_url: str = "http://127.0.0.1:8080/api/v1/inpaint",
                 model: str = "lama",
                 resize: Optional[Tuple[int, int]] = (640, 360),
                 use_gpu: bool = False,
                 cleanup: bool = True):
        """
        Initialize pipeline.
        
        Args:
            iopaint_url: IOPaint API endpoint
            model: Inpainting model name
            resize: Resize frames before processing (width, height), None for original size
            use_gpu: Whether to use GPU acceleration (reserved for future)
            cleanup: Whether to remove temp files after processing
        """
        self.resize = resize
        self.use_gpu = use_gpu
        self.cleanup = cleanup
        
        # Initialize components
        self.detector = WatermarkDetector()
        self.tracker = WatermarkTracker()
        self.inpainter = IOPaintClient(api_url=iopaint_url, model=model)
        self.video_processor = VideoProcessor()
        self.file_manager = FileManager()
        
        logger.info(
            f"Pipeline initialized: model={model}, resize={resize}, gpu={use_gpu}"
        )
    
    def process(self,
                input_path: str,
                output_path: str,
                progress_callback: Optional[Callable[[int, int, str], None]] = None) -> str:
        """
        Process a single video to remove watermarks.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            progress_callback: Optional callback(current, total, status_message)
            
        Returns:
            Path to output video
        """
        start_time = time.time()
        
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Get video info
        logger.info(f"Processing video: {input_path}")
        video_info = self.video_processor.get_video_info(input_path)
        logger.info(
            f"Video info: {video_info['width']}x{video_info['height']} "
            f"@ {video_info['fps']:.2f}fps, {video_info['frame_count']} frames"
        )
        
        # Setup temp directories
        temp_dirs = self.file_manager.create_temp_dirs()
        frames_dir = temp_dirs["frames"]
        clean_frames_dir = temp_dirs["clean_frames"]
        
        try:
            # Stage 1: Extract frames
            self._report_progress(progress_callback, 0, 100, "Extracting frames...")
            frame_count = self.video_processor.extract_frames(
                input_path, frames_dir, resize=self.resize
            )
            
            # Stage 2: Detect watermark
            self._report_progress(progress_callback, 10, 100, "Detecting watermark...")
            bbox = self.detector.detect_from_frame_files(frames_dir)
            
            if not bbox:
                logger.warning("No watermark detected, copying original video")
                self.file_manager.safe_copy(input_path, output_path)
                return output_path
            
            # Stage 3: Process frames with tracking + inpainting
            self._process_frames(
                frames_dir, clean_frames_dir, bbox, frame_count, progress_callback
            )
            
            # Stage 4: Rebuild video
            self._report_progress(progress_callback, 90, 100, "Rebuilding video...")
            output_fps = video_info["fps"]
            self.video_processor.rebuild_video(
                clean_frames_dir, output_path, 
                fps=output_fps, audio_source=input_path
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Pipeline completed in {elapsed:.1f}s: {output_path}")
            
            self._report_progress(progress_callback, 100, 100, "Complete!")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            if self.cleanup:
                logger.info("Cleaning up temporary files")
                self.file_manager.cleanup_dirs([frames_dir, clean_frames_dir])
    
    def _process_frames(self,
                        frames_dir: str,
                        clean_frames_dir: str,
                        initial_bbox: Tuple[int, int, int, int],
                        total_frames: int,
                        progress_callback: Optional[Callable] = None):
        """
        Process all frames: track watermark, generate mask, inpaint.
        
        Args:
            frames_dir: Source frames directory
            clean_frames_dir: Output clean frames directory
            initial_bbox: Initial watermark bounding box
            total_frames: Total number of frames
            progress_callback: Progress callback
        """
        frame_files = self.file_manager.get_frame_files(frames_dir)
        
        if not frame_files:
            raise RuntimeError(f"No frames found in {frames_dir}")
        
        # Initialize tracker with first frame
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            raise RuntimeError(f"Cannot read first frame: {frame_files[0]}")
        
        self.tracker.initialize(first_frame, initial_bbox)
        
        # Process each frame
        for i, frame_path in enumerate(tqdm(frame_files, desc="Processing frames")):
            frame = cv2.imread(str(frame_path))
            
            if frame is None:
                logger.warning(f"Cannot read frame: {frame_path}")
                continue
            
            # Track watermark
            success, bbox = self.tracker.update(frame)
            
            if not success or bbox == (0, 0, 0, 0):
                logger.warning(f"Tracking failed for frame {i}, skipping inpaint")
                # Copy original frame
                out_path = Path(clean_frames_dir) / frame_path.name
                cv2.imwrite(str(out_path), frame)
                continue
            
            # Generate mask
            mask = self.detector.generate_mask(frame.shape, bbox)
            
            # Check if mask has any inpaint region
            if cv2.countNonZero(mask) == 0:
                logger.debug(f"Empty mask for frame {i}, copying original")
                out_path = Path(clean_frames_dir) / frame_path.name
                cv2.imwrite(str(out_path), frame)
                continue
            
            # Inpaint via API
            clean_frame = self.inpainter.inpaint(frame, mask)
            
            if clean_frame is None:
                logger.warning(f"Inpaint failed for frame {i}, using original")
                clean_frame = frame
            
            # Save clean frame
            out_path = Path(clean_frames_dir) / frame_path.name
            cv2.imwrite(str(out_path), clean_frame)
            
            # Report progress
            progress = int(10 + (i / total_frames) * 80) if total_frames > 0 else 10
            self._report_progress(
                progress_callback, progress, 100, 
                f"Processing frame {i+1}/{total_frames}"
            )
    
    def _report_progress(self, callback, current: int, total: int, message: str):
        """Safely call progress callback."""
        if callback:
            try:
                callback(current, total, message)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")
    
    def close(self):
        """Release resources."""
        self.inpainter.close()
        logger.info("Pipeline resources released")

