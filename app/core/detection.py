import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from app.utils.logger import get_logger

logger = get_logger(__name__)


class WatermarkDetector:
    """
    Automatic watermark detection using low-variance analysis.
    
    Detects static watermarks by analyzing variance across multiple frames.
    Static elements (watermarks) have low variance compared to moving content.
    """
    
    def __init__(self, 
                 sample_frames: int = 15,
                 variance_threshold: float = 15.0,
                 min_watermark_area: float = 500.0,
                 morph_kernel_size: int = 5,
                 dilation_iterations: int = 3):
        """
        Initialize detector parameters.
        
        Args:
            sample_frames: Number of frames to sample for variance analysis
            variance_threshold: Pixels with variance below this are considered static
            min_watermark_area: Minimum area for a valid watermark region
            morph_kernel_size: Kernel size for morphology operations
            dilation_iterations: Number of dilation iterations
        """
        self.sample_frames = sample_frames
        self.variance_threshold = variance_threshold
        self.min_watermark_area = min_watermark_area
        self.morph_kernel_size = morph_kernel_size
        self.dilation_iterations = dilation_iterations
    
    def detect_from_video(self, video_path: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect watermark bounding box from a video file.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Bounding box as (x, y, w, h) or None if detection fails
        """
        logger.info(f"Starting watermark detection on: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_count = min(self.sample_frames, total_frames)
        
        if total_frames < 5:
            logger.warning("Video too short for reliable detection")
            cap.release()
            return None
        
        # Sample frames evenly distributed
        frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
        
        cap.release()
        
        if len(frames) < 3:
            logger.error("Could not read enough frames for detection")
            return None
        
        bbox = self._detect_from_frames(frames)
        
        if bbox:
            x, y, w, h = bbox
            logger.info(f"Watermark detected at: x={x}, y={y}, w={w}, h={h}")
        else:
            logger.warning("No watermark detected automatically")
        
        return bbox
    
    def detect_from_frame_files(self, frame_dir: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect watermark from a directory of frame images.
        
        Args:
            frame_dir: Directory containing frame images
            
        Returns:
            Bounding box as (x, y, w, h) or None
        """
        from app.utils.file_manager import FileManager
        
        logger.info(f"Starting watermark detection from frames in: {frame_dir}")
        
        fm = FileManager()
        frames = fm.get_frame_files(frame_dir)
        
        if not frames:
            logger.error(f"No frames found in {frame_dir}")
            return None
        
        sample = frames[:self.sample_frames]
        gray_frames = []
        
        for frame_path in sample:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frames.append(gray)
        
        if len(gray_frames) < 3:
            logger.error("Could not read enough frames")
            return None
        
        bbox = self._detect_from_frames(gray_frames)
        
        if bbox:
            x, y, w, h = bbox
            logger.info(f"Watermark detected at: x={x}, y={y}, w={w}, h={h}")
        else:
            logger.warning("No watermark detected automatically")
        
        return bbox
    
    def _detect_from_frames(self, gray_frames: list) -> Optional[Tuple[int, int, int, int]]:
        """
        Core detection logic using low-variance analysis.
        
        Algorithm:
        1. Stack all grayscale frames into a 3D array
        2. Compute variance map across frames (axis=0)
        3. Threshold low-variance pixels (static = watermark candidates)
        4. Apply morphology: open (remove noise) + dilate (fill gaps)
        5. Find largest contour as watermark region
        6. Return bounding box
        """
        logger.debug(f"Analyzing {len(gray_frames)} frames for low-variance regions")
        
        # Stack frames: shape (N, H, W)
        stack = np.stack(gray_frames, axis=0).astype(np.float32)
        
        # Compute variance across frames for each pixel
        variance_map = np.var(stack, axis=0)
        
        # Normalize for visualization/debugging
        var_norm = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Threshold: low variance = static = watermark candidate
        # Pixels with variance below threshold are likely watermarks
        _, binary = cv2.threshold(
            var_norm, 
            int(self.variance_threshold), 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        # Morphology: open to remove noise, then dilate to fill gaps
        kernel_open = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        kernel_dilate = np.ones((self.morph_kernel_size * 2, self.morph_kernel_size * 2), np.uint8)
        
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        dilated = cv2.dilate(opened, kernel_dilate, iterations=self.dilation_iterations)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.debug("No contours found after morphology")
            return None
        
        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < self.min_watermark_area:
            logger.debug(f"Largest contour area {area} below minimum {self.min_watermark_area}")
            return None
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add small padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        
        logger.debug(f"Detected watermark bbox: ({x}, {y}, {w}, {h}), area={area}")
        
        return (x, y, w, h)
    
    def generate_mask(self, frame_shape: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Generate binary mask from bounding box.
        
        Args:
            frame_shape: (height, width) of frame
            bbox: (x, y, w, h) bounding box
            
        Returns:
            Binary mask with white (255) rectangle on black background
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x, y, bw, bh = bbox
        
        # Clamp to frame bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        
        # Draw white filled rectangle
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        logger.debug(f"Generated mask for bbox: ({x1}, {y1}, {x2-x1}, {y2-y1})")
        
        return mask


def detect_watermark_by_threshold(frame: np.ndarray,
                                  threshold: int = 220,
                                  min_area: int = 500) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """
    Simple fallback detection using brightness threshold.
    Useful for bright watermarks on dark backgrounds.
    
    Args:
        frame: Input frame (BGR)
        threshold: Brightness threshold
        min_area: Minimum contour area
        
    Returns:
        Tuple of (bbox, mask) where bbox is (x, y, w, h)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect bright areas
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(frame)
    
    if not contours:
        return ((0, 0, 0, 0), mask)
    
    # Find largest contour
    largest = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest) < min_area:
        return ((0, 0, 0, 0), mask)
    
    x, y, w, h = cv2.boundingRect(largest)
    cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
    
    return ((x, y, w, h), mask)

