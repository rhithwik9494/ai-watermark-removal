import cv2
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

from app.utils.logger import get_logger

logger = get_logger(__name__)


class WatermarkDetector:
    """
    Automatic watermark detection using multi-method analysis.
    
    Detects static watermarks by trying multiple strategies:
    1. Low-variance analysis (static elements)
    2. Edge-enhanced detection (text/logos with borders)
    3. Brightness/contrast threshold (white/light watermarks)
    4. Bottom-right fallback (most common watermark location)
    """
    
    def __init__(self, 
                 sample_frames: int = 15,
                 variance_threshold: float = 15.0,
                 min_watermark_area: float = 500.0,
                 morph_kernel_size: int = 5,
                 dilation_iterations: int = 3,
                 mask_padding: int = 15,
                 mask_dilation: int = 8):
        """
        Initialize detector parameters.
        
        Args:
            sample_frames: Number of frames to sample for variance analysis
            variance_threshold: Pixels with variance below this are considered static
            min_watermark_area: Minimum area for a valid watermark region
            morph_kernel_size: Kernel size for morphology operations
            dilation_iterations: Number of dilation iterations
            mask_padding: Extra pixels to add around detected bbox
            mask_dilation: Morphological dilation pixels for mask edges
        """
        self.sample_frames = sample_frames
        self.variance_threshold = variance_threshold
        self.min_watermark_area = min_watermark_area
        self.morph_kernel_size = morph_kernel_size
        self.dilation_iterations = dilation_iterations
        self.mask_padding = mask_padding
        self.mask_dilation = mask_dilation
    
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
        Multi-method detection. Tries several strategies in order.
        
        Returns:
            Bounding box as (x, y, w, h) or None
        """
        logger.debug(f"Running multi-method detection on {len(gray_frames)} frames")
        
        # Method 1: Low-variance analysis (static elements)
        bbox = self._detect_by_variance(gray_frames)
        if bbox:
            logger.info("Detection method: variance-based")
            return bbox
        
        # Method 2: Edge-enhanced detection
        bbox = self._detect_by_edges(gray_frames)
        if bbox:
            logger.info("Detection method: edge-enhanced")
            return bbox
        
        # Method 3: Brightness threshold (bright watermarks)
        bbox = self._detect_by_brightness(gray_frames)
        if bbox:
            logger.info("Detection method: brightness-based")
            return bbox
        
        # Method 4: Bottom-right fallback (most common location)
        bbox = self._detect_bottom_right(gray_frames[0])
        if bbox:
            logger.info("Detection method: bottom-right fallback")
            return bbox
        
        logger.warning("All detection methods failed")
        return None
    
    def _detect_by_variance(self, gray_frames: list) -> Optional[Tuple[int, int, int, int]]:
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
        
        # Threshold: low variance = static = watermark candidate
        # Threshold the RAW variance map (not normalized) for semantically correct behavior
        binary = np.where(variance_map < self.variance_threshold, 255, 0).astype(np.uint8)
        
        # Morphology: open to remove noise, then dilate to fill gaps
        kernel_open = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        kernel_dilate = np.ones((self.morph_kernel_size * 2, self.morph_kernel_size * 2), np.uint8)
        
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        dilated = cv2.dilate(opened, kernel_dilate, iterations=self.dilation_iterations)
        
        return self._contour_to_bbox(dilated, "variance")
    
    def _detect_by_edges(self, gray_frames: list) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect watermarks using edge analysis. Good for text/logos with sharp borders.
        """
        logger.debug("Trying edge-enhanced detection")
        
        # Use first frame for edge detection
        frame = gray_frames[0]
        
        # Compute edges
        edges = cv2.Canny(frame, 50, 150)
        
        # Dilate edges to connect nearby components
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        return self._contour_to_bbox(edges_dilated, "edge")
    
    def _detect_by_brightness(self, gray_frames: list) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect bright watermark regions. Good for white/light watermarks.
        """
        logger.debug("Trying brightness-based detection")
        
        # Compute mean brightness across frames
        stack = np.stack(gray_frames, axis=0).astype(np.float32)
        mean_frame = np.mean(stack, axis=0).astype(np.uint8)
        
        # Detect bright areas
        _, bright = cv2.threshold(mean_frame, 200, 255, cv2.THRESH_BINARY)
        
        # Also detect high-contrast edges in bright regions
        grad_x = cv2.Sobel(mean_frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mean_frame, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2).astype(np.uint8)
        _, grad_thresh = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
        
        # Combine bright regions with edges
        combined = cv2.bitwise_and(bright, grad_thresh)
        
        # Dilate to fill gaps
        kernel = np.ones((7, 7), np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=2)
        
        return self._contour_to_bbox(combined, "brightness")
    
    def _detect_bottom_right(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Fallback: assume watermark is in bottom-right corner.
        This is the most common watermark placement.
        """
        logger.debug("Using bottom-right fallback detection")
        
        h, w = frame.shape[:2]
        
        # Define a region in the bottom-right (roughly 1/4 of frame)
        region_w = w // 4
        region_h = h // 6
        x = w - region_w - 10
        y = h - region_h - 10
        
        # Ensure valid coordinates
        x = max(0, x)
        y = max(0, y)
        region_w = min(region_w, w - x)
        region_h = min(region_h, h - y)
        
        if region_w < 20 or region_h < 20:
            return None
        
        logger.info(f"Bottom-right fallback bbox: ({x}, {y}, {region_w}, {region_h})")
        return (x, y, region_w, region_h)
    
    def _contour_to_bbox(self, binary: np.ndarray, method: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Convert binary image to bounding box from largest contour.
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.debug(f"No contours found for method: {method}")
            return None
        
        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < self.min_watermark_area:
            logger.debug(f"Largest contour area {area} below minimum {self.min_watermark_area} for method: {method}")
            return None
        
        # Get bounding box (raw — padding is added in generate_mask)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        logger.debug(f"Detected watermark bbox ({method}): ({x}, {y}, {w}, {h}), area={area}")
        
        return (x, y, w, h)
    
    def generate_mask(self, frame_shape: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Generate binary mask from bounding box with padding and dilation.
        
        Args:
            frame_shape: (height, width) of frame
            bbox: (x, y, w, h) bounding box
            
        Returns:
            Binary mask with white (255) region on black background
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x, y, bw, bh = bbox
        
        # Add padding
        x1 = max(0, x - self.mask_padding)
        y1 = max(0, y - self.mask_padding)
        x2 = min(w, x + bw + self.mask_padding)
        y2 = min(h, y + bh + self.mask_padding)
        
        # Draw white filled rectangle
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Dilate mask to ensure full watermark coverage
        if self.mask_dilation > 0:
            kernel = np.ones((self.mask_dilation, self.mask_dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        logger.debug(f"Generated mask for bbox: ({x1}, {y1}, {x2-x1}, {y2-y1}) with dilation={self.mask_dilation}")
        
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

