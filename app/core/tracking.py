"""
Watermark tracking module using OpenCV CSRT tracker.
Tracks watermark region across frames with fallback support.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)


class WatermarkTracker:
    """
    CSRT-based tracker for watermark regions across video frames.
    
    Features:
    - Automatic tracker initialization with detected bounding box
    - Frame-by-frame tracking with quality assessment
    - Fallback to last known good bounding box on tracking failure
    - Adaptive re-initialization when tracking confidence drops
    """
    
    def __init__(self, fallback_threshold: int = 1, bbox_margin: int = 10):
        """
        Initialize tracker.
        
        Args:
            fallback_threshold: Number of consecutive failures before using fallback bbox
            bbox_margin: Extra pixels to add around tracked bbox for safety
        """
        self.tracker = None
        self.initialized = False
        self.last_bbox = None
        self.fallback_bbox = None
        self.consecutive_failures = 0
        self.fallback_threshold = fallback_threshold
        self.bbox_margin = bbox_margin
        self.frame_count = 0
        
    def _create_tracker(self):
        """Create best available OpenCV tracker with auto-fallback."""
        # Priority order: CSRT (best for occlusion) > KCF (fast) > MIL (stable)
        tracker_factories = [
            ('CSRT', getattr(cv2, 'TrackerCSRT_create', None)),
            ('KCF', getattr(cv2, 'TrackerKCF_create', None)),
            ('MIL', getattr(cv2, 'TrackerMIL_create', None)),
        ]
        
        for name, factory in tracker_factories:
            if factory is None:
                logger.debug(f"{name} tracker not available")
                continue
            try:
                tracker = factory()
                logger.info(f"Using {name} tracker")
                return tracker, name
            except cv2.error as e:
                logger.debug(f"{name} tracker failed: {e}")
                continue
        
        logger.warning("No OpenCV tracker available. Will use static bbox fallback only.")
        return None, None
    
    def initialize(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Initialize tracker with first frame and bounding box.
        
        Args:
            frame: First video frame (BGR)
            bbox: (x, y, w, h) bounding box of watermark region
            
        Returns:
            True if initialization successful
        """
        try:
            # Try to create best available tracker
            self.tracker, tracker_name = self._create_tracker()
            if self.tracker is None:
                # No tracker available - use static bbox fallback
                h, w = frame.shape[:2]
                x, y, bw, bh = bbox
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                bw = min(bw, w - x)
                bh = min(bh, h - y)
                
                if bw < 10 or bh < 10:
                    logger.error(f"Bounding box too small: {bw}x{bh}")
                    return False
                
                bbox = (x, y, bw, bh)
                self.initialized = True
                self.last_bbox = bbox
                self.fallback_bbox = bbox
                self.frame_count = 0
                logger.info(f"Static fallback tracker initialized with bbox: {bbox}")
                return True
            
            # Ensure bbox is valid
            h, w = frame.shape[:2]
            x, y, bw, bh = bbox
            
            # Clamp to frame bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = min(bw, w - x)
            bh = min(bh, h - y)
            
            if bw < 10 or bh < 10:
                logger.error(f"Bounding box too small: {bw}x{bh}")
                return False
                
            bbox = (x, y, bw, bh)
            
            # OpenCV 4.x init() returns None on success, not True/False
            try:
                self.tracker.init(frame, bbox)
                self.initialized = True
                self.last_bbox = bbox
                self.fallback_bbox = bbox
                self.frame_count = 0
                logger.info(f"Tracker initialized with bbox: {bbox}")
                return True
            except Exception as e:
                logger.error(f"Tracker initialization failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing tracker: {e}")
            return False
    
    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        """
        Update tracker with new frame.
        
        Args:
            frame: Current video frame (BGR)
            
        Returns:
            Tuple of (success, bbox) where bbox is (x, y, w, h)
        """
        if not self.initialized or self.tracker is None:
            logger.warning("Tracker not initialized, using fallback bbox")
            if self.fallback_bbox:
                return True, self.fallback_bbox
            return False, (0, 0, 0, 0)
        
        try:
            success, bbox = self.tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                
                # Validate tracking quality
                frame_h, frame_w = frame.shape[:2]
                
                # Check if bbox is within reasonable bounds
                if x < 0 or y < 0 or w < 5 or h < 5:
                    success = False
                elif x + w > frame_w or y + h > frame_h:
                    success = False
                elif w > frame_w * 0.5 or h > frame_h * 0.5:
                    # Watermark shouldn't be more than half the frame
                    success = False
                else:
                    # Apply margin for safety
                    frame_h, frame_w = frame.shape[:2]
                    mx = max(0, x - self.bbox_margin)
                    my = max(0, y - self.bbox_margin)
                    mw = min(w + 2 * self.bbox_margin, frame_w - mx)
                    mh = min(h + 2 * self.bbox_margin, frame_h - my)
                    
                    self.last_bbox = (mx, my, mw, mh)
                    self.consecutive_failures = 0
                    self.frame_count += 1
                    
                    # Periodically update fallback (every 30 frames)
                    if self.frame_count % 30 == 0:
                        self.fallback_bbox = self.last_bbox
                        
                    return True, self.last_bbox
            
            # Tracking failed
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= self.fallback_threshold:
                logger.warning(
                    f"Tracking failed {self.consecutive_failures} times, "
                    f"using fallback bbox: {self.fallback_bbox}"
                )
                if self.fallback_bbox:
                    return True, self.fallback_bbox
            else:
                logger.debug(
                    f"Tracking failure #{self.consecutive_failures}, "
                    f"using last bbox: {self.last_bbox}"
                )
                if self.last_bbox:
                    return True, self.last_bbox
            
            return False, (0, 0, 0, 0)
            
        except Exception as e:
            logger.error(f"Error during tracking update: {e}")
            self.consecutive_failures += 1
            if self.fallback_bbox:
                return True, self.fallback_bbox
            return False, (0, 0, 0, 0)
    
    def reinitialize(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Re-initialize tracker with new bounding box.
        
        Args:
            frame: Current frame
            bbox: New bounding box
            
        Returns:
            True if re-initialization successful
        """
        logger.info(f"Re-initializing tracker with bbox: {bbox}")
        self.tracker = None
        self.initialized = False
        self.consecutive_failures = 0
        return self.initialize(frame, bbox)
    
    def get_last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get last known bounding box."""
        return self.last_bbox or self.fallback_bbox
    
    def reset(self):
        """Reset tracker state."""
        self.tracker = None
        self.initialized = False
        self.last_bbox = None
        self.fallback_bbox = None
        self.consecutive_failures = 0
        self.frame_count = 0
        logger.info("Tracker reset")

