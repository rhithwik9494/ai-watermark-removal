import os
import shutil
import glob
from pathlib import Path
from typing import Optional, List

from app.utils.logger import get_logger

logger = get_logger(__name__)


class FileManager:
    """
    Manages file system operations for the watermark removal pipeline.
    Handles temp directories, cleanup, path validation, and file organization.
    """
    
    def __init__(self, 
                 base_dir: str = ".",
                 frames_dir: str = "data/frames",
                 clean_frames_dir: str = "data/clean_frames", 
                 videos_dir: str = "data/videos",
                 outputs_dir: str = "data/outputs"):
        self.base_dir = Path(base_dir).resolve()
        self.frames_dir = self.base_dir / frames_dir
        self.clean_frames_dir = self.base_dir / clean_frames_dir
        self.videos_dir = self.base_dir / videos_dir
        self.outputs_dir = self.base_dir / outputs_dir
        
        self._ensure_directories()
        logger.info(f"FileManager initialized. Base: {self.base_dir}")
    
    def _ensure_directories(self):
        """Create all required directories."""
        for directory in [self.frames_dir, self.clean_frames_dir, 
                         self.videos_dir, self.outputs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory: {directory}")
    
    def clean_temp_frames(self):
        """Remove all temporary frame directories."""
        for directory in [self.frames_dir, self.clean_frames_dir]:
            if directory.exists():
                shutil.rmtree(directory)
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cleaned temp directory: {directory}")
    
    def get_video_list(self, folder: Optional[str] = None) -> List[Path]:
        """
        Get list of video files from a folder.
        
        Args:
            folder: Folder to scan (default: self.videos_dir)
            
        Returns:
            List of video file paths
        """
        target = Path(folder) if folder else self.videos_dir
        if not target.exists():
            logger.warning(f"Video directory not found: {target}")
            return []
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        videos = [f for f in target.iterdir() 
                  if f.is_file() and f.suffix.lower() in video_extensions]
        videos.sort()
        logger.info(f"Found {len(videos)} video(s) in {target}")
        return videos
    
    def get_frame_files(self, directory: Optional[str] = None) -> List[Path]:
        """
        Get sorted list of frame files.
        
        Args:
            directory: Directory to scan (default: self.frames_dir)
            
        Returns:
            List of frame file paths
        """
        target = Path(directory) if directory else self.frames_dir
        if not target.exists():
            return []
        
        frames = sorted(target.glob("frame_*.png"))
        logger.debug(f"Found {len(frames)} frame(s) in {target}")
        return frames
    
    def generate_output_path(self, input_path: str, suffix: str = "_cleaned") -> Path:
        """
        Generate output video path based on input name.
        
        Args:
            input_path: Input video path
            suffix: Suffix to append to filename
            
        Returns:
            Output path in outputs directory
        """
        input_file = Path(input_path)
        output_name = f"{input_file.stem}{suffix}{input_file.suffix}"
        output_path = self.outputs_dir / output_name
        logger.debug(f"Output path: {output_path}")
        return output_path
    
    def get_video_fps(self, video_path: str) -> float:
        """
        Get FPS of a video file using OpenCV.
        
        Args:
            video_path: Path to video file
            
        Returns:
            FPS value (default 30.0 if cannot detect)
        """
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if fps <= 0:
            logger.warning(f"Could not detect FPS for {video_path}, using default 30.0")
            return 30.0
        
        logger.debug(f"Detected FPS: {fps} for {video_path}")
        return fps
    
    def get_video_resolution(self, video_path: str) -> tuple:
        """
        Get resolution of a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            (width, height) tuple
        """
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        logger.debug(f"Detected resolution: {width}x{height} for {video_path}")
        return (width, height)
    
    def create_temp_dirs(self) -> dict:
        """
        Create temporary directories for frame processing.
        
        Returns:
            Dictionary with 'frames' and 'clean_frames' paths
        """
        import tempfile
        import uuid
        
        base = Path(tempfile.gettempdir()) / f"ai_watermark_{uuid.uuid4().hex[:8]}"
        frames_dir = base / "frames"
        clean_dir = base / "clean_frames"
        
        frames_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Created temp dirs: {base}")
        return {
            "frames": str(frames_dir),
            "clean_frames": str(clean_dir)
        }
    
    def cleanup_dirs(self, directories: list):
        """
        Remove temporary directories.
        
        Args:
            directories: List of directory paths to remove
        """
        for directory in directories:
            try:
                d = Path(directory)
                if d.exists():
                    shutil.rmtree(d)
                    logger.debug(f"Cleaned up: {directory}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {directory}: {e}")
    
    @staticmethod
    def safe_copy(src: str, dst: str):
        """
        Safely copy a file, creating directories if needed.
        
        Args:
            src: Source file path
            dst: Destination file path
        """
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info(f"Copied {src} -> {dst}")
    
    @staticmethod
    def validate_video_path(path: str) -> bool:
        """
        Validate that a video file exists and is readable.
        
        Args:
            path: Path to check
            
        Returns:
            True if valid
        """
        p = Path(path)
        if not p.exists():
            logger.error(f"Video file does not exist: {path}")
            return False
        if not p.is_file():
            logger.error(f"Path is not a file: {path}")
            return False
        if p.suffix.lower() not in {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}:
            logger.warning(f"Unusual video extension: {p.suffix}")
        return True

