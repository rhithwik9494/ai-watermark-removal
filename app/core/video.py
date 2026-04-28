"""
Video processing module using FFmpeg.
Handles frame extraction, video reconstruction, and audio preservation.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


class VideoProcessor:
    """
    FFmpeg-based video processing for watermark removal pipeline.
    
    Features:
    - Extract frames to PNG sequence
    - Rebuild video from PNG sequence
    - Preserve original audio stream
    - Get video metadata (fps, resolution, duration)
    """
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        """
        Initialize video processor.
        
        Args:
            ffmpeg_path: Path to ffmpeg executable
            ffprobe_path: Path to ffprobe executable
        """
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path
        
        # Verify ffmpeg is available
        if not shutil.which(self.ffmpeg):
            raise RuntimeError(
                f"FFmpeg not found: {self.ffmpeg}. "
                "Please install FFmpeg and add it to PATH."
            )
        
        logger.info(f"VideoProcessor initialized: ffmpeg={ffmpeg_path}")
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with fps, width, height, duration, frame_count
        """
        cmd = [
            self.ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
            "-of", "json",
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            
            # Parse FPS from fraction string
            fps_str = stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            
            info = {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "fps": fps,
                "duration": float(stream.get("duration", 0) or 0),
                "frame_count": int(stream.get("nb_frames", 0) or 0)
            }
            
            logger.debug(f"Video info: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            # Fallback using OpenCV
            return self._get_video_info_opencv(video_path)
    
    def _get_video_info_opencv(self, video_path: str) -> dict:
        """Fallback video info using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": 0
        }
        if info["fps"] > 0:
            info["duration"] = info["frame_count"] / info["fps"]
        cap.release()
        return info
    
    def extract_frames(self, 
                       video_path: str, 
                       output_dir: str,
                       resize: Optional[Tuple[int, int]] = None,
                       start_time: Optional[float] = None,
                       duration: Optional[float] = None) -> int:
        """
        Extract frames from video using FFmpeg.
        
        Args:
            video_path: Input video path
            output_dir: Directory to save frame images
            resize: Optional (width, height) to resize frames
            start_time: Optional start time in seconds
            duration: Optional duration in seconds
            
        Returns:
            Number of frames extracted
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean existing frames
        for f in Path(output_dir).glob("frame_*.png"):
            f.unlink()
        
        output_pattern = os.path.join(output_dir, "frame_%04d.png")
        
        cmd = [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        
        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])
        
        cmd.extend(["-i", video_path])
        
        if duration is not None:
            cmd.extend(["-t", str(duration)])
        
        cmd.extend(["-vsync", "vfr"])
        
        if resize is not None:
            cmd.extend(["-vf", f"scale={resize[0]}:{resize[1]}:flags=lanczos"])
        
        cmd.extend(["-pix_fmt", "rgb24", output_pattern])
        
        logger.info(f"Extracting frames from {video_path} to {output_dir}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Count extracted frames
            frame_files = sorted(Path(output_dir).glob("frame_*.png"))
            count = len(frame_files)
            
            logger.info(f"Extracted {count} frames to {output_dir}")
            return count
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg frame extraction failed: {e.stderr}")
            raise RuntimeError(f"Frame extraction failed: {e.stderr}")
    
    def rebuild_video(self,
                      frame_dir: str,
                      output_path: str,
                      fps: float = 30.0,
                      audio_source: Optional[str] = None,
                      codec: str = "libx264",
                      crf: int = 18,
                      preset: str = "medium") -> str:
        """
        Rebuild video from frame sequence using FFmpeg.
        
        Args:
            frame_dir: Directory containing frame_*.png sequence
            output_path: Output video path
            fps: Frames per second
            audio_source: Optional path to source video for audio extraction
            codec: Video codec
            crf: Constant rate factor (lower = higher quality)
            preset: Encoding preset (ultrafast to veryslow)
            
        Returns:
            Path to output video
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        frame_pattern = os.path.join(frame_dir, "frame_%04d.png")
        
        # Check if frames exist
        frame_files = sorted(Path(frame_dir).glob("frame_*.png"))
        if not frame_files:
            raise RuntimeError(f"No frames found in {frame_dir}")
        
        logger.info(f"Rebuilding video: {len(frame_files)} frames @ {fps}fps -> {output_path}")
        
        if audio_source and os.path.exists(audio_source):
            # Build with audio from original video
            # Use two-pass approach: build video first, then merge audio
            temp_video = output_path + ".temp.mp4"
            
            # Step 1: Build video from frames
            cmd_video = [
                self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", codec,
                "-crf", str(crf),
                "-preset", preset,
                "-pix_fmt", "yuv420p",
                "-an",  # No audio in first pass
                temp_video
            ]
            
            subprocess.run(cmd_video, capture_output=True, text=True, check=True)
            
            # Step 2: Merge with original audio
            cmd_merge = [
                self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", temp_video,
                "-i", audio_source,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path
            ]
            
            try:
                result = subprocess.run(cmd_merge, capture_output=True, text=True, check=True)
                logger.info(f"Video rebuilt with audio: {output_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Audio merge failed: {e.stderr}")
                # Fallback: just use temp video
                shutil.move(temp_video, output_path)
                logger.warning(f"Used video without audio due to merge error")
            finally:
                if os.path.exists(temp_video):
                    os.remove(temp_video)
        else:
            # Build without audio
            cmd = [
                self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", codec,
                "-crf", str(crf),
                "-preset", preset,
                "-pix_fmt", "yuv420p",
                "-an",
                output_path
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.info(f"Video rebuilt (no audio): {output_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Video rebuild failed: {e.stderr}")
                raise RuntimeError(f"Video rebuild failed: {e.stderr}")
        
        return output_path
    
    def extract_audio(self, video_path: str, output_path: str) -> str:
        """
        Extract audio stream from video.
        
        Args:
            video_path: Source video
            output_path: Output audio file (e.g., .aac, .wav)
            
        Returns:
            Path to extracted audio
        """
        cmd = [
            self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vn",  # No video
            "-c:a", "aac",
            "-b:a", "192k",
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Audio extracted: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio extraction failed: {e.stderr}")
            raise RuntimeError(f"Audio extraction failed: {e.stderr}")

