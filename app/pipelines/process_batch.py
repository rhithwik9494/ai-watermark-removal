"""
Batch processing pipeline for multiple videos.
Supports sequential and parallel processing with progress tracking.
"""

import os
from pathlib import Path
from typing import List, Optional, Callable
import concurrent.futures
from tqdm import tqdm

from app.utils.logger import get_logger
from app.pipelines.process_video import VideoRemovalPipeline

logger = get_logger(__name__)


class BatchProcessor:
    """
    Batch processor for watermark removal on multiple videos.
    
    Features:
    - Process all videos in a directory
    - Sequential or parallel processing
    - Automatic output naming
    - Per-video and overall progress tracking
    - Summary report generation
    """
    
    def __init__(self,
                 iopaint_url: str = "http://127.0.0.1:8080/api/v1/inpaint",
                 model: str = "lama",
                 max_workers: int = 1,
                 **pipeline_kwargs):
        """
        Initialize batch processor.
        
        Args:
            iopaint_url: IOPaint API endpoint
            model: Inpainting model
            max_workers: Number of parallel workers (1 = sequential)
            **pipeline_kwargs: Additional args for VideoRemovalPipeline
        """
        self.iopaint_url = iopaint_url
        self.model = model
        self.max_workers = max_workers
        self.pipeline_kwargs = pipeline_kwargs
        
        logger.info(f"BatchProcessor initialized: max_workers={max_workers}, model={model}")
    
    def process_directory(self,
                          input_dir: str,
                          output_dir: str,
                          video_extensions: tuple = (".mp4", ".avi", ".mov", ".mkv", ".wmv"),
                          progress_callback: Optional[Callable[[int, int, str], None]] = None) -> dict:
        """
        Process all videos in a directory.
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory for output videos
            video_extensions: Tuple of valid video extensions
            progress_callback: Overall progress callback(current, total, message)
            
        Returns:
            Summary dictionary with results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_dir.glob(f"*{ext}"))
            video_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        video_files = sorted(list(set(video_files)))
        
        if not video_files:
            logger.warning(f"No videos found in {input_dir}")
            return {"total": 0, "successful": 0, "failed": 0, "results": []}
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        results = []
        
        if self.max_workers > 1:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for video_path in video_files:
                    output_path = output_dir / f"{video_path.stem}_cleaned{video_path.suffix}"
                    future = executor.submit(
                        self._process_single,
                        str(video_path),
                        str(output_path)
                    )
                    futures[future] = video_path
                
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    video_path = futures[future]
                    try:
                        result = future.result()
                        results.append({
                            "input": str(video_path),
                            "output": result,
                            "status": "success"
                        })
                        logger.info(f"Completed: {video_path.name}")
                    except Exception as e:
                        results.append({
                            "input": str(video_path),
                            "output": None,
                            "status": "failed",
                            "error": str(e)
                        })
                        logger.error(f"Failed: {video_path.name} - {e}")
                    
                    progress = int((i + 1) / len(video_files) * 100)
                    self._report_progress(
                        progress_callback, progress, 100,
                        f"Processed {i+1}/{len(video_files)}: {video_path.name}"
                    )
        else:
            # Sequential processing
            for i, video_path in enumerate(video_files):
                output_path = output_dir / f"{video_path.stem}_cleaned{video_path.suffix}"
                
                self._report_progress(
                    progress_callback, int(i / len(video_files) * 100), 100,
                    f"Processing {i+1}/{len(video_files)}: {video_path.name}"
                )
                
                try:
                    result = self._process_single(str(video_path), str(output_path))
                    results.append({
                        "input": str(video_path),
                        "output": result,
                        "status": "success"
                    })
                    logger.info(f"Completed: {video_path.name}")
                except Exception as e:
                    results.append({
                        "input": str(video_path),
                        "output": None,
                        "status": "failed",
                        "error": str(e)
                    })
                    logger.error(f"Failed: {video_path.name} - {e}")
                
                progress = int((i + 1) / len(video_files) * 100)
                self._report_progress(
                    progress_callback, progress, 100,
                    f"Processed {i+1}/{len(video_files)}"
                )
        
        # Generate summary
        summary = {
            "total": len(results),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "results": results
        }
        
        logger.info(
            f"Batch complete: {summary['successful']}/{summary['total']} successful"
        )
        
        return summary
    
    def _process_single(self, input_path: str, output_path: str) -> str:
        """
        Process a single video.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            
        Returns:
            Path to output video
        """
        pipeline = VideoRemovalPipeline(
            iopaint_url=self.iopaint_url,
            model=self.model,
            **self.pipeline_kwargs
        )
        
        try:
            result = pipeline.process(input_path, output_path)
            return result
        finally:
            pipeline.close()
    
    def _report_progress(self, callback, current: int, total: int, message: str):
        """Safely call progress callback."""
        if callback:
            try:
                callback(current, total, message)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")

