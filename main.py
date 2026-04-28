#!/usr/bin/env python3
"""
CLI entry point for AI Video Watermark Removal System.

Usage:
    python main.py --input video.mp4 --output cleaned.mp4
    python main.py --batch --input-dir ./videos --output-dir ./cleaned
    python main.py --gui
"""

import argparse
import sys
from pathlib import Path

from app.pipelines.process_video import VideoRemovalPipeline
from app.pipelines.process_batch import BatchProcessor
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Video Watermark Removal System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python main.py -i video.mp4 -o cleaned.mp4

  # Batch process folder
  python main.py --batch -i ./videos -o ./output

  # Launch GUI
  python main.py --gui

  # Use custom API endpoint
  python main.py -i video.mp4 --api-url http://localhost:8080/inpaint
        """
    )
    
    # Input/Output
    parser.add_argument("-i", "--input", dest="input_path",
                        help="Input video file path")
    parser.add_argument("-o", "--output", dest="output_path",
                        help="Output video file path")
    
    # Batch mode
    parser.add_argument("--batch", action="store_true",
                        help="Enable batch processing mode")
    parser.add_argument("--input-dir", dest="input_dir",
                        help="Input directory for batch processing")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Output directory for batch processing")
    
    # API settings
    parser.add_argument("--api-url", default="http://127.0.0.1:8080/api/v1/inpaint",
                        help="IOPaint API endpoint URL")
    parser.add_argument("--model", default="lama",
                        choices=["lama", "ldm", "mat"],
                        help="Inpainting model to use")
    
    # Processing options
    parser.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"),
                        default=None,
                        help="Resize frames before processing (default: no resize)")
    parser.add_argument("--no-resize", action="store_true",
                        help="Process at original resolution (default)")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU acceleration (if available)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for batch mode")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep temporary files after processing")
    
    # GUI
    parser.add_argument("--gui", action="store_true",
                        help="Launch graphical user interface")
    
    # Other
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    return parser


def process_single(args) -> str:
    """Process a single video."""
    logger.info("Starting single video processing")
    
    resize = None if args.no_resize else (tuple(args.resize) if args.resize else None)
    
    pipeline = VideoRemovalPipeline(
        iopaint_url=args.api_url,
        model=args.model,
        resize=resize,
        use_gpu=args.gpu,
        cleanup=not args.no_cleanup
    )
    
    try:
        result = pipeline.process(args.input_path, args.output_path)
        logger.info(f"Processing complete: {result}")
        return result
    finally:
        pipeline.close()


def process_batch(args) -> dict:
    """Process a batch of videos."""
    logger.info("Starting batch processing")
    
    resize = None if args.no_resize else (tuple(args.resize) if args.resize else None)
    
    processor = BatchProcessor(
        iopaint_url=args.api_url,
        model=args.model,
        max_workers=args.workers,
        resize=resize,
        use_gpu=args.gpu,
        cleanup=not args.no_cleanup
    )
    
    summary = processor.process_directory(args.input_dir, args.output_dir)
    
    logger.info(
        f"Batch complete: {summary['successful']}/{summary['total']} successful"
    )
    
    return summary


def launch_gui():
    """Launch the GUI."""
    logger.info("Launching GUI")
    from app.ui.gui import main as gui_main
    gui_main()


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Launch GUI if requested
    if args.gui:
        launch_gui()
        return
    
    # Validate arguments
    if args.batch:
        if not args.input_dir or not args.output_dir:
            parser.error("--batch requires --input-dir and --output-dir")
        if not Path(args.input_dir).exists():
            parser.error(f"Input directory not found: {args.input_dir}")
        
        summary = process_batch(args)
        
        # Print summary
        print("\n" + "=" * 50)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total videos:    {summary['total']}")
        print(f"Successful:      {summary['successful']}")
        print(f"Failed:          {summary['failed']}")
        print("=" * 50)
        
        for result in summary["results"]:
            status_icon = "OK" if result["status"] == "success" else "FAIL"
            print(f"[{status_icon}] {Path(result['input']).name}")
            if result["status"] == "failed":
                print(f"      Error: {result.get('error', 'Unknown')}")
        
        sys.exit(0 if summary["failed"] == 0 else 1)
    
    else:
        if not args.input_path or not args.output_path:
            parser.error("Single mode requires --input and --output (or use --gui)")
        if not Path(args.input_path).exists():
            parser.error(f"Input file not found: {args.input_path}")
        
        result = process_single(args)
        print(f"\nOutput saved to: {result}")


if __name__ == "__main__":
    main()

