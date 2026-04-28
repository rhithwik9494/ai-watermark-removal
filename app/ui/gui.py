"""
Tkinter GUI for AI Video Watermark Removal.
Provides user-friendly interface for video upload, processing, and preview.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.pipelines.process_video import VideoRemovalPipeline
from app.pipelines.process_batch import BatchProcessor
from app.utils.logger import get_logger

logger = get_logger(__name__)


class WatermarkRemovalGUI:
    """
    Tkinter-based GUI for the watermark removal system.
    
    Features:
    - Single video upload and processing
    - Batch folder processing
    - Progress bar with status updates
    - Real-time log display
    - Output preview
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Video Watermark Removal")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)
        
        # Styling
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value=str(Path.home() / "Videos" / "output.mp4"))
        self.api_url = tk.StringVar(value="http://127.0.0.1:8080/api/v1/inpaint")
        self.model = tk.StringVar(value="lama")
        self.resize_enabled = tk.BooleanVar(value=False)
        self.resize_w = tk.StringVar(value="640")
        self.resize_h = tk.StringVar(value="360")
        self.is_processing = False
        self.pipeline = None
        
        self._build_ui()
        logger.info("GUI initialized")
    
    def _build_ui(self):
        """Build the user interface."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="AI Video Watermark Removal",
            font=("Helvetica", 18, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20), sticky=tk.W)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Processing Mode", padding="10")
        mode_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(
            mode_frame, text="Single Video", variable=self.mode_var, 
            value="single", command=self._on_mode_change
        ).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(
            mode_frame, text="Batch Folder", variable=self.mode_var, 
            value="batch", command=self._on_mode_change
        ).pack(side=tk.LEFT)
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Label(input_frame, text="Input Path:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Entry(input_frame, textvariable=self.input_path).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(input_frame, text="Browse...", command=self._browse_input).grid(row=0, column=2, padx=(10, 0))
        
        # Output section
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="Output Path:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Entry(output_frame, textvariable=self.output_path).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(output_frame, text="Browse...", command=self._browse_output).grid(row=0, column=2, padx=(10, 0))
        
        # Settings section
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(settings_frame, text="API URL:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Entry(settings_frame, textvariable=self.api_url, width=40).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(settings_frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        model_combo = ttk.Combobox(settings_frame, textvariable=self.model, values=["lama", "ldm", "mat"], width=15)
        model_combo.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # Resize options
        resize_frame = ttk.Frame(settings_frame)
        resize_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        ttk.Checkbutton(resize_frame, text="Resize frames to", variable=self.resize_enabled).pack(side=tk.LEFT)
        ttk.Entry(resize_frame, textvariable=self.resize_w, width=6).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(resize_frame, text="x").pack(side=tk.LEFT, padx=(2, 2))
        ttk.Entry(resize_frame, textvariable=self.resize_h, width=6).pack(side=tk.LEFT)
        ttk.Label(resize_frame, text="(faster processing)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, 
            maximum=100, mode="determinate", length=400
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), columnspan=2)
        
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        self.percent_label = ttk.Label(progress_frame, text="0%")
        self.percent_label.grid(row=1, column=1, sticky=tk.E, pady=(5, 0))
        
        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Status Log", padding="10")
        log_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=10, wrap=tk.WORD, 
            font=("Consolas", 9)
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=(10, 0))
        
        self.start_button = ttk.Button(
            button_frame, text="Start Processing", 
            command=self._start_processing, width=20
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame, text="Clear Log", 
            command=self._clear_log, width=15
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame, text="Exit", 
            command=self.root.quit, width=10
        ).pack(side=tk.LEFT)
        
        # Configure weights for resizing
        main_frame.rowconfigure(6, weight=1)
    
    def _on_mode_change(self):
        """Handle mode change between single and batch."""
        mode = self.mode_var.get()
        if mode == "batch":
            self.output_path.set(str(Path.home() / "Videos" / "batch_output"))
        else:
            self.output_path.set(str(Path.home() / "Videos" / "output.mp4"))
    
    def _browse_input(self):
        """Open file or folder dialog for input."""
        mode = self.mode_var.get()
        if mode == "single":
            path = filedialog.askopenfilename(
                title="Select Video",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                    ("All files", "*.*")
                ]
            )
        else:
            path = filedialog.askdirectory(title="Select Input Folder")
        
        if path:
            self.input_path.set(path)
            # Auto-set output if not set
            if not self.output_path.get() or self.output_path.get().endswith("output.mp4"):
                p = Path(path)
                if mode == "single":
                    self.output_path.set(str(p.parent / f"{p.stem}_cleaned{p.suffix}"))
                else:
                    self.output_path.set(str(p.parent / "batch_output"))
    
    def _browse_output(self):
        """Open file or folder dialog for output."""
        mode = self.mode_var.get()
        if mode == "single":
            path = filedialog.asksaveasfilename(
                title="Save Output",
                defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")]
            )
        else:
            path = filedialog.askdirectory(title="Select Output Folder")
        
        if path:
            self.output_path.set(path)
    
    def _log(self, message: str):
        """Add message to log display."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        logger.info(message)
    
    def _clear_log(self):
        """Clear log display."""
        self.log_text.delete(1.0, tk.END)
    
    def _update_progress(self, current: int, total: int, message: str):
        """Update progress bar from callback."""
        def _update():
            percent = min(100, int((current / total) * 100)) if total > 0 else 0
            self.progress_var.set(percent)
            self.percent_label.config(text=f"{percent}%")
            self.status_label.config(text=message)
            self._log(message)
        
        # Schedule on main thread
        self.root.after(0, _update)
    
    def _start_processing(self):
        """Start processing in background thread."""
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing is already running!")
            return
        
        input_path = self.input_path.get()
        output_path = self.output_path.get()
        
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid input path!")
            return
        
        if not output_path:
            messagebox.showerror("Error", "Please specify an output path!")
            return
        
        # Get resize settings
        resize = None
        if self.resize_enabled.get():
            try:
                w = int(self.resize_w.get())
                h = int(self.resize_h.get())
                resize = (w, h)
            except ValueError:
                messagebox.showerror("Error", "Invalid resize dimensions!")
                return
        
        # Start processing thread
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self._log("=" * 50)
        self._log("Starting processing...")
        self._log(f"Input: {input_path}")
        self._log(f"Output: {output_path}")
        self._log(f"Model: {self.model.get()}")
        self._log(f"Resize: {resize}")
        self._log("=" * 50)
        
        thread = threading.Thread(
            target=self._process_thread,
            args=(input_path, output_path, resize),
            daemon=True
        )
        thread.start()
    
    def _process_thread(self, input_path: str, output_path: str, resize):
        """Background processing thread."""
        try:
            mode = self.mode_var.get()
            
            if mode == "single":
                self._process_single(input_path, output_path, resize)
            else:
                self._process_batch(input_path, output_path, resize)
                
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            self.root.after(0, lambda: self._log(f"ERROR: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed:\n{str(e)}"))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
    
    def _process_single(self, input_path: str, output_path: str, resize):
        """Process a single video."""
        pipeline = VideoRemovalPipeline(
            iopaint_url=self.api_url.get(),
            model=self.model.get(),
            resize=resize,
            cleanup=True
        )
        
        try:
            result = pipeline.process(
                input_path, 
                output_path,
                progress_callback=self._update_progress
            )
            self.root.after(0, lambda: self._log(f"Output saved: {result}"))
            self.root.after(0, lambda: messagebox.showinfo(
                "Success", f"Video processed successfully!\nSaved to:\n{result}"
            ))
        finally:
            pipeline.close()
    
    def _process_batch(self, input_dir: str, output_dir: str, resize):
        """Process a batch of videos."""
        processor = BatchProcessor(
            iopaint_url=self.api_url.get(),
            model=self.model.get(),
            resize=resize,
            max_workers=1  # Sequential for stability in GUI
        )
        
        summary = processor.process_directory(
            input_dir, 
            output_dir,
            progress_callback=self._update_progress
        )
        
        msg = (
            f"Batch processing complete!\n"
            f"Total: {summary['total']}\n"
            f"Successful: {summary['successful']}\n"
            f"Failed: {summary['failed']}"
        )
        
        self.root.after(0, lambda: self._log(msg.replace('\n', ' | ')))
        self.root.after(0, lambda: messagebox.showinfo("Batch Complete", msg))
    
    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()


def main():
    """Entry point for GUI."""
    gui = WatermarkRemovalGUI()
    gui.run()


if __name__ == "__main__":
    main()

