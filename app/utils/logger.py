import logging
import sys
import os
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{log_color}{record.levelname}{reset}"
        return super().format(record)


def get_logger(name: str = "ai_watermark_removal", log_dir: str = "logs") -> logging.Logger:
    """
    Get a configured logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # File handler - detailed logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Console handler - colored, less verbose
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger


# Global logger instance
_default_logger = None

def default_logger() -> logging.Logger:
    """Get or create the default logger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()
    return _default_logger

