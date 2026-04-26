import logging
import sys
from pathlib import Path

# Ensure logs directory exists
LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name: str = "modelforge") -> logging.Logger:
    """Configures and returns a centralized logger."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        file_handler = logging.FileHandler(LOG_DIR / "app.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

logger = get_logger()
