import sys
from typing import Optional
from loguru import logger

# global flag to ensure stdout logger only setup once
_logging_configured = False

def setup_logging(level: str = "INFO", format_str: Optional[str] = None, colorize: bool = True):
    """Setup custom loguru logging configuration."""
    global _logging_configured
    if _logging_configured:
        return
    
    if format_str is None:
        # format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        format_str = "<green>{time:HH:mm:ss}</green> - {message}"
    
    logger.remove()  # remove default handler and add custom logger
    logger.add(sys.stdout, format=format_str, colorize=colorize, level=level)
    _logging_configured = True