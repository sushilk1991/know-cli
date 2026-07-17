"""Logging configuration for know."""

import logging
import sys
from pathlib import Path
from typing import Optional


def teardown_logging(logger: Optional[logging.Logger] = None) -> None:
    """Detach and close handlers that may reference short-lived CLI streams."""
    target = logger or logging.getLogger("know")
    for handler in list(target.handlers):
        target.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def setup_logging(
    verbose: bool = False,
    quiet: bool = False,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Configure logging for know-cli.
    
    Args:
        verbose: Enable DEBUG level logging
        quiet: Only show ERROR and above
        log_file: Optional file to write logs to
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("know")
    teardown_logging(logger)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    
    # Console handler
    if quiet:
        console_level = logging.ERROR
    elif verbose:
        console_level = logging.DEBUG
    else:
        console_level = logging.INFO
    
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_format = logging.Formatter(
        "%(levelname)s: %(message)s" if verbose else "%(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (always DEBUG for troubleshooting)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger() -> logging.Logger:
    """Get the know logger instance."""
    return logging.getLogger("know")
