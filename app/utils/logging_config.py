# app/utils/logging_config.py
"""
Logging Configuration Module
Sets up consistent, structured logging for the entire application.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_file: str = "logs/rag_system.log",
) -> None:
    """
    Configure application-wide logging.

    Sets up:
    - Console handler (stdout) with color-friendly format
    - File handler for persistent log storage
    - Consistent format across all modules

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_to_file: Whether to write logs to a file.
        log_file: Path to the log file.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create logs directory if logging to file
    if log_to_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Log format: timestamp | level | module | message
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ["httpx", "chromadb", "urllib3", "asyncio"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        f"Logging configured. Level={level}, File={'enabled' if log_to_file else 'disabled'}"
    )