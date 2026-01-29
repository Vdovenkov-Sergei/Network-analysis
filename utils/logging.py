"""Logging utilities.

This module provides utilities for configuring logging.
"""

import logging

from rich.logging import RichHandler


def setup_logger(logger_name: str, log_level: str = "INFO") -> logging.Logger:
    """Configure logging for a specific logger.

    Args:
        logger_name: Name of the logger.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        The configured logger.
    """
    logger = logging.getLogger(logger_name)

    # --- Avoid duplicate handlers ---
    if logger.handlers:
        return logger

    level = getattr(logging, log_level.upper())
    logger.setLevel(level)

    handler = RichHandler(rich_tracebacks=True, show_time=False, show_level=False, show_path=False)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
