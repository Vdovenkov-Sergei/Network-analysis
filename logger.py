"""
Logging utilities.
"""

import logging
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger.

    Args:
        name: Logger name. If None (default), the root logger is returned.

    Returns:
        Ready-to-use logger instance.
    """
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    return logging.getLogger(name)
