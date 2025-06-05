"""
Author: Jesús Maldonado
Description: This module provides a utility function to set up a logger with both file and console handlers.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

logger: logging.Logger | None = None

levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}
level_name = {v: k for k, v in levels.items()}

def setup_logger(level: int|str = logging.DEBUG) -> logging.Logger:
    """
    Setup a logger with a file handler and a console handler.

    Args:
        level (int|str): Minimum logging level. Can be an integer or a string (DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50).

    Returns:
        logging.Logger: Configured logger instance.
    """
    level = level if isinstance(level, int) else levels[level.upper()]

    global logger
    if logger is not None:
        logger.debug(f"Logger already created with name '{logger.name}' and level '{level_name[logger.level]}'.")
        if logger.level != level:
            logger.warning(f"Logger level mismatch: Required '{level_name[level]}' but found '{level_name[logger.level]}'. Setting to '{level_name[level]}'.")
            logger.setLevel(level)
        return logger
    
    logger = logging.getLogger("prediccion de decisiones")

    try:
        logger.setLevel(level)
    except (ValueError, TypeError):
        logger.setLevel(logging.DEBUG)
        logger.warning(f"Invalid logging level '{level}' provided. Defaulting to DEBUG level.")
    
    # Logs base path
    project_root_path = Path(__file__).resolve().parent.parent
    logs_path = project_root_path / "logs"

    # Create logs directory if it doesn't exist
    os.makedirs(logs_path, exist_ok=True)

    # Generate log file name based on the current date
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(logs_path, f"{date_str}.log")

    # Logs formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(module)s/%(funcName)s/%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
