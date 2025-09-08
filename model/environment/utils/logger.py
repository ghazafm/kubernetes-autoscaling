import logging
from datetime import datetime
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(
    service_name, log_level="INFO", log_to_file=True, log_dir="logs"
) -> Logger:
    """
    Configure a logger with console and optional file output.

    Args:
        service_name (str): Name of the service (used for the log file name)
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file (bool): Whether to log to a file
        log_dir (str): Directory to store log files

    Returns:
        logging.Logger: Configured logger instance
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        if not Path(log_dir).exists():
            Path(log_dir).mkdir()

        today = datetime.now().strftime("%Y-%m-%d")
        log_file = Path(log_dir) / f"{service_name}_{today}.log"

        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
