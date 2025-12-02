import logging
from datetime import datetime
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(
    service_name: str,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
) -> Logger:
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger(service_name)
    logger.setLevel(numeric_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    logging.getLogger("kubernetes.client.rest").setLevel(logging.WARNING)
    logging.getLogger("kubernetes").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        now = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_dir_time = log_dir + "/" + now
        if not Path(log_dir_time).exists():
            Path(log_dir_time).mkdir(parents=True, exist_ok=True)

        log_file = Path(log_dir_time) / f"{service_name}_{now}.log"

        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger, log_dir_time
