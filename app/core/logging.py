
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging() -> None:
    """
    Configure application-wide logging.

    Supports configurable log levels, console output, and rotating log files.
    """

    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_format = (
        "%(asctime)s | %(levelname)-8s | "
        "%(name)s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    formatter = logging.Formatter(
        fmt=log_format,
        datefmt=date_format,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if configuration runs more than once.
    if root_logger.handlers:
        for handler in root_logger.handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
        return

    # Console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional rotating file logging
    if os.getenv("LOG_TO_FILE", "false").lower() == "true":
        log_dir = Path(os.getenv("LOG_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Logging configured successfully at %s level",
        logging.getLevelName(log_level),
    )
