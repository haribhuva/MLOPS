"""Rotating-file + console logging configured at import time."""
import logging
import os
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

_LOG_DIR = "logs"
_LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
_MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3

_log_dir_path = os.path.join(from_root(), _LOG_DIR)
os.makedirs(_log_dir_path, exist_ok=True)
_log_file_path = os.path.join(_log_dir_path, _LOG_FILE)


def configure_logger() -> logging.Logger:
    """Set up root logger with rotating file and console handlers."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        _log_file_path, maxBytes=_MAX_LOG_SIZE, backupCount=_BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Silence noisy third-party loggers
    for lib in [
        "tensorflow", "h5py", "pydot", "matplotlib", "PIL",
        "absl", "urllib3", "google", "mlflow", "werkzeug", "git",
    ]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    return logger


configure_logger()