import logging
import os
import sys


def setup_logging(log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger → console + rotating file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Remove any handlers already attached (safe for re-runs in notebooks)
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Silence noisy third-party loggers
    for noisy in ("matplotlib", "PIL", "joblib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger()
