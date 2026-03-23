"""
src/utils/logger.py
───────────────────
Set up a logger that writes simultaneously to:
  • the console (stdout)
  • experiments/<name>/logs/train.log

Usage
-----
    from src.utils.logger import get_logger
    log = get_logger("train", log_dir="experiments/exp_001/logs")
    log.info("Epoch 1 started")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str | Path) -> logging.Logger:
    """
    Create (or retrieve) a named logger with a file handler and a
    StreamHandler that writes to stdout.

    Parameters
    ----------
    name : str
        Logger name (also used as the log-file stem).
    log_dir : str | Path
        Directory where ``<name>.log`` will be written.

    Returns
    -------
    logging.Logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the function is called more than once
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # File handler (full DEBUG output)
    file_handler = logging.FileHandler(log_dir / f"{name}.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger
