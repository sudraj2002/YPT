import logging
import os
import sys
from datetime import datetime


def setup_logger(output_dir, name="train", rank=0):
    """
    rank=0 -> main process logs
    rank>0 -> silent (avoids duplicated logs)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger  # avoid duplicate handlers

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(
            output_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

import os
import sys
from datetime import datetime


def redirect_prints(output_dir, rank=0):
    """
    Redirect stdout and stderr to a log file.
    Only rank 0 writes to avoid duplicated logs in DDP.
    """
    if rank != 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(
        output_dir,
        f"stdout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    log_file = open(log_path, "w", buffering=1)  # line-buffered
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"[Logging redirected to {log_path}]")

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def redirect_prints_tee(output_dir, rank=0):
    if rank != 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(
        output_dir,
        f"stdout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    log_file = open(log_path, "w", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"[Logging redirected to {log_path}]")