import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        log_dir = ROOT / "reports" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "pipeline.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
    return logger
