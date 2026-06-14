import logging
import os
import sys


def _build_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)

    if log.handlers:
        return log

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    log.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    log.addHandler(handler)
    log.propagate = False
    return log

logger = _build_logger("genai_agent")