import logging
import sys

def _build_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)         

    if log.handlers:                     
        return log                         

    log.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)   
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    log.addHandler(handler)
    log.propagate = False           

    return log

logger = _build_logger("genai_agent")