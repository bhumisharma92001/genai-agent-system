import concurrent.futures
from utils.logger import logger


def run_parallel(tasks: list[dict]) -> dict:
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = {ex.submit(t["fn"], **t["args"]): t["name"] for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            name = futures[f]
            try:
                results[name] = f.result()
            except Exception as e:
                logger.error(f"Parallel task '{name}' failed: {e}")
                results[name] = None
    return results