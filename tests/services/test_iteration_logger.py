import time

from nexum.services.iteration_logger import IterationLogger


def test():
    logger = IterationLogger(
        modules={"test": lambda iteration, i: f"[{iteration}; {i}]"}
    )
    for i in logger(range(100)):
        time.sleep(0.1)
        logger.ds.i = i
