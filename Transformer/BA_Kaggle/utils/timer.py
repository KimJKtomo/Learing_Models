import logging
import time
import timeit


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def load_timer(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        # print(f"[{func.__name__}] load time: {timeit.default_timer()-start}")
        logging.info(f"[{func.__name__}] load time: {timeit.default_timer()-start:.2f}")
        return result
    return wrapper


def inference_timer(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        print(f"[{func.__name__}] inference time: {timeit.default_timer()-start:.2f}")
        return result
    return wrapper
