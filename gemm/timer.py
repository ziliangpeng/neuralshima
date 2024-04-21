import time


class Timed:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.elapsed = self.end - self.start
        print(f"{self.name} elapsed time: {self.elapsed} seconds")
