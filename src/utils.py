"""Shared helpers: progress logging and stage timing."""

import time
from contextlib import contextmanager

_T0 = time.perf_counter()


def log(message: str) -> None:
    """Print a progress message prefixed with the time since the run started."""
    elapsed = time.perf_counter() - _T0
    print(f"[{elapsed / 60:5.1f} min] {message}", flush=True)


@contextmanager
def stage(name: str):
    """Log the start and finish of a pipeline stage with its elapsed time.

    Usage:
        with stage("Stage 1: ingest"):
            ...
    """
    log(f"{name} started")
    start = time.perf_counter()
    yield
    log(f"{name} completed in {time.perf_counter() - start:.1f} s")


def ensure_dirs(*paths) -> None:
    """Create each directory (and its parents) if it does not exist yet."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
