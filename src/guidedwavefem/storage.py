"""Atomic file replacement and optional disk-backed measurement histories."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np

HISTORY_CHUNK_BYTES = 8 * 1024**2


@contextlib.contextmanager
def atomic_output(path: str | os.PathLike[str], *, binary: bool = False) -> Iterator[Any]:
    """Write beside the destination, then atomically replace it after flushing.

    Failure before replacement leaves the previous file intact. The temporary
    file is on the same filesystem, so replacement never becomes a copy.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    stream: BinaryIO | TextIO | None = None
    try:
        stream = os.fdopen(fd, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"}))
        with stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if stream is None:
            os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name)


class HistoryStore:
    """Own memory maps for one simulation, closing them before scratch cleanup.

    Disk mode bounds measurement allocations, not OS page cache, solver
    workspace, timestamps, or the separate FFT postprocessing stage.
    """

    def __init__(self, mode: str = "memory", directory: str | None = None) -> None:
        if mode not in ("memory", "disk"):
            raise ValueError("history storage must be 'memory' or 'disk'")
        self.mode = mode
        self._scratch = (
            tempfile.TemporaryDirectory(prefix=".gwfem-history-", dir=directory)
            if mode == "disk"
            else None
        )
        self._maps: list[np.memmap] = []

    def __enter__(self) -> HistoryStore:
        return self

    def __exit__(self, *_exc: Any) -> None:
        for array in self._maps:
            # Closing the mapping also releases its file descriptor. No durable
            # flush is needed: these are scratch files and the archive is saved.
            array._mmap.close()
        self._maps.clear()
        if self._scratch is not None:
            self._scratch.cleanup()

    def zeros(self, shape: tuple[int, int], dtype: Any = np.float64) -> np.ndarray:
        if self._scratch is None or 0 in shape:
            return np.zeros(shape, dtype=dtype)
        path = Path(self._scratch.name) / f"history-{len(self._maps)}.bin"
        result = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
        # Newly extended files read as zeros without touching every memory page.
        self._maps.append(result)
        return result


def all_finite_history(array: np.ndarray, chunk_bytes: int = HISTORY_CHUNK_BYTES) -> bool:
    """Check recorded values without allocating a full-history boolean array."""
    if not array.size:
        return True
    rows = max(1, chunk_bytes // (array.shape[1] * array.dtype.itemsize))
    return all(
        np.isfinite(array[first : first + rows]).all() for first in range(0, len(array), rows)
    )
