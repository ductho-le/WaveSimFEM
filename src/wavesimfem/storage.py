"""Atomic file replacement and optional disk-backed measurement histories."""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np

HISTORY_CHUNK_BYTES = 8 * 1024**2


def _read_umask() -> int:
    # The only portable query sets and restores the mask, so do it once at
    # import rather than racing other threads that create files.
    mask = os.umask(0)
    os.umask(mask)
    return mask


_UMASK = _read_umask()


def _output_mode(target: Path) -> int:
    """Permission bits for a replacement: keep an existing file's, else honour umask.

    ``mkstemp`` always creates ``0600`` files and ``os.replace`` preserves that,
    which would make every result unreadable to collaborators on shared systems.
    """
    try:
        return stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        return 0o666 & ~_UMASK


@contextlib.contextmanager
def atomic_output(path: str | os.PathLike[str], *, binary: bool = False) -> Iterator[Any]:
    """Write beside the destination, then atomically replace it after flushing.

    Failure before replacement leaves the previous file intact. The temporary
    file is on the same filesystem, so replacement never becomes a copy. The
    result keeps the permissions of the file it replaces, or follows the umask.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    stream: BinaryIO | TextIO | None = None
    try:
        os.chmod(name, _output_mode(target))
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
        if mode == "disk" and directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)
        self._scratch = (
            tempfile.TemporaryDirectory(prefix=".wavesim-history-", dir=directory)
            if mode == "disk"
            else None
        )
        self._maps: list[np.memmap] = []

    def __enter__(self) -> HistoryStore:
        return self

    def close(self) -> None:
        """Close all memory maps and remove the temporary scratch directory."""
        try:
            for array in self._maps:
                # Closing the mapping also releases its file descriptor. No durable
                # flush is needed: these are scratch files and the archive is saved.
                # Contract: callers must not use these arrays (or views of them)
                # after exit; copy anything that has to outlive the store.
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()
        finally:
            self._maps.clear()
            if self._scratch is not None:
                self._scratch.cleanup()
                self._scratch = None

    def __exit__(self, *_exc: Any) -> None:
        self.close()

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
    row_width = int(np.prod(array.shape[1:])) or 1
    rows = max(1, chunk_bytes // (row_width * array.dtype.itemsize))
    return all(
        np.isfinite(array[first : first + rows]).all() for first in range(0, len(array), rows)
    )
