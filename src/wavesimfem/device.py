"""Hardware detection and device selection for the single-process explicit engine.

Device requests are canonical strings:

``auto``
    First available CuPy CUDA device, else CPU.
``cpu``
    Host CPU via SciPy sparse kernels.
``cuda`` / ``cuda:N``
    NVIDIA GPU *N* (default 0) via CuPy.

Explicit accelerator requests that cannot be honored raise
:class:`DeviceUnavailableError` instead of silently running on the CPU.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEVICE_CHOICES_HELP = "auto, cpu, cuda, cuda:N"
_DEVICE_RE = re.compile(r"^(auto|cpu|cuda(:\d+)?)$")


class DeviceUnavailableError(RuntimeError):
    """An explicitly requested compute device is not available."""


@dataclass(frozen=True)
class DeviceInfo:
    """Hardware device descriptor."""

    backend: str  # 'cupy' or 'scipy'
    device: str  # e.g. 'cuda:0', 'cpu'
    name: str  # e.g. 'NVIDIA GeForce RTX 4090', 'Host CPU'
    total_memory_bytes: int | None = None
    compute_capability: tuple[int, int] | None = None

    @property
    def is_gpu(self) -> bool:
        return self.backend == "cupy" and self.device != "cpu"

    @property
    def memory_gb(self) -> float | None:
        if self.total_memory_bytes is None:
            return None
        return self.total_memory_bytes / (1024**3)

    def summary(self) -> str:
        if not self.is_gpu:
            return f"CPU engine ({self.backend})"
        mem_str = f" ({self.memory_gb:.1f} GB)" if self.memory_gb is not None else ""
        cc_str = (
            f" [SM {self.compute_capability[0]}.{self.compute_capability[1]}]"
            if self.compute_capability
            else ""
        )
        return f"{self.name} [{self.backend} → {self.device}]{mem_str}{cc_str}"


CPU_DEVICE = DeviceInfo(backend="scipy", device="cpu", name="Host CPU")


def normalize_device_request(requested: object) -> str:
    """Validate and canonicalize a device request string.

    Raises
    ------
    ValueError
        If the request is not one of ``auto``, ``cpu``, ``cuda``, ``cuda:N``.
    """
    req = str(requested).strip().lower() if requested is not None else "auto"
    if not _DEVICE_RE.match(req):
        raise ValueError(f"Invalid device '{requested}'. Choose from: {DEVICE_CHOICES_HELP}.")
    if req == "cuda":
        return "cuda:0"
    if req.startswith("cuda:"):
        return f"cuda:{int(req.split(':', 1)[1])}"
    return req


def is_cupy_available() -> bool:
    """Return True if CuPy is installed and sees at least one CUDA device."""
    try:
        import cupy as cp

        return bool(cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0)
    except ImportError:
        return False
    except Exception as exc:  # driver/runtime errors
        logger.debug("CuPy present but unusable: %s", exc)
        return False


def detect_available_devices() -> list[DeviceInfo]:
    """Scan the environment and return all usable compute devices (CPU last)."""
    devices: list[DeviceInfo] = []

    if is_cupy_available():
        import cupy as cp

        for i in range(cp.cuda.runtime.getDeviceCount()):
            try:
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props["name"]
                devices.append(
                    DeviceInfo(
                        backend="cupy",
                        device=f"cuda:{i}",
                        name=name.decode("utf-8") if isinstance(name, bytes) else str(name),
                        total_memory_bytes=props.get("totalGlobalMem"),
                        compute_capability=(props.get("major", 0), props.get("minor", 0)),
                    )
                )
            except Exception as exc:
                logger.debug("Failed querying CuPy device %d: %s", i, exc)

    devices.append(CPU_DEVICE)
    return devices


def select_device(
    requested: str = "auto",
    log_func: Callable[[str], None] | None = None,
) -> DeviceInfo:
    """Resolve a device request to a concrete :class:`DeviceInfo`.

    ``auto`` falls back to the CPU when no accelerator is present. Explicit
    accelerator requests (``cuda``, ``cuda:N``) raise
    :class:`DeviceUnavailableError` when they cannot be satisfied.
    """
    req = normalize_device_request(requested)

    def _log(msg: str) -> None:
        if log_func is not None:
            log_func(msg)

    if req == "cpu":
        return CPU_DEVICE

    avail = detect_available_devices()
    if req == "auto":
        gpus = [d for d in avail if d.is_gpu]
        if gpus:
            _log(f"  [Device] Auto-selected accelerator: {gpus[0].summary()}")
            return gpus[0]
        _log("  [Device] No supported accelerator found; using the host CPU engine.")
        return CPU_DEVICE

    matches = [d for d in avail if d.device == req]
    if matches:
        _log(f"  [Device] Selected {matches[0].summary()}")
        return matches[0]

    available = ", ".join(d.device for d in avail)
    hint = (
        ""
        if any(d.is_gpu for d in avail)
        else " Install CuPy (pip install cupy-cuda12x) with a compatible CUDA driver."
    )
    raise DeviceUnavailableError(
        f"Requested device '{req}' is not available (available: {available}).{hint}"
    )
