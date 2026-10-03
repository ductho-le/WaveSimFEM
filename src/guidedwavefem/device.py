"""Hardware detection and device selection for the single-process explicit engine.

Device requests are canonical strings:

``auto``
    Best available accelerator (CuPy CUDA → PyTorch CUDA → PyTorch MPS), else CPU.
``cpu``
    Host CPU via SciPy sparse kernels.
``cuda`` / ``cuda:N``
    NVIDIA GPU *N* (default 0) via CuPy, or PyTorch if CuPy is not installed.
``mps``
    Apple-Silicon GPU via PyTorch, only if PyTorch supports sparse CSR SpMV on
    Metal (not the case as of PyTorch 2.x).

Explicit accelerator requests that cannot be honored raise
:class:`DeviceUnavailableError` instead of silently running on the CPU.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEVICE_CHOICES_HELP = "auto, cpu, cuda, cuda:N, mps"
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")


class DeviceUnavailableError(RuntimeError):
    """An explicitly requested compute device is not available."""


@dataclass(frozen=True)
class DeviceInfo:
    """Hardware device descriptor."""

    backend: str  # 'cupy', 'torch', or 'scipy'
    device: str  # e.g. 'cuda:0', 'mps', 'cpu'
    name: str  # e.g. 'NVIDIA GeForce RTX 4090', 'Host CPU'
    total_memory_bytes: int | None = None
    compute_capability: tuple[int, int] | None = None

    @property
    def is_gpu(self) -> bool:
        return self.backend in ("cupy", "torch") and self.device != "cpu"

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
        If the request is not one of ``auto``, ``cpu``, ``cuda``, ``cuda:N``, ``mps``.
    """
    req = str(requested).strip().lower() if requested is not None else "auto"
    if not _DEVICE_RE.match(req):
        raise ValueError(f"Invalid device '{requested}'. Choose from: {DEVICE_CHOICES_HELP}.")
    return "cuda:0" if req == "cuda" else req


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


def is_torch_cuda_available() -> bool:
    """Return True if PyTorch is installed with a usable CUDA device."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except ImportError:
        return False
    except Exception as exc:
        logger.debug("PyTorch CUDA unusable: %s", exc)
        return False


def is_torch_mps_available() -> bool:
    """Return True if PyTorch MPS exists *and* supports sparse CSR SpMV.

    As of PyTorch 2.x the Metal backend does not implement sparse CSR
    matrix-vector products, so this probe normally returns False.
    """
    try:
        import torch

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return False
        dev = torch.device("mps")
        a_sp = torch.sparse_csr_tensor(
            torch.tensor([0, 1], dtype=torch.int64, device=dev),
            torch.tensor([0], dtype=torch.int64, device=dev),
            torch.tensor([1.0], dtype=torch.float32, device=dev),
            size=(1, 1),
            device=dev,
        )
        torch.mv(a_sp, torch.ones(1, dtype=torch.float32, device=dev))
        return True
    except ImportError:
        return False
    except Exception as exc:
        logger.debug("PyTorch MPS sparse SpMV unsupported: %s", exc)
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

    if not devices and is_torch_cuda_available():
        import torch

        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    DeviceInfo(
                        backend="torch",
                        device=f"cuda:{i}",
                        name=props.name,
                        total_memory_bytes=props.total_memory,
                        compute_capability=(props.major, props.minor),
                    )
                )
            except Exception as exc:
                logger.debug("Failed querying PyTorch device %d: %s", i, exc)

    if is_torch_mps_available():
        devices.append(DeviceInfo(backend="torch", device="mps", name="Apple Silicon GPU (Metal)"))

    devices.append(CPU_DEVICE)
    return devices


def select_device(
    requested: str = "auto",
    log_func: Callable[[str], None] | None = None,
) -> DeviceInfo:
    """Resolve a device request to a concrete :class:`DeviceInfo`.

    ``auto`` falls back to the CPU when no accelerator is present. Explicit
    accelerator requests (``cuda``, ``cuda:N``, ``mps``) raise
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
        " PyTorch does not support sparse CSR SpMV on Apple Metal (MPS); use 'auto' or 'cpu'."
        if req == "mps"
        else " Install CuPy (pip install cupy-cuda12x) or a CUDA build of PyTorch."
    )
    raise DeviceUnavailableError(
        f"Requested device '{req}' is not available (available: {available}).{hint}"
    )
