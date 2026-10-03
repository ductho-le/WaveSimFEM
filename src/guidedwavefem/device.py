"""Hardware detection and device selection abstraction for GuidedWaveFEM.

Provides unified device querying across NVIDIA CUDA (via CuPy or PyTorch),
Apple Silicon Metal (via PyTorch MPS), and host CPU (via SciPy / PETSc).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Hardware device descriptor."""

    backend: str  # 'cupy', 'torch', or 'scipy'
    device: str  # e.g. 'cuda:0', 'mps', 'cpu'
    name: str  # e.g. 'NVIDIA GeForce RTX 4090', 'Apple M3 Max', 'Host CPU'
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
            return f"CPU Engine ({self.backend})"
        mem_str = (
            f" ({self.memory_gb:.1f} GB VRAM)"
            if self.memory_gb is not None
            else ""
        )
        cc_str = (
            f" [SM {self.compute_capability[0]}.{self.compute_capability[1]}]"
            if self.compute_capability
            else ""
        )
        return f"{self.name} [{self.backend} -> {self.device}]{mem_str}{cc_str}"


def is_cupy_available() -> bool:
    """Return True if CuPy is installed and has at least one CUDA device."""
    try:
        import cupy as cp

        return bool(cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0)
    except (ImportError, Exception):
        return False


def is_torch_cuda_available() -> bool:
    """Return True if PyTorch is installed with CUDA support."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except (ImportError, Exception):
        return False


def is_torch_mps_available() -> bool:
    """Return True if PyTorch is installed with functional MPS sparse SpMV support.

    Note: As of PyTorch 2.x, PyTorch's Metal (MPS) backend does not implement
    compressed sparse tensors (sparse CSR SpMV), raising NotImplementedError.
    We verify sparse SpMV capability explicitly to ensure reliable hardware dispatch.
    """
    try:
        import torch

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return False
        # Verify sparse SpMV operation on MPS
        dev = torch.device("mps")
        crow = torch.tensor([0, 1], dtype=torch.int64, device=dev)
        col = torch.tensor([0], dtype=torch.int64, device=dev)
        val = torch.tensor([1.0], dtype=torch.float32, device=dev)
        a_sp = torch.sparse_csr_tensor(crow, col, val, size=(1, 1), device=dev)
        x_vec = torch.tensor([1.0], dtype=torch.float32, device=dev)
        _ = torch.mv(a_sp, x_vec)
        return True
    except (ImportError, Exception):
        return False


def detect_available_devices() -> list[DeviceInfo]:
    """Scan the environment and return all available compute devices."""
    devices: list[DeviceInfo] = []

    # 1. CuPy CUDA devices
    if is_cupy_available():
        import cupy as cp

        for i in range(cp.cuda.runtime.getDeviceCount()):
            try:
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props["name"].decode("utf-8") if isinstance(props["name"], bytes) else str(props["name"])
                total_mem = props.get("totalGlobalMem")
                cc = (props.get("major", 0), props.get("minor", 0))
                devices.append(
                    DeviceInfo(
                        backend="cupy",
                        device=f"cuda:{i}",
                        name=name,
                        total_memory_bytes=total_mem,
                        compute_capability=cc,
                    )
                )
            except Exception as e:
                logger.debug(f"Failed querying CuPy device {i}: {e}")

    # 2. PyTorch CUDA devices (if not already found via CuPy)
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
            except Exception as e:
                logger.debug(f"Failed querying Torch device {i}: {e}")

    # 3. Apple Silicon MPS via PyTorch
    if is_torch_mps_available():
        devices.append(
            DeviceInfo(
                backend="torch",
                device="mps",
                name="Apple Silicon GPU (Metal)",
            )
        )

    # 4. CPU Fallback
    devices.append(
        DeviceInfo(
            backend="scipy",
            device="cpu",
            name="Host CPU",
        )
    )

    return devices


def select_device(
    requested: str = "auto",
    log_func: Callable[[str], None] | None = None,
) -> DeviceInfo:
    """Select the target compute device based on user request and availability.

    Parameters
    ----------
    requested : str
        One of 'auto', 'cuda', 'cuda:X', 'mps', 'cpu', 'torch:cuda', 'torch:mps'.
    log_func : Callable[[str], None] | None
        Optional logging function (e.g. print).

    Returns
    -------
    DeviceInfo
        Selected device descriptor.
    """
    req = requested.strip().lower()
    avail = detect_available_devices()
    gpus = [d for d in avail if d.is_gpu]

    def _log(msg: str) -> None:
        if log_func is not None:
            log_func(msg)

    # Automatic detection: prioritize CuPy CUDA -> Torch CUDA -> Torch MPS -> CPU
    if req in ("auto", "gpu"):
        if gpus:
            selected = gpus[0]
            _log(f"  [Device] Auto-selected GPU accelerator: {selected.summary()}")
            return selected
        _log("  [Device] No GPU accelerator found. Defaulting to Host CPU.")
        return DeviceInfo(backend="scipy", device="cpu", name="Host CPU")

    # Explicit CPU request
    if req in ("cpu", "host"):
        return DeviceInfo(backend="scipy", device="cpu", name="Host CPU")

    # Explicit Apple Silicon MPS request
    if req in ("mps", "apple", "metal", "torch:mps"):
        mps_devs = [d for d in avail if d.device == "mps"]
        if mps_devs:
            selected = mps_devs[0]
            _log(f"  [Device] Selected Apple Silicon GPU: {selected.summary()}")
            return selected
        _log(
            "  [Device INFO] Apple Silicon MPS requested, but PyTorch currently lacks sparse CSR SpMV on Metal. Defaulting to optimized CPU engine."
        )
        return DeviceInfo(backend="scipy", device="cpu", name="Host CPU")

    # Explicit CUDA request (e.g. 'cuda', 'cuda:0', 'cuda:1')
    if req.startswith("cuda"):
        target_id = 0
        if ":" in req:
            try:
                target_id = int(req.split(":", 1)[1])
            except ValueError:
                target_id = 0

        # Check CuPy first
        cupy_devs = [d for d in avail if d.backend == "cupy" and d.device == f"cuda:{target_id}"]
        if cupy_devs:
            selected = cupy_devs[0]
            _log(f"  [Device] Selected CUDA device via CuPy: {selected.summary()}")
            return selected

        # Check PyTorch CUDA
        torch_devs = [d for d in avail if d.backend == "torch" and d.device == f"cuda:{target_id}"]
        if torch_devs:
            selected = torch_devs[0]
            _log(f"  [Device] Selected CUDA device via PyTorch: {selected.summary()}")
            return selected

        _log(f"  [Device WARNING] CUDA device '{req}' not available. Falling back to CPU.")
        return DeviceInfo(backend="scipy", device="cpu", name="Host CPU")

    # Unknown request
    _log(f"  [Device WARNING] Unrecognized device '{requested}'. Defaulting to CPU.")
    return DeviceInfo(backend="scipy", device="cpu", name="Host CPU")
