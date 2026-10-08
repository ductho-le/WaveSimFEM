"""Single-process explicit RK4 engine for CPU (SciPy), CUDA (CuPy/PyTorch) and MPS.

Design
------
* **One integrator, many backends.** The RK4 recurrence is written once
  (:func:`integrate_explicit_rk4`) against a minimal backend protocol
  (sparse mat-vec, fused-free axpy updates, host transfer). Backends differ
  only in their array library.
* **Stacked operator.** The state ``z = [u; v]`` is advanced with a single
  sparse product ``[K | C] @ z`` per stage instead of two.
* **Preallocated stage vectors.** Updates use ``out=``/in-place operations;
  sparse products may allocate temporary vectors in the array library.
* **Device-resident telemetry.** Sensor samples (decimated to the configured
  sample rate) are written into a bounded on-device ring buffer and flushed to
  host memory in large chunks, so there is no host synchronization per time step.
* **Lazy imports.** CuPy/PyTorch are only imported when their backend is used.

The distributed (MPI) explicit solver uses :class:`guidedwavefem.numerics.PetscRK4`
with PETSc operators; both implement the scheme of
:func:`guidedwavefem.numerics.rk4_step`.
"""

from __future__ import annotations

import contextlib
import time as pytime
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    import dolfinx

    from guidedwavefem.device import DeviceInfo

CsrLike = sp.spmatrix | sp.sparray | tuple[np.ndarray, np.ndarray, np.ndarray]
SnapshotCallback = Callable[[int, float, np.ndarray], None]

DEFAULT_HISTORY_CHUNK_BYTES = 64 * 1024**2


# ---------------------------------------------------------------------------
# Sensor interpolation operator
# ---------------------------------------------------------------------------
def build_sensor_interpolation_matrix(
    V: dolfinx.fem.FunctionSpace,
    points: np.ndarray,
    active_cells: list[int] | np.ndarray,
    n_owned_dofs: int,
) -> sp.csr_matrix:
    """Exact sparse point-evaluation operator ``S`` with ``S @ u = u(points)``.

    Row ``2i`` returns the x-displacement and row ``2i + 1`` the y-displacement
    of sensor ``i``. Basis weights are obtained from DOLFINx point evaluation.
    Cells are greedily coloured so that cells sharing a DOF are never probed in
    the same batch; the number of ``eval`` calls is therefore
    ``n_colours × n_cell_dofs × block_size`` rather than per sensor and DOF.
    """
    import dolfinx

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    cells = np.asarray(active_cells, dtype=np.int32)
    n_pts = len(points)
    if n_pts == 0:
        return sp.csr_matrix((0, n_owned_dofs), dtype=np.float64)
    if np.any(cells < 0):
        raise ValueError("All sensor points must have a valid local cell")

    bs = V.dofmap.index_map_bs
    unique_cells = np.unique(cells)
    cell_nodes = {int(c): np.asarray(V.dofmap.cell_dofs(int(c))) for c in unique_cells}

    # Greedy colouring: cells within a colour share no DOF nodes.
    colour_nodes: list[set[int]] = []
    colour_of: dict[int, int] = {}
    for c in unique_cells:
        nodes = set(cell_nodes[int(c)].tolist())
        for k, used in enumerate(colour_nodes):
            if used.isdisjoint(nodes):
                used.update(nodes)
                colour_of[int(c)] = k
                break
        else:
            colour_nodes.append(nodes)
            colour_of[int(c)] = len(colour_nodes) - 1

    probe = dolfinx.fem.Function(V)
    n_cell_nodes = len(next(iter(cell_nodes.values())))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    point_colour = np.array([colour_of[int(c)] for c in cells])

    for k in range(len(colour_nodes)):
        sel = np.flatnonzero(point_colour == k)
        sel_cells = cells[sel]
        node_table = np.stack([cell_nodes[int(c)] for c in sel_cells])  # (m, n_cell_nodes)
        group_cells = np.unique(sel_cells)
        group_nodes = np.stack([cell_nodes[int(c)] for c in group_cells])
        for j in range(n_cell_nodes):
            for comp in range(bs):
                probe.x.array[:] = 0.0
                probe.x.array[group_nodes[:, j] * bs + comp] = 1.0
                values = probe.eval(points[sel], sel_cells).reshape(len(sel), -1)
                dofs = node_table[:, j] * bs + comp
                owned = dofs < n_owned_dofs
                for out_comp in range(2):
                    w = values[:, out_comp]
                    keep = owned & (np.abs(w) > 1e-14)
                    rows.append(2 * sel[keep] + out_comp)
                    cols.append(dofs[keep])
                    data.append(w[keep])

    return sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(2 * n_pts, n_owned_dofs),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class _NumpyLikeBackend:
    """Backend for array libraries following the NumPy ufunc API (NumPy, CuPy)."""

    name = "scipy"

    def __init__(self, dtype: np.dtype) -> None:
        self.dtype = np.dtype(dtype)
        self.xp: Any = np

    # -- lifecycle -----------------------------------------------------------
    def context(self) -> contextlib.AbstractContextManager[Any]:
        # Divergence is reported by the periodic finite check, not per-op warnings.
        return np.errstate(over="ignore", invalid="ignore")

    def synchronize(self) -> None:
        return None

    # -- data movement -------------------------------------------------------
    def from_host(self, array: np.ndarray) -> Any:
        return self.xp.asarray(np.ascontiguousarray(array, dtype=self.dtype))

    def csr(self, matrix: sp.csr_matrix) -> Any:
        return sp.csr_matrix(matrix, dtype=self.dtype)

    def zeros(self, shape: int | tuple[int, ...]) -> Any:
        return self.xp.zeros(shape, dtype=self.dtype)

    def to_host(self, array: Any) -> np.ndarray:
        return np.array(array, copy=True)

    # -- arithmetic ----------------------------------------------------------
    def spmv(self, matrix: Any, vector: Any) -> Any:
        return matrix @ vector

    def axpy(self, out: Any, x: Any, a: float, y: Any) -> None:
        """``out = x + a*y`` (``out`` must not alias ``x`` or ``y``)."""
        self.xp.multiply(y, a, out=out)
        self.xp.add(out, x, out=out)

    def add_scaled(self, acc: Any, a: float, y: Any, scratch: Any) -> None:
        """``acc += a*y`` using ``scratch`` (same shape) as workspace."""
        self.xp.multiply(y, a, out=scratch)
        self.xp.add(acc, scratch, out=acc)

    def force_balance(self, out: Any, f: float, load: Any, r: Any, inv_mass: Any) -> None:
        """``out = (f*load - r) * inv_mass``."""
        self.xp.multiply(load, f, out=out)
        self.xp.subtract(out, r, out=out)
        self.xp.multiply(out, inv_mass, out=out)

    def all_finite(self, array: Any) -> bool:
        return bool(self.xp.isfinite(array).all())


class _CupyBackend(_NumpyLikeBackend):
    name = "cupy"

    def __init__(self, dtype: np.dtype, device_id: int = 0) -> None:
        super().__init__(dtype)
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        self.xp = cp
        self._cpsp = cpsp
        self.device_id = device_id

    def context(self) -> contextlib.AbstractContextManager[Any]:
        return self.xp.cuda.Device(self.device_id)

    def synchronize(self) -> None:
        self.xp.cuda.get_current_stream().synchronize()

    def csr(self, matrix: sp.csr_matrix) -> Any:
        return self._cpsp.csr_matrix(sp.csr_matrix(matrix, dtype=self.dtype))

    def to_host(self, array: Any) -> np.ndarray:
        return self.xp.asnumpy(array)


class _TorchBackend:
    name = "torch"

    def __init__(self, dtype: np.dtype, device: str = "cpu") -> None:
        import torch

        self.torch = torch
        self.dtype = np.dtype(dtype)
        self.tdtype = torch.float32 if self.dtype == np.float32 else torch.float64
        self.device = torch.device(device)

    def context(self) -> contextlib.AbstractContextManager[Any]:
        return contextlib.nullcontext()

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            self.torch.mps.synchronize()

    def from_host(self, array: np.ndarray) -> Any:
        return self.torch.as_tensor(
            np.ascontiguousarray(array, dtype=self.dtype), device=self.device
        )

    def csr(self, matrix: sp.csr_matrix) -> Any:
        m = sp.csr_matrix(matrix, dtype=self.dtype)
        return self.torch.sparse_csr_tensor(
            self.torch.as_tensor(m.indptr.astype(np.int64)),
            self.torch.as_tensor(m.indices.astype(np.int64)),
            self.torch.as_tensor(m.data),
            size=m.shape,
            device=self.device,
        )

    def zeros(self, shape: int | tuple[int, ...]) -> Any:
        return self.torch.zeros(shape, dtype=self.tdtype, device=self.device)

    def to_host(self, array: Any) -> np.ndarray:
        return array.detach().cpu().numpy().copy()

    def spmv(self, matrix: Any, vector: Any) -> Any:
        return self.torch.mv(matrix, vector)

    def axpy(self, out: Any, x: Any, a: float, y: Any) -> None:
        self.torch.add(x, y, alpha=a, out=out)

    def add_scaled(self, acc: Any, a: float, y: Any, scratch: Any) -> None:
        acc.add_(y, alpha=a)

    def force_balance(self, out: Any, f: float, load: Any, r: Any, inv_mass: Any) -> None:
        self.torch.mul(load, f, out=out)
        out.sub_(r)
        out.mul_(inv_mass)

    def all_finite(self, array: Any) -> bool:
        return bool(self.torch.isfinite(array).all())


def resolve_precision(precision: str) -> np.dtype:
    """Map ``'double'``/``'single'`` to a NumPy dtype."""
    p = str(precision).lower()
    if p not in ("double", "single"):
        raise ValueError(f"precision must be 'double' or 'single', got '{precision}'")
    return np.dtype(np.float32 if p == "single" else np.float64)


def make_backend(
    device_info: DeviceInfo,
    precision: str = "double",
    log_func: Callable[[str], None] | None = None,
) -> _NumpyLikeBackend | _TorchBackend:
    """Instantiate the array backend for a resolved device."""
    dtype = resolve_precision(precision)
    if device_info.backend == "cupy":
        dev_id = int(device_info.device.split(":")[1]) if ":" in device_info.device else 0
        return _CupyBackend(dtype, dev_id)
    if device_info.backend == "torch":
        if device_info.device == "mps" and dtype == np.float64:
            if log_func:
                log_func("  [Device] MPS does not support float64; using single precision.")
            dtype = np.dtype(np.float32)
        return _TorchBackend(dtype, device_info.device)
    if device_info.backend == "scipy":
        return _NumpyLikeBackend(dtype)
    raise ValueError(f"Unknown backend '{device_info.backend}'")


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------
def _as_csr(matrix: CsrLike, n: int) -> sp.csr_matrix:
    if isinstance(matrix, tuple):
        indptr, indices, data = matrix
        for name, values in (("indptr", indptr), ("indices", indices)):
            values = np.asarray(values)
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise ValueError(f"CSR {name} must be a one-dimensional integer array")
        matrix = sp.csr_matrix((data, indices, indptr), shape=(n, n))
    else:
        matrix = sp.csr_matrix(matrix, copy=True)
    matrix.check_format(full_check=True)
    if np.iscomplexobj(matrix.data) or not np.isfinite(matrix.data).all():
        raise ValueError("CSR coefficients must be real and finite")
    return matrix


def _integer_parameter(value: int, name: str, minimum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class ExplicitRK4Result:
    """Output of :func:`integrate_explicit_rk4`."""

    times: np.ndarray
    sensor_data_x: np.ndarray
    sensor_data_y: np.ndarray
    loop_seconds: float


def integrate_explicit_rk4(
    stiffness: CsrLike,
    damping: CsrLike,
    inv_mass: np.ndarray,
    load_shape: np.ndarray,
    sensor_matrix: sp.spmatrix | sp.sparray,
    dt: float,
    n_steps: int,
    source: Callable[[float], float],
    backend: _NumpyLikeBackend | _TorchBackend,
    save_every: int = 1,
    snapshot_callback: SnapshotCallback | None = None,
    report_interval: int = 0,
    log_func: Callable[[str], None] | None = None,
    history_chunk_bytes: int = DEFAULT_HISTORY_CHUNK_BYTES,
    sample_every: int = 1,
    history_callback: Callable[[np.ndarray, np.ndarray], None] | None = None,
    retain_history: bool = True,
) -> ExplicitRK4Result:
    """Integrate ``M_L ü + C u̇ + K u = s(t) f`` from rest with classical RK4.

    Parameters
    ----------
    stiffness, damping
        Square CSR operators (SciPy matrices or ``(indptr, indices, data)``).
    inv_mass
        Inverse lumped (diagonal) mass, shape ``(n,)``.
    load_shape
        Spatial load vector ``f`` scaled by ``source(t)``.
    sensor_matrix
        Point-evaluation operator of shape ``(2*n_sensors, n)``.
    save_every
        Snapshot cadence in steps (snapshots at steps ``0, save_every, ...``).
    report_interval
        Progress/finite-check cadence in steps (``0`` → every ``n_steps//20``).
    history_chunk_bytes
        Sensor ring-buffer budget (at least one sample is buffered). With no
        sensors, bounds the timestamps per streamed chunk instead.
    sample_every
        Sensor sampling cadence in steps (samples at ``t = 0, sample_every·dt, ...``).
    history_callback
        Receive chronological chunks ``(times, interleaved_sensor_values)`` on
        the host, including the initial sample. Callback exceptions propagate.
    retain_history
        Set False with a history_callback to stream without retaining measurement
        histories. Returned histories have zero rows; the full ``times`` array
        still describes all samples and uses memory proportional to their count.

    Raises
    ------
    FloatingPointError
        If the state becomes non-finite (instability or bad input).
    """
    n_steps = _integer_parameter(n_steps, "n_steps", 1)
    if not isinstance(retain_history, bool) or (not retain_history and history_callback is None):
        raise ValueError("retain_history=False requires a history_callback")
    sample_every = _integer_parameter(sample_every, "sample_every", 1)
    save_every = _integer_parameter(save_every, "save_every", 1)
    report_interval = _integer_parameter(report_interval, "report_interval", 0)
    history_chunk_bytes = _integer_parameter(history_chunk_bytes, "history_chunk_bytes", 1)
    if isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt) or not (np.isfinite(dt) and dt > 0):
        raise ValueError("dt must be finite and positive")
    inv_mass, load_shape = np.asarray(inv_mass), np.asarray(load_shape)
    for name, vector in (("inv_mass", inv_mass), ("load_shape", load_shape)):
        if (
            vector.ndim != 1
            or vector.size == 0
            or np.iscomplexobj(vector)
            or not np.isfinite(vector).all()
        ):
            raise ValueError(f"{name} must be a nonempty, finite real vector")
    if np.any(inv_mass <= 0):
        raise ValueError("inv_mass must be strictly positive")
    log = log_func or (lambda _msg: None)
    n = len(inv_mass)
    K = _as_csr(stiffness, n)
    C = _as_csr(damping, n)
    S = _as_csr(sensor_matrix, n)
    if K.shape != (n, n) or C.shape != (n, n) or len(load_shape) != n or S.shape[1] != n:
        raise ValueError("Inconsistent operator dimensions")
    n_rows = S.shape[0]
    if n_rows % 2:
        raise ValueError("sensor_matrix must have paired x/y rows")
    n_sensors = n_rows // 2
    stacked = sp.hstack([K, C], format="csr")
    stacked.eliminate_zeros()  # sponge damping vanishes outside the absorbing layers

    n_samples = n_steps // sample_every + 1
    times = np.arange(n_samples, dtype=np.float64) * (sample_every * dt)
    host_hist = np.zeros((n_samples if retain_history else 0, n_rows), dtype=backend.dtype)
    if history_callback is not None:
        history_callback(times[:1].copy(), np.zeros((1, n_rows), dtype=backend.dtype))
    check_every = report_interval if report_interval > 0 else max(n_steps // 20, 1)
    h2 = 0.5 * dt

    with backend.context():
        t_setup = pytime.perf_counter()
        A = backend.csr(stacked)
        S_dev = backend.csr(S) if n_rows else None
        inv_m = backend.from_host(inv_mass)
        load = backend.from_host(load_shape)

        z = backend.zeros(2 * n)  # state [u; v]
        zt = backend.zeros(2 * n)  # stage state
        acc = backend.zeros(2 * n)  # RK accumulator
        scratch = backend.zeros(2 * n)
        a = backend.zeros(n)  # acceleration
        u, v = z[:n], z[n:]
        ut, vt = zt[:n], zt[n:]
        du, dv = acc[:n], acc[n:]
        a_scr = scratch[n:]

        sample_bytes = n_rows * backend.dtype.itemsize if n_rows else times.dtype.itemsize
        chunk_rows = int(min(n_samples - 1 or 1, max(1, history_chunk_bytes // sample_bytes)))
        ring = backend.zeros((chunk_rows, n_rows)) if n_rows else None
        backend.synchronize()
        log(
            f"  [{backend.name}] {backend.dtype.name} | DOFs: {n:,} | operator nnz: "
            f"{stacked.nnz:,} | sensors: {n_sensors} | setup "
            f"{(pytime.perf_counter() - t_setup) * 1e3:.1f} ms"
        )

        def acceleration(state: Any, t: float) -> None:
            r = backend.spmv(A, state)
            backend.force_balance(a, float(source(t)), load, r, inv_m)

        def flush(last_row: int, count: int) -> None:
            if count:
                first = last_row - count + 1
                values = (
                    backend.to_host(ring[:count])
                    if ring is not None
                    else np.empty((count, 0), dtype=backend.dtype)
                )
                if retain_history:
                    host_hist[first : last_row + 1] = values
                if history_callback is not None:
                    history_callback(times[first : last_row + 1].copy(), values)

        t_loop = pytime.perf_counter()
        pending = 0
        row = 0  # last recorded sample row (row 0 is the initial state)
        for step in range(n_steps):
            t = step * dt
            # Stage 1: k1 = (v, a(u, v, t))
            acceleration(z, t)
            du[...] = v
            dv[...] = a
            # Stage 2: k2u = v + h/2 k1v ; k2v = a(u + h/2 k1u, k2u)
            backend.axpy(ut, u, h2, v)
            backend.axpy(vt, v, h2, a)
            acceleration(zt, t + h2)
            backend.add_scaled(du, 2.0, vt, scratch[:n])
            backend.add_scaled(dv, 2.0, a, a_scr)
            # Stage 3: ut uses k2u (= vt) before vt is overwritten with k3u
            backend.axpy(ut, u, h2, vt)
            backend.axpy(vt, v, h2, a)
            acceleration(zt, t + h2)
            backend.add_scaled(du, 2.0, vt, scratch[:n])
            backend.add_scaled(dv, 2.0, a, a_scr)
            # Stage 4
            backend.axpy(ut, u, dt, vt)
            backend.axpy(vt, v, dt, a)
            acceleration(zt, t + dt)
            backend.add_scaled(du, 1.0, vt, scratch[:n])
            backend.add_scaled(dv, 1.0, a, a_scr)
            # z += dt/6 * acc
            backend.add_scaled(z, dt / 6.0, acc, scratch)

            if (step + 1) % sample_every == 0:
                if ring is not None:
                    ring[pending] = backend.spmv(S_dev, u)
                pending += 1
                row += 1
                if pending == chunk_rows:
                    flush(row, pending)
                    pending = 0

            if snapshot_callback is not None and step % save_every == 0:
                snapshot_callback(step, t + dt, backend.to_host(u))

            if (step + 1) % check_every == 0 or step + 1 == n_steps:
                if not backend.all_finite(z):
                    raise FloatingPointError(
                        f"Non-finite state at step {step + 1}/{n_steps} (t = {t + dt:.3e} s); "
                        "reduce dt/CFL or check material and damping parameters."
                    )
                if report_interval > 0:
                    elapsed = pytime.perf_counter() - t_loop
                    rate = (step + 1) / max(elapsed, 1e-12)
                    log(
                        f"  [{100.0 * (step + 1) / n_steps:5.1f}%]  step {step + 1}/{n_steps} | "
                        f"{rate:8.1f} steps/s | elapsed {elapsed:6.1f} s | "
                        f"ETA {(n_steps - step - 1) / rate:6.1f} s"
                    )
        flush(row, pending)
        backend.synchronize()
        loop_seconds = pytime.perf_counter() - t_loop

    log(
        f"  [{backend.name}] RK4 loop complete: {loop_seconds:.2f} s "
        f"({n_steps / max(loop_seconds, 1e-12):.1f} steps/s)"
    )
    return ExplicitRK4Result(
        times=times,
        sensor_data_x=np.ascontiguousarray(host_hist[:, 0::2]),
        sensor_data_y=np.ascontiguousarray(host_hist[:, 1::2]),
        loop_seconds=loop_seconds,
    )


def solve_explicit_rk4_device(
    K_csr: CsrLike,
    C_csr: CsrLike,
    M_inv_np: np.ndarray,
    f_spatial_np: np.ndarray,
    sensor_mat_csr: sp.spmatrix | sp.sparray,
    dt: float,
    Nsteps: int,
    source_func: Callable[[float], float],
    device_info: DeviceInfo,
    precision: str = "double",
    save_freq: int = 1,
    snapshot_callback: SnapshotCallback | None = None,
    report_interval: int = 0,
    log_func: Callable[[str], None] | None = None,
    sample_every: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the explicit RK4 engine on ``device_info``; return ``(times, u_x, u_y)``."""
    backend = make_backend(device_info, precision, log_func)
    result = integrate_explicit_rk4(
        K_csr,
        C_csr,
        M_inv_np,
        f_spatial_np,
        sensor_mat_csr,
        dt=dt,
        n_steps=Nsteps,
        source=source_func,
        backend=backend,
        save_every=save_freq,
        snapshot_callback=snapshot_callback,
        report_interval=report_interval,
        log_func=log_func,
        sample_every=sample_every,
    )
    return result.times, result.sensor_data_x, result.sensor_data_y
