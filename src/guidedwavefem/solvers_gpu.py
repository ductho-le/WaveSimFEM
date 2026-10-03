"""High-performance, zero-copy GPU and accelerated device explicit solvers.

Implements the zero-transfer full-offloading principle (Trotter et al. 2023),
the precision guidelines for wave propagation (Komatitsch et al. 2010),
the pinned memory telemetry throughput optimizations (Li et al. 2026),
and the element-wise coalesced vector update strategies (Cao et al. 2026).
"""

from __future__ import annotations

import time as pytime
from typing import TYPE_CHECKING, Callable
import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    import dolfinx
    from guidedwavefem.device import DeviceInfo

# ---------------------------------------------------------------------------
# Lazy Optional Imports
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    HAS_CUPY = True
except (ImportError, Exception):
    HAS_CUPY = False
    cp = None
    cp_sparse = None

try:
    import torch

    HAS_TORCH = True
except (ImportError, Exception):
    HAS_TORCH = False
    torch = None

# Fused Elementwise CUDA kernel for CuPy (4-stage RK4 update in a single pass)
if HAS_CUPY:
    try:
        _cupy_rk4_fused_update = cp.ElementwiseKernel(
            "T u, T v, T k1u, T k2u, T k3u, T k4u, T k1v, T k2v, T k3v, T k4v, T dt",
            "T u_out, T v_out",
            """
            u_out = u + (dt / (T)6.0) * (k1u + (T)2.0 * k2u + (T)2.0 * k3u + k4u);
            v_out = v + (dt / (T)6.0) * (k1v + (T)2.0 * k2v + (T)2.0 * k3v + k4v);
            """,
            "rk4_fused_update",
        )
    except Exception:
        _cupy_rk4_fused_update = None
else:
    _cupy_rk4_fused_update = None


def build_sensor_interpolation_matrix(
    V: dolfinx.fem.FunctionSpace,
    points: np.ndarray,
    active_cells: list[int] | np.ndarray,
    n_owned_dofs: int,
) -> sp.csr_matrix:
    """Precompute the exact sparse interpolation matrix for point sensor sampling.

    Evaluates exact finite element basis function weights at each sensor point,
    transforming point sensor evaluation into a single sparse matrix-vector
    multiplication S.dot(u) that can be executed directly on device VRAM.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        Vector displacement function space.
    points : np.ndarray
        Array of sensor probe coordinates (N_sensors, 3).
    active_cells : Sequence[int]
        Cell indices containing each sensor point (-1 if not on this rank).
    n_owned_dofs : int
        Number of owned scalar DOFs.

    Returns
    -------
    scipy.sparse.csr_matrix
        Interpolation operator of shape (2 * n_sensors, n_owned_dofs).
        Row 2*i corresponds to x-displacement, row 2*i + 1 to y-displacement.
    """
    import dolfinx

    bs = V.dofmap.index_map_bs
    n_pts = len(points)
    if n_pts == 0 or len(active_cells) == 0:
        return sp.csr_matrix((0, n_owned_dofs), dtype=np.float64)

    u_tmp = dolfinx.fem.Function(V)
    u_tmp.x.array[:] = 0.0
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for pt_idx, (pt, c) in enumerate(zip(points, active_cells)):
        if c < 0:
            continue
        cell_dofs = V.dofmap.cell_dofs(c)
        for nd in cell_dofs:
            for comp in range(bs):
                scalar_dof = nd * bs + comp
                if scalar_dof < n_owned_dofs:
                    u_tmp.x.array[scalar_dof] = 1.0
                    val = u_tmp.eval(np.array([pt]), [c])
                    u_tmp.x.array[scalar_dof] = 0.0
                    if abs(val[0]) > 1e-14:
                        rows.append(2 * pt_idx)
                        cols.append(scalar_dof)
                        data.append(float(val[0]))
                    if abs(val[1]) > 1e-14:
                        rows.append(2 * pt_idx + 1)
                        cols.append(scalar_dof)
                        data.append(float(val[1]))

    return sp.csr_matrix(
        (data, (rows, cols)), shape=(2 * n_pts, n_owned_dofs), dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Backend 1: CuPy (NVIDIA CUDA)
# ---------------------------------------------------------------------------
def solve_explicit_rk4_cupy(
    K_csr: tuple[np.ndarray, np.ndarray, np.ndarray],  # (indptr, indices, data)
    C_csr: tuple[np.ndarray, np.ndarray, np.ndarray],
    M_inv_np: np.ndarray,
    f_spatial_np: np.ndarray,
    sensor_mat_csr: sp.csr_matrix,
    dt: float,
    Nsteps: int,
    source_func: Callable[[float], float],
    precision: str = "double",
    device_id: int = 0,
    save_freq: int = 1,
    snapshot_callback: Callable[[int, float, np.ndarray], None] | None = None,
    report_interval: int = 0,
    log_func: Callable[[str], None] = print,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute explicit RK4 time-stepping entirely within NVIDIA GPU VRAM via CuPy."""
    if not HAS_CUPY:
        raise RuntimeError("CuPy is not available.")

    dtype = cp.float32 if precision == "single" else cp.float64
    np_dtype = np.float32 if precision == "single" else np.float64
    bytes_per_elem = 4 if precision == "single" else 8

    with cp.cuda.Device(device_id):
        n_dofs = len(M_inv_np)
        n_sensors = sensor_mat_csr.shape[0] // 2

        # 1. ONE-TIME Host -> Device Transfer (Trotter et al. 2023)
        t_trans_start = pytime.perf_counter()

        kindptr, kindices, kdata = K_csr
        K_dev = cp_sparse.csr_matrix(
            (
                cp.asarray(kdata, dtype=dtype),
                cp.asarray(kindices, dtype=cp.int32),
                cp.asarray(kindptr, dtype=cp.int32),
            ),
            shape=(n_dofs, n_dofs),
        )

        cindptr, cindices, cdata = C_csr
        C_dev = cp_sparse.csr_matrix(
            (
                cp.asarray(cdata, dtype=dtype),
                cp.asarray(cindices, dtype=cp.int32),
                cp.asarray(cindptr, dtype=cp.int32),
            ),
            shape=(n_dofs, n_dofs),
        )

        m_inv_dev = cp.asarray(M_inv_np, dtype=dtype)
        f_spatial_dev = cp.asarray(f_spatial_np, dtype=dtype)

        if sensor_mat_csr.nnz > 0:
            S_dev = cp_sparse.csr_matrix(
                (
                    cp.asarray(sensor_mat_csr.data, dtype=dtype),
                    cp.asarray(sensor_mat_csr.indices, dtype=cp.int32),
                    cp.asarray(sensor_mat_csr.indptr, dtype=cp.int32),
                ),
                shape=sensor_mat_csr.shape,
            )
        else:
            S_dev = None

        # 2. State & Pinned Telemetry Allocation (Li et al. 2026 Table 7)
        u_dev = cp.zeros(n_dofs, dtype=dtype)
        v_dev = cp.zeros(n_dofs, dtype=dtype)

        times = np.linspace(0, Nsteps * dt, Nsteps + 1, dtype=np.float64)

        # Allocate pinned host memory for sensor data
        if n_sensors > 0:
            pinned_x = cp.cuda.alloc_pinned_memory(
                (Nsteps + 1) * n_sensors * bytes_per_elem
            )
            pinned_y = cp.cuda.alloc_pinned_memory(
                (Nsteps + 1) * n_sensors * bytes_per_elem
            )
            sensor_data_x = np.frombuffer(pinned_x, dtype=np_dtype).reshape(
                (Nsteps + 1, n_sensors)
            )
            sensor_data_y = np.frombuffer(pinned_y, dtype=np_dtype).reshape(
                (Nsteps + 1, n_sensors)
            )
            sensor_data_x.fill(0.0)
            sensor_data_y.fill(0.0)
        else:
            sensor_data_x = np.zeros((Nsteps + 1, 0), dtype=np_dtype)
            sensor_data_y = np.zeros((Nsteps + 1, 0), dtype=np_dtype)

        comm_stream = cp.cuda.Stream(non_blocking=True)

        t_trans_end = pytime.perf_counter()
        dev_props = cp.cuda.runtime.getDeviceProperties(device_id)
        dev_name = (
            dev_props["name"].decode("utf-8")
            if isinstance(dev_props["name"], bytes)
            else str(dev_props["name"])
        )
        log_func(
            f"  [GPU CuPy] Device: {dev_name} | VRAM transfer: {(t_trans_end - t_trans_start)*1000:.1f} ms"
        )
        log_func(
            f"  [GPU CuPy] Precision: {precision.upper()} | DOFs: {n_dofs:,} | Matrix NNZ: {len(kdata):,}"
        )

        dt_val = dtype(dt)

        def compute_accel(u_vec, v_vec, t_val):
            r = K_dev.dot(u_vec)
            r += C_dev.dot(v_vec)
            f_val = dtype(source_func(float(t_val)))
            return (f_val * f_spatial_dev - r) * m_inv_dev

        t_loop_start = pytime.perf_counter()

        for step in range(Nsteps):
            t_curr = step * dt

            # Stage 1
            k1u = v_dev
            k1v = compute_accel(u_dev, v_dev, t_curr)

            # Stage 2
            u_temp = u_dev + (dtype(0.5) * dt_val) * k1u
            v_temp = v_dev + (dtype(0.5) * dt_val) * k1v
            k2u = v_temp
            k2v = compute_accel(u_temp, v_temp, t_curr + 0.5 * dt)

            # Stage 3
            u_temp = u_dev + (dtype(0.5) * dt_val) * k2u
            v_temp = v_dev + (dtype(0.5) * dt_val) * k2v
            k3u = v_temp
            k3v = compute_accel(u_temp, v_temp, t_curr + 0.5 * dt)

            # Stage 4
            u_temp = u_dev + dt_val * k3u
            v_temp = v_dev + dt_val * k3v
            k4u = v_temp
            k4v = compute_accel(u_temp, v_temp, t_curr + dt)

            # Fused update
            if _cupy_rk4_fused_update is not None:
                _cupy_rk4_fused_update(
                    u_dev,
                    v_dev,
                    k1u,
                    k2u,
                    k3u,
                    k4u,
                    k1v,
                    k2v,
                    k3v,
                    k4v,
                    dt_val,
                    u_dev,
                    v_dev,
                )
            else:
                u_dev += (dt_val / dtype(6.0)) * (
                    k1u + dtype(2.0) * k2u + dtype(2.0) * k3u + k4u
                )
                v_dev += (dt_val / dtype(6.0)) * (
                    k1v + dtype(2.0) * k2v + dtype(2.0) * k3v + k4v
                )

            # Extract sensor readings
            if S_dev is not None:
                s_vals = S_dev.dot(u_dev)
                with comm_stream:
                    s_vals_host = s_vals.get()
                    sensor_data_x[step + 1, :] = s_vals_host[0::2]
                    sensor_data_y[step + 1, :] = s_vals_host[1::2]

            # Snapshots
            if snapshot_callback and step % save_freq == 0:
                snapshot_callback(
                    step, t_curr + dt, cp.asnumpy(u_dev[:n_dofs])
                )

            if report_interval > 0 and (step + 1) % report_interval == 0:
                elapsed = pytime.perf_counter() - t_loop_start
                rate = (step + 1) / elapsed
                log_func(
                    f"  [{100.0*(step+1)/Nsteps:5.1f}%]  step {step + 1}/{Nsteps} | Rate: {rate:6.1f} steps/s | Elapsed: {elapsed:5.1f}s"
                )

        cp.cuda.Stream.null.synchronize()
        comm_stream.synchronize()
        total_time = pytime.perf_counter() - t_loop_start
        log_func(
            f"  [GPU CuPy] RK4 loop complete: {total_time:.2f} s ({Nsteps / max(total_time, 1e-6):.1f} steps/s)"
        )

        return times, sensor_data_x.copy(), sensor_data_y.copy()


# ---------------------------------------------------------------------------
# Backend 2: PyTorch (CUDA / Apple Silicon MPS)
# ---------------------------------------------------------------------------
def solve_explicit_rk4_torch(
    K_csr: tuple[np.ndarray, np.ndarray, np.ndarray],
    C_csr: tuple[np.ndarray, np.ndarray, np.ndarray],
    M_inv_np: np.ndarray,
    f_spatial_np: np.ndarray,
    sensor_mat_csr: sp.csr_matrix,
    dt: float,
    Nsteps: int,
    source_func: Callable[[float], float],
    device_name: str = "mps",
    precision: str = "double",
    save_freq: int = 1,
    snapshot_callback: Callable[[int, float, np.ndarray], None] | None = None,
    report_interval: int = 0,
    log_func: Callable[[str], None] = print,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute explicit RK4 time-stepping via PyTorch Sparse on CUDA or Apple Silicon MPS."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available.")

    torch_dev = torch.device(device_name)
    dtype = (
        torch.float32
        if (precision == "single" or device_name == "mps")
        else torch.float64
    )
    np_dtype = np.float32 if dtype == torch.float32 else np.float64

    n_dofs = len(M_inv_np)
    n_sensors = sensor_mat_csr.shape[0] // 2

    t_trans_start = pytime.perf_counter()

    kindptr, kindices, kdata = K_csr
    K_dev = torch.sparse_csr_tensor(
        torch.as_tensor(kindptr, dtype=torch.int64),
        torch.as_tensor(kindices, dtype=torch.int64),
        torch.as_tensor(kdata, dtype=dtype),
        size=(n_dofs, n_dofs),
        device=torch_dev,
    )

    cindptr, cindices, cdata = C_csr
    C_dev = torch.sparse_csr_tensor(
        torch.as_tensor(cindptr, dtype=torch.int64),
        torch.as_tensor(cindices, dtype=torch.int64),
        torch.as_tensor(cdata, dtype=dtype),
        size=(n_dofs, n_dofs),
        device=torch_dev,
    )

    m_inv_dev = torch.as_tensor(M_inv_np, dtype=dtype, device=torch_dev)
    f_spatial_dev = torch.as_tensor(f_spatial_np, dtype=dtype, device=torch_dev)

    if sensor_mat_csr.nnz > 0:
        S_dev = torch.sparse_csr_tensor(
            torch.as_tensor(sensor_mat_csr.indptr, dtype=torch.int64),
            torch.as_tensor(sensor_mat_csr.indices, dtype=torch.int64),
            torch.as_tensor(sensor_mat_csr.data, dtype=dtype),
            size=sensor_mat_csr.shape,
            device=torch_dev,
        )
    else:
        S_dev = None

    u_dev = torch.zeros(n_dofs, dtype=dtype, device=torch_dev)
    v_dev = torch.zeros(n_dofs, dtype=dtype, device=torch_dev)

    times = np.linspace(0, Nsteps * dt, Nsteps + 1, dtype=np.float64)
    sensor_data_x = np.zeros((Nsteps + 1, n_sensors), dtype=np_dtype)
    sensor_data_y = np.zeros((Nsteps + 1, n_sensors), dtype=np_dtype)

    t_trans_end = pytime.perf_counter()
    log_func(
        f"  [GPU Torch] Device: {device_name.upper()} | VRAM transfer: {(t_trans_end - t_trans_start)*1000:.1f} ms"
    )
    log_func(
        f"  [GPU Torch] Precision: {str(dtype).split('.')[-1].upper()} | DOFs: {n_dofs:,}"
    )

    dt_val = float(dt)

    def compute_accel(u_vec, v_vec, t_val):
        r = torch.mv(K_dev, u_vec)
        r = torch.addmv(r, C_dev, v_vec)
        f_val = float(source_func(float(t_val)))
        return (f_val * f_spatial_dev - r) * m_inv_dev

    t_loop_start = pytime.perf_counter()

    for step in range(Nsteps):
        t_curr = step * dt

        # Stage 1
        k1u = v_dev
        k1v = compute_accel(u_dev, v_dev, t_curr)

        # Stage 2
        u_temp = u_dev + (0.5 * dt_val) * k1u
        v_temp = v_dev + (0.5 * dt_val) * k1v
        k2u = v_temp
        k2v = compute_accel(u_temp, v_temp, t_curr + 0.5 * dt)

        # Stage 3
        u_temp = u_dev + (0.5 * dt_val) * k2u
        v_temp = v_dev + (0.5 * dt_val) * k2v
        k3u = v_temp
        k3v = compute_accel(u_temp, v_temp, t_curr + 0.5 * dt)

        # Stage 4
        u_temp = u_dev + dt_val * k3u
        v_temp = v_dev + dt_val * k3v
        k4u = v_temp
        k4v = compute_accel(u_temp, v_temp, t_curr + dt)

        # Update
        u_dev += (dt_val / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        v_dev += (dt_val / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

        if S_dev is not None:
            s_vals = torch.mv(S_dev, u_dev).cpu().numpy()
            sensor_data_x[step + 1, :] = s_vals[0::2]
            sensor_data_y[step + 1, :] = s_vals[1::2]

        if snapshot_callback and step % save_freq == 0:
            snapshot_callback(step, t_curr + dt, u_dev.cpu().numpy()[:n_dofs])

        if report_interval > 0 and (step + 1) % report_interval == 0:
            elapsed = pytime.perf_counter() - t_loop_start
            rate = (step + 1) / elapsed
            log_func(
                f"  [{100.0*(step+1)/Nsteps:5.1f}%]  step {step + 1}/{Nsteps} | Rate: {rate:6.1f} steps/s | Elapsed: {elapsed:5.1f}s"
            )

    total_time = pytime.perf_counter() - t_loop_start
    log_func(
        f"  [GPU Torch] RK4 loop complete: {total_time:.2f} s ({Nsteps / max(total_time, 1e-6):.1f} steps/s)"
    )

    return times, sensor_data_x, sensor_data_y


# ---------------------------------------------------------------------------
# Backend 3: CPU SciPy (Pure Python / CPU Baseline)
# ---------------------------------------------------------------------------
def solve_explicit_rk4_scipy(
    K_csr: tuple[np.ndarray, np.ndarray, np.ndarray],
    C_csr: tuple[np.ndarray, np.ndarray, np.ndarray],
    M_inv_np: np.ndarray,
    f_spatial_np: np.ndarray,
    sensor_mat_csr: sp.csr_matrix,
    dt: float,
    Nsteps: int,
    source_func: Callable[[float], float],
    precision: str = "double",
    save_freq: int = 1,
    snapshot_callback: Callable[[int, float, np.ndarray], None] | None = None,
    report_interval: int = 0,
    log_func: Callable[[str], None] = print,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute explicit RK4 time-stepping on CPU via SciPy CSR sparse matrix."""
    dtype = np.float32 if precision == "single" else np.float64
    n_dofs = len(M_inv_np)
    n_sensors = sensor_mat_csr.shape[0] // 2

    kindptr, kindices, kdata = K_csr
    K_sp = sp.csr_matrix(
        (kdata.astype(dtype), kindices, kindptr), shape=(n_dofs, n_dofs)
    )

    cindptr, cindices, cdata = C_csr
    C_sp = sp.csr_matrix(
        (cdata.astype(dtype), cindices, cindptr), shape=(n_dofs, n_dofs)
    )

    m_inv = M_inv_np.astype(dtype)
    f_spatial = f_spatial_np.astype(dtype)
    S_sp = sensor_mat_csr.astype(dtype) if sensor_mat_csr.nnz > 0 else None

    u = np.zeros(n_dofs, dtype=dtype)
    v = np.zeros(n_dofs, dtype=dtype)

    times = np.linspace(0, Nsteps * dt, Nsteps + 1, dtype=np.float64)
    sensor_data_x = np.zeros((Nsteps + 1, n_sensors), dtype=dtype)
    sensor_data_y = np.zeros((Nsteps + 1, n_sensors), dtype=dtype)

    log_func(
        f"  [CPU SciPy] Running explicit RK4 ({precision.upper()}) | DOFs: {n_dofs:,}"
    )

    dt_val = dtype(dt)

    def compute_accel(u_vec, v_vec, t_val):
        r = K_sp.dot(u_vec)
        r += C_sp.dot(v_vec)
        f_val = dtype(source_func(float(t_val)))
        return (f_val * f_spatial - r) * m_inv

    t_loop_start = pytime.perf_counter()

    for step in range(Nsteps):
        t_curr = step * dt

        # Stage 1
        k1u = v
        k1v = compute_accel(u, v, t_curr)

        # Stage 2
        u_temp = u + (dtype(0.5) * dt_val) * k1u
        v_temp = v + (dtype(0.5) * dt_val) * k1v
        k2u = v_temp
        k2v = compute_accel(u_temp, v_temp, t_curr + 0.5 * dt)

        # Stage 3
        u_temp = u + (dtype(0.5) * dt_val) * k2u
        v_temp = v + (dtype(0.5) * dt_val) * k2v
        k3u = v_temp
        k3v = compute_accel(u_temp, v_temp, t_curr + 0.5 * dt)

        # Stage 4
        u_temp = u + dt_val * k3u
        v_temp = v + dt_val * k3v
        k4u = v_temp
        k4v = compute_accel(u_temp, v_temp, t_curr + dt)

        # Update
        u += (dt_val / dtype(6.0)) * (
            k1u + dtype(2.0) * k2u + dtype(2.0) * k3u + k4u
        )
        v += (dt_val / dtype(6.0)) * (
            k1v + dtype(2.0) * k2v + dtype(2.0) * k3v + k4v
        )

        if S_sp is not None:
            s_vals = S_sp.dot(u)
            sensor_data_x[step + 1, :] = s_vals[0::2]
            sensor_data_y[step + 1, :] = s_vals[1::2]

        if snapshot_callback and step % save_freq == 0:
            snapshot_callback(step, t_curr + dt, u)

        if report_interval > 0 and (step + 1) % report_interval == 0:
            elapsed = pytime.perf_counter() - t_loop_start
            rate = (step + 1) / elapsed
            log_func(
                f"  [{100.0*(step+1)/Nsteps:5.1f}%]  step {step + 1}/{Nsteps} | Rate: {rate:6.1f} steps/s | Elapsed: {elapsed:5.1f}s"
            )

    total_time = pytime.perf_counter() - t_loop_start
    log_func(
        f"  [CPU SciPy] RK4 loop complete: {total_time:.2f} s ({Nsteps / max(total_time, 1e-6):.1f} steps/s)"
    )

    return times, sensor_data_x, sensor_data_y


# ---------------------------------------------------------------------------
# Universal Dispatcher
# ---------------------------------------------------------------------------
def solve_explicit_rk4_device(
    K_csr: tuple[np.ndarray, np.ndarray, np.ndarray],
    C_csr: tuple[np.ndarray, np.ndarray, np.ndarray],
    M_inv_np: np.ndarray,
    f_spatial_np: np.ndarray,
    sensor_mat_csr: sp.csr_matrix,
    dt: float,
    Nsteps: int,
    source_func: Callable[[float], float],
    device_info: DeviceInfo,
    precision: str = "double",
    save_freq: int = 1,
    snapshot_callback: Callable[[int, float, np.ndarray], None] | None = None,
    report_interval: int = 0,
    log_func: Callable[[str], None] = print,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Route explicit RK4 simulation to the selected hardware backend."""
    if device_info.backend == "cupy" and HAS_CUPY:
        dev_id = 0
        if ":" in device_info.device:
            try:
                dev_id = int(device_info.device.split(":")[1])
            except ValueError:
                dev_id = 0
        return solve_explicit_rk4_cupy(
            K_csr=K_csr,
            C_csr=C_csr,
            M_inv_np=M_inv_np,
            f_spatial_np=f_spatial_np,
            sensor_mat_csr=sensor_mat_csr,
            dt=dt,
            Nsteps=Nsteps,
            source_func=source_func,
            precision=precision,
            device_id=dev_id,
            save_freq=save_freq,
            snapshot_callback=snapshot_callback,
            report_interval=report_interval,
            log_func=log_func,
        )

    if device_info.backend == "torch" and HAS_TORCH:
        return solve_explicit_rk4_torch(
            K_csr=K_csr,
            C_csr=C_csr,
            M_inv_np=M_inv_np,
            f_spatial_np=f_spatial_np,
            sensor_mat_csr=sensor_mat_csr,
            dt=dt,
            Nsteps=Nsteps,
            source_func=source_func,
            device_name=device_info.device,
            precision=precision,
            save_freq=save_freq,
            snapshot_callback=snapshot_callback,
            report_interval=report_interval,
            log_func=log_func,
        )

    # Fallback to optimized SciPy CPU engine
    return solve_explicit_rk4_scipy(
        K_csr=K_csr,
        C_csr=C_csr,
        M_inv_np=M_inv_np,
        f_spatial_np=f_spatial_np,
        sensor_mat_csr=sensor_mat_csr,
        dt=dt,
        Nsteps=Nsteps,
        source_func=source_func,
        precision=precision,
        save_freq=save_freq,
        snapshot_callback=snapshot_callback,
        report_interval=report_interval,
        log_func=log_func,
    )
