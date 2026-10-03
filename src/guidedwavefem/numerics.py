"""Testable numerical utilities for GuidedWaveFEM.

The explicit solver uses classical RK4 on displacement and velocity. The
stability estimate assumes symmetric positive (semi)definite FE operators.

Heavy dependencies (DOLFINx, mpi4py) are imported lazily so that configuration
handling and post-processing do not initialize MPI or the FEM stack.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from dolfinx.mesh import Mesh

ArrayT = TypeVar("ArrayT")


def rk4_step(
    u: ArrayT,
    v: ArrayT,
    t: float,
    dt: float,
    acceleration: Callable[[ArrayT, ArrayT, float], ArrayT],
) -> tuple[ArrayT, ArrayT]:
    """Advance ``ü = a(u, u̇, t)`` by one classical RK4 step.

    This functional reference implementation is used by the distributed
    (PETSc/MPI) explicit solver and as the oracle for the array-backend engine
    in :mod:`guidedwavefem.solvers_gpu`.
    """
    k1u = v
    k1v = acceleration(u, v, t)
    k2u = v + dt * k1v / 2
    k2v = acceleration(u + dt * k1u / 2, k2u, t + dt / 2)
    k3u = v + dt * k2v / 2
    k3v = acceleration(u + dt * k2u / 2, k3u, t + dt / 2)
    k4u = v + dt * k3v
    k4v = acceleration(u + dt * k3u, k4u, t + dt)
    return (
        u + dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6,
        v + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6,
    )


def csr_row_abs_sums(indptr: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Vectorized absolute row sums of a CSR matrix (empty rows give 0)."""
    indptr = np.asarray(indptr)
    n_rows = len(indptr) - 1
    rows = np.repeat(np.arange(n_rows), np.diff(indptr))
    return np.bincount(rows, weights=np.abs(values), minlength=n_rows)


def operator_bound(matrix: Any, mass: np.ndarray) -> float:
    """Upper spectral bound from absolute row sums of ``M_L⁻¹ A`` (Gershgorin).

    Includes the assembled spatial damping, including overlapping sponges.
    Unlike a fixed number of power iterations, this cannot underestimate the
    spectral radius for real matrices (up to floating-point roundoff).
    """
    from mpi4py import MPI

    indptr, _, values = matrix.getValuesCSR()
    sums = csr_row_abs_sums(indptr, values)
    local = float(np.max(sums / mass, initial=0.0))
    return float(matrix.comm.tompi4py().allreduce(local, op=MPI.MAX))


def time_grid(duration: float, requested_dt: float) -> tuple[float, int]:
    """Fit an integer number of steps to the exact requested duration."""
    if not (
        np.isfinite(duration) and duration > 0 and np.isfinite(requested_dt) and requested_dt > 0
    ):
        raise ValueError("duration and time step must be finite and positive")
    steps = max(1, int(np.ceil(duration / requested_dt)))
    return duration / steps, steps


class PointSensors:
    """Evaluate exact FE point values with exactly one owning rank per point."""

    def __init__(self, domain: Mesh, points: Sequence[Sequence[float]] | np.ndarray) -> None:
        from dolfinx import geometry
        from mpi4py import MPI

        self.comm = domain.comm
        points_arr = np.asarray(points, dtype=domain.geometry.x.dtype)
        cells = np.arange(domain.topology.index_map(domain.topology.dim).size_local, dtype=np.int32)
        tree = geometry.bb_tree(domain, domain.topology.dim, entities=cells, padding=1e-12)
        candidates = geometry.compute_collisions_points(tree, points_arr)
        collisions = geometry.compute_colliding_cells(domain, candidates, points_arr)
        local_cells = np.array(
            [
                collisions.links(i)[0] if len(collisions.links(i)) else -1
                for i in range(len(points_arr))
            ],
            dtype=np.int32,
        )
        owners = np.where(local_cells >= 0, self.comm.rank, self.comm.size)
        self.comm.Allreduce(MPI.IN_PLACE, owners, op=MPI.MIN)
        if np.any(owners == self.comm.size):
            raise ValueError(
                "Sensor points outside the mesh: "
                f"{np.flatnonzero(owners == self.comm.size).tolist()}"
            )
        self.indices = np.flatnonzero(owners == self.comm.rank)
        self.points = points_arr[self.indices]
        self.cells = local_cells[self.indices]
        self.count = len(points_arr)

    def sample(self, field: Any) -> np.ndarray:
        """Return ``(n_owned, 2)`` displacement values at the owned points."""
        if not len(self.indices):
            return np.empty((0, 2))
        return field.eval(self.points, self.cells).reshape(-1, 2)

    def gather(self, local: np.ndarray) -> np.ndarray | None:
        """Assemble ``(n_times, n_points)`` histories on rank 0 via ``Gatherv``.

        Each rank sends only its owned columns as one contiguous buffer, which
        avoids pickling and the 2 GiB limit of object-based collectives.
        Returns ``None`` on non-root ranks.
        """
        from mpi4py.util.dtlib import from_numpy_dtype

        local = np.asarray(local)
        n_times = local.shape[0]
        send = np.ascontiguousarray(local.T)  # (n_owned, n_times)
        counts = np.asarray(self.comm.gather(send.size, root=0) or [], dtype=np.int64)
        all_indices = self.comm.gather(self.indices, root=0)
        if self.comm.rank == 0:
            recv = np.empty(int(counts.sum()), dtype=local.dtype)
            displs = np.concatenate([[0], np.cumsum(counts)[:-1]])
            recv_spec = [recv, counts, displs, from_numpy_dtype(local.dtype)]
        else:
            recv_spec = None
        self.comm.Gatherv([send, from_numpy_dtype(local.dtype)], recv_spec, root=0)
        if self.comm.rank:
            return None
        result = np.empty((n_times, self.count), dtype=local.dtype)
        columns = recv.reshape(-1, n_times)
        result[:, np.concatenate(all_indices).astype(np.intp)] = columns.T
        return result


def validate_tensor(
    values: Sequence[float] | np.ndarray, name: str, positive: bool = True
) -> np.ndarray:
    """Return the symmetric 3×3 Voigt matrix after checking definiteness."""
    values_arr = np.asarray(values, dtype=float)
    if values_arr.shape != (6,) or not np.isfinite(values_arr).all():
        raise ValueError(f"{name} must contain six finite Voigt coefficients")
    a, b, c, d, e, f = values_arr
    matrix = np.array([[a, b, c], [b, d, e], [c, e, f]])
    eig = np.linalg.eigvalsh(matrix)
    if (positive and eig[0] <= 0) or (not positive and eig[0] < -1e-12 * max(1.0, abs(eig[-1]))):
        raise ValueError(f"{name} must be positive {'definite' if positive else 'semidefinite'}")
    return matrix


def dispersion_spectrum(
    values: np.ndarray,
    times: Sequence[float] | np.ndarray,
    positions: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positive-frequency, forward-wave power on physical FFT bins.

    Hann tapers reduce leakage. Twofold zero padding interpolates the display,
    but does not improve the physical aperture/record resolution.

    Returns
    -------
    (frequencies [Hz], angular wavenumbers [rad/m], power)
    """
    values_arr = np.asarray(values)
    times_arr, positions_arr = np.asarray(times), np.asarray(positions)
    if values_arr.shape != (len(times_arr), len(positions_arr)) or min(values_arr.shape) < 3:
        raise ValueError("Dispersion requires at least three times and three sensors")
    if not np.isfinite(values_arr).all():
        raise ValueError("Dispersion input must be finite")
    for coordinates in (times_arr, positions_arr):
        spacing = np.diff(coordinates)
        if (
            not np.isfinite(coordinates).all()
            or np.any(spacing <= 0)
            or not np.allclose(spacing, spacing[0], rtol=1e-7, atol=0)
        ):
            raise ValueError("FFT coordinates must be finite, increasing and uniform")
    nt, nx = values_arr.shape
    window = np.hanning(nt)[:, None] * np.hanning(nx)[None, :]
    transform = np.fft.fftshift(np.fft.fft2(values_arr * window, s=(2 * nt, 2 * nx)))
    frequencies = np.fft.fftshift(np.fft.fftfreq(2 * nt, times_arr[1] - times_arr[0]))
    wavenumbers = np.fft.fftshift(np.fft.fftfreq(2 * nx, positions_arr[1] - positions_arr[0]))
    fi = np.flatnonzero(frequencies >= 0)
    ki = np.flatnonzero(wavenumbers <= 0)[::-1]
    power = np.abs(transform[np.ix_(fi, ki)]) ** 2
    return frequencies[fi], -wavenumbers[ki] * 2.0 * np.pi, power
