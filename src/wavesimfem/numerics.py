"""Testable numerical utilities for WaveSimFEM.

The explicit solver uses classical RK4 on displacement and velocity. The
stability estimate assumes symmetric positive (semi)definite FE operators.

Explicit stability
------------------
For ``M_L ü + C u̇ + K u = f`` with diagonal positive ``M_L`` and symmetric
positive semidefinite ``K`` and ``C``, every eigenpair of the first-order system
satisfies ``λ² m + λ c + k = 0`` with Rayleigh quotients ``m > 0`` and
``c, k ≥ 0``. Complex roots have ``|λ|² = k/m ≤ ρ(M_L⁻¹K)``; real roots have
``|λ| ≤ c/m ≤ ρ(M_L⁻¹C)``; and always ``Re λ ≤ 0``. All eigenvalues therefore
lie in the closed left half-disk of radius ``Ω = max(√B_K, B_C)``, where
``B_K`` and ``B_C`` are Gershgorin bounds (:func:`operator_bound`). RK4 is
stable for ``Δt Ω ≤ R``, with ``R`` the radius of the largest left half-disk in
the RK4 stability region (:data:`RK4_STABILITY_RADIUS`).

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

#: Radius of the largest closed half-disk ``{|z| ≤ R, Re z ≤ 0}`` inside the
#: classical RK4 stability region ``|1 + z + z²/2 + z³/6 + z⁴/24| ≤ 1``. The
#: exact value is 2.6156… (binding boundary point at ``arg z ≈ 122.5°``, see
#: :func:`rk4_half_disk_radius`); it is rounded down for a safety margin.
RK4_STABILITY_RADIUS = 2.6


def rk4_amplification(z: complex | np.ndarray) -> complex | np.ndarray:
    """RK4 amplification factor ``R(z) = 1 + z + z²/2 + z³/6 + z⁴/24``."""
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def rk4_half_disk_radius(n_angles: int = 721, tol: float = 1e-12) -> float:
    """Numerically compute the largest left half-disk inside the RK4 stability region.

    For each ray ``z = r e^{iθ}`` with ``θ ∈ [π/2, π]`` the first exit radius is
    bracketed on a fine grid and refined by bisection; the result is the minimum
    over all rays. Used to verify :data:`RK4_STABILITY_RADIUS`.
    """
    radii = np.linspace(0.0, 3.5, 3501)
    best = np.inf
    for theta in np.linspace(np.pi / 2, np.pi, n_angles):
        ray = np.exp(1j * theta)
        unstable = np.abs(rk4_amplification(radii * ray)) > 1.0 + tol
        unstable[0] = False
        first = int(np.argmax(unstable))
        if not unstable[first]:
            continue
        lo, hi = radii[first - 1], radii[first]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if abs(rk4_amplification(mid * ray)) > 1.0 + tol:
                hi = mid
            else:
                lo = mid
        best = min(best, lo)
    return float(best)


def rk4_step(
    u: ArrayT,
    v: ArrayT,
    t: float,
    dt: float,
    acceleration: Callable[[ArrayT, ArrayT, float], ArrayT],
) -> tuple[ArrayT, ArrayT]:
    """Advance ``ü = a(u, u̇, t)`` by one classical RK4 step.

    This functional reference implementation is the oracle for the distributed
    :class:`PetscRK4` integrator and for the array-backend engine in
    :mod:`wavesimfem.solvers_gpu`.
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


class PetscRK4:
    """Allocation-free classical RK4 for ``M_L ü + C u̇ + K u = s(t) f`` on PETSc vectors.

    Algebraically identical to :func:`rk4_step` (the functional oracle), but all
    stage vectors are preallocated and updated with fused PETSc kernels, so a
    step costs eight SpMVs and a handful of BLAS-1 operations with no temporaries.

    Parameters
    ----------
    stiffness, damping
        Assembled PETSc matrices ``K`` and ``C`` (same parallel layout).
    inv_mass
        Owned entries of the inverse lumped mass ``M_L⁻¹``.
    load
        Owned entries of the spatial load vector ``f``.
    source
        Scalar time function ``s(t)``.
    """

    def __init__(
        self,
        stiffness: Any,
        damping: Any,
        inv_mass: np.ndarray,
        load: np.ndarray,
        source: Callable[[float], float],
    ) -> None:
        self.K, self.C, self.source = stiffness, damping, source
        self._inv_mass = stiffness.createVecLeft()
        self._inv_mass.array[:] = inv_mass
        self._load = stiffness.createVecLeft()
        self._load.array[:] = load
        names = ("u", "v", "_ut", "_vt", "_du", "_dv", "_a")
        for name in names:
            vec = stiffness.createVecLeft()
            vec.set(0.0)
            setattr(self, name, vec)
        self._vectors = [self._inv_mass, self._load, *(getattr(self, n) for n in names)]

    def _acceleration(self, u: Any, v: Any, t: float) -> Any:
        a = self._a
        self.K.mult(u, a)
        self.C.multAdd(v, a, a)  # a = K u + C v
        a.scale(-1.0)
        a.axpy(float(self.source(t)), self._load)
        a.pointwiseMult(a, self._inv_mass)
        return a

    def step(self, t: float, dt: float) -> None:
        """Advance the state ``(u, v)`` from ``t`` to ``t + dt`` in place."""
        u, v, ut, vt, du, dv = self.u, self.v, self._ut, self._vt, self._du, self._dv
        h2 = 0.5 * dt
        a = self._acceleration(u, v, t)
        v.copy(du)
        a.copy(dv)
        ut.waxpy(h2, v, u)
        vt.waxpy(h2, a, v)  # k2u
        a = self._acceleration(ut, vt, t + h2)
        du.axpy(2.0, vt)
        dv.axpy(2.0, a)
        ut.waxpy(h2, vt, u)  # uses k2u before vt becomes k3u
        vt.waxpy(h2, a, v)
        a = self._acceleration(ut, vt, t + h2)
        du.axpy(2.0, vt)
        dv.axpy(2.0, a)
        ut.waxpy(dt, vt, u)
        vt.waxpy(dt, a, v)
        a = self._acceleration(ut, vt, t + dt)
        du.axpy(1.0, vt)
        dv.axpy(1.0, a)
        u.axpy(dt / 6.0, du)
        v.axpy(dt / 6.0, dv)

    def destroy(self) -> None:
        """Release the PETSc work vectors."""
        for vec in self._vectors:
            vec.destroy()


def sample_stride(
    dt: float,
    n_steps: int,
    sample_rate: float,
    *,
    f_max: float | None = None,
    min_samples: int = 1,
) -> int:
    """Steps between recorded sensor samples for a target ``sample_rate`` [Hz].

    The stride is reduced when needed so that the Nyquist frequency stays above
    ``1.25 f_max`` and at least ``min_samples`` samples (including ``t = 0``) are
    recorded. The returned stride is always ≥ 1.
    """
    if not (np.isfinite(dt) and dt > 0 and np.isfinite(sample_rate) and sample_rate > 0):
        raise ValueError("dt and sample_rate must be finite and positive")
    stride = max(1, int(np.floor(1.0 / (sample_rate * dt) * (1.0 + 1e-9))))
    if f_max is not None and f_max > 0:
        stride = min(stride, max(1, int(np.floor(1.0 / (2.5 * f_max * dt)))))
    if min_samples > 1:
        stride = min(stride, max(1, n_steps // (min_samples - 1)))
    return stride


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

    def gather(
        self, local: np.ndarray, *, out: np.ndarray | None = None, chunk_bytes: int = 8 * 1024**2
    ) -> np.ndarray | None:
        """Assemble histories on rank 0 using bounded ``Gatherv`` chunks.

        ``out`` may be a disk-backed array on rank 0. Transfer buffers contain
        at most ``chunk_bytes`` each, or one complete sample row if larger.
        Returns ``None`` on other ranks. All ranks must supply the same number
        of times, dtype and chunk budget.
        """
        from mpi4py.util.dtlib import from_numpy_dtype

        local = np.asarray(local)
        error = None
        if local.ndim != 2 or local.shape[1] != len(self.indices):
            error = "History columns must match the locally owned sensors"
        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
            error = "chunk_bytes must be a positive integer"
        # Validate collectively so a bad rank-local argument cannot strand peers
        # in a later Gatherv when this helper is used outside run_simulation.
        metadata = self.comm.allgather((error, local.shape, local.dtype.str, chunk_bytes))
        errors = [entry[0] for entry in metadata if entry[0] is not None]
        if errors:
            raise ValueError(errors[0])
        if any(
            shape[0] != local.shape[0] or dtype != local.dtype.str or budget != chunk_bytes
            for _, shape, dtype, budget in metadata
        ):
            raise ValueError("History times, dtype and chunk budget must agree across ranks")
        n_times = local.shape[0]
        owners = np.asarray(self.comm.gather(len(self.indices), root=0) or [], dtype=np.int64)
        all_indices = self.comm.gather(self.indices, root=0)
        rows = max(1, chunk_bytes // max(1, self.count * local.dtype.itemsize))
        mpi_dtype = from_numpy_dtype(local.dtype)
        if (
            self.comm.rank == 0
            and out is not None
            and (
                out.shape != (n_times, self.count)
                or out.dtype != local.dtype
                or not out.flags.writeable
            )
        ):
            error = "Gather output must be writable with the global history shape and dtype"
        error = self.comm.bcast(error, root=0)
        if error is not None:
            raise ValueError(error)
        if self.comm.rank == 0:
            result = np.empty((n_times, self.count), dtype=local.dtype) if out is None else out
            indices = np.concatenate(all_indices).astype(np.intp)
            recv = np.empty(min(n_times, rows) * self.count, dtype=local.dtype)
        for first in range(0, n_times, rows):
            last = min(first + rows, n_times)
            size = last - first
            send = np.ascontiguousarray(local[first:last].T)
            recv_spec = None
            if self.comm.rank == 0:
                counts = owners * size
                displs = np.concatenate([[0], np.cumsum(counts)[:-1]])
                recv_spec = [recv[: size * self.count], counts, displs, mpi_dtype]
            self.comm.Gatherv([send, mpi_dtype], recv_spec, root=0)
            if self.comm.rank == 0:
                result[first:last, indices] = recv[: size * self.count].reshape(-1, size).T
        return result if self.comm.rank == 0 else None


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
