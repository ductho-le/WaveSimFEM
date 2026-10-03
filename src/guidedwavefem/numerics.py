"""Testable numerical utilities for GuidedWaveFEM.

The explicit solver uses classical RK4 on displacement and velocity. The
stability estimate assumes symmetric positive (semi)definite FE operators.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from dolfinx import geometry
from mpi4py import MPI

if TYPE_CHECKING:
    from dolfinx.mesh import Mesh


def rk4_step(u, v, t, dt, acceleration):
    """Advance a second-order ODE using four consistent force evaluations."""
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


def operator_bound(matrix: Any, mass: np.ndarray) -> float:
    """Upper spectral bound from absolute row sums of M_L^{-1} A.

    Includes the assembled spatial damping, including overlapping sponges.
    Unlike a fixed number of power iterations, this cannot underestimate the
    spectral radius for real matrices (up to floating-point roundoff).
    """
    indptr, _, values = matrix.getValuesCSR()
    sums = np.array([np.sum(np.abs(values[a:b])) for a, b in pairwise(indptr)])
    local = float(np.max(sums / mass, initial=0.0))
    return float(matrix.comm.tompi4py().allreduce(local, op=MPI.MAX))


def time_grid(duration: float, requested_dt: float) -> tuple[float, int]:
    """Fit an integer number of steps to the exact requested duration."""
    steps = max(1, int(np.ceil(duration / requested_dt)))
    return duration / steps, steps


class PointSensors:
    """Evaluate exact FE point values with one owner per measurement."""

    def __init__(self, domain: Mesh, points: Sequence[Sequence[float]] | np.ndarray) -> None:
        self.comm = domain.comm
        points_arr = np.asarray(points, dtype=domain.geometry.x.dtype)
        cells = np.arange(
            domain.topology.index_map(domain.topology.dim).size_local,
            dtype=np.int32,
        )
        tree = geometry.bb_tree(
            domain, domain.topology.dim, entities=cells, padding=1e-12
        )
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
        if not len(self.indices):
            return np.empty((0, 2))
        return field.eval(self.points, self.cells).reshape(-1, 2)

    def gather(self, local: np.ndarray) -> np.ndarray | None:
        """Only root allocates the global history; each rank sends owned columns."""
        chunks = self.comm.gather((self.indices, local), root=0)
        if self.comm.rank:
            return None
        result = np.empty((local.shape[0], self.count), dtype=local.dtype)
        for indices, values in chunks:
            result[:, indices] = values
        return result


def validate_tensor(values: Sequence[float] | np.ndarray, name: str, positive: bool = True) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    if values_arr.shape != (6,) or not np.isfinite(values_arr).all():
        raise ValueError(f"{name} must contain six finite Voigt coefficients")
    a, b, c, d, e, f = values_arr
    matrix = np.array([[a, b, c], [b, d, e], [c, e, f]])
    eig = np.linalg.eigvalsh(matrix)
    if (positive and eig[0] <= 0) or (
        not positive and eig[0] < -1e-12 * max(1.0, abs(eig[-1]))
    ):
        raise ValueError(
            f"{name} must be positive {'definite' if positive else 'semidefinite'}"
        )
    return matrix


def dispersion_spectrum(
    values: np.ndarray, times: Sequence[float] | np.ndarray, positions: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positive-frequency, forward-wave power on physical FFT bins.

    Hann tapers reduce leakage. Twofold zero padding interpolates the display,
    but does not improve the physical aperture/record resolution.
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
    wavenumbers = np.fft.fftshift(
        np.fft.fftfreq(2 * nx, positions_arr[1] - positions_arr[0])
    )
    fi = np.flatnonzero(frequencies >= 0)
    ki = np.flatnonzero(wavenumbers <= 0)[::-1]
    power = np.abs(transform[np.ix_(fi, ki)]) ** 2
    return frequencies[fi] / 1e6, -wavenumbers[ki] * 2 * np.pi / 1000, power
