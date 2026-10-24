"""Distributed pressure/displacement time integration, independent of FE assembly.

The normal points from solid to fluid. The operators obey
``Ms u'' + Ds u' + Ks u + C p = fs`` and
``Mp p'' + Dp p' + Kp p - C.T u'' = fp``. Pressure velocity ``pdot`` is
the time derivative of pressure, in Pa/s, rather than particle velocity.

All matrices and owned vector entries use matching PETSc distributions. The
caller owns the matrices; the integrators own their states and work objects.
Prescribed DOFs must already be eliminated, including their force corrections.

A forcing callback receives zeroed PETSc vectors ``(t, fs, fp)`` and must be a
pure function of time: the RK4 integrators reuse the end-of-step acceleration
as the next step's first-stage derivative.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from petsc4py import PETSc

from wavesimfem.numerics import ITERATIVE_DOF_THRESHOLD, RK4_STABILITY_RADIUS, operator_bound

if TYPE_CHECKING:
    from wavesimfem.device import DeviceInfo

Forcing = Callable[[float, Any, Any], None]

# Classical RK4 (stage time fraction, accumulation weight).
_RK4_STAGES = ((0.0, 1.0), (0.5, 2.0), (0.5, 2.0), (1.0, 1.0))


@dataclass(frozen=True)
class SeparableForcing:
    """Load ``(fs, fp)(t) = sum_k c_k(t) * (solid[k], pressure[k])``.

    The spatial vectors are fixed PETSc vectors with the restricted field
    layouts; ``None`` marks an identically zero block. ``coefficients(t)``
    returns one scalar per term. Calling the object has the generic
    :data:`Forcing` signature, so every PETSc integrator accepts it unchanged.
    :class:`CoupledArrayRK4` instead uploads the vectors once and forms each
    stage load on its device from the scalar coefficients alone.
    The caller owns the vectors.
    """

    solid: tuple[Any | None, ...]
    pressure: tuple[Any | None, ...]
    coefficients: Callable[[float], Sequence[float]]

    def __post_init__(self) -> None:
        if len(self.solid) != len(self.pressure):
            raise ValueError("SeparableForcing needs one solid and one pressure entry per term")

    def __call__(self, t: float, fs: Any, fp: Any) -> None:
        for coefficient, solid, pressure in zip(
            self.coefficients(t), self.solid, self.pressure, strict=True
        ):
            if coefficient == 0.0:
                continue
            if solid is not None:
                fs.axpy(float(coefficient), solid)
            if pressure is not None:
                fp.axpy(float(coefficient), pressure)


@dataclass(frozen=True)
class AcousticOperators:
    """Restricted assembled matrices; optional damping matrices may be ``None``.

    A pure fluid has square zero-sized solid matrices and a ``(0, np)`` C.
    A pure solid analogously has zero-sized fluid matrices. ``Ms``, ``Mp``
    are positive mass matrices, and stiffness/damping are symmetric positive
    semidefinite. No artificial interface mass belongs in either field mass.
    """

    Ms: Any
    Ks: Any
    Ds: Any | None
    Mp: Any
    Kp: Any
    Dp: Any | None
    C: Any

    def __post_init__(self) -> None:
        ns, np_ = self.Ms.getSize()[0], self.Mp.getSize()[0]
        for matrix, size, name in (
            (self.Ms, ns, "Ms"),
            (self.Ks, ns, "Ks"),
            (self.Ds, ns, "Ds"),
            (self.Mp, np_, "Mp"),
            (self.Kp, np_, "Kp"),
            (self.Dp, np_, "Dp"),
        ):
            if matrix is not None and matrix.getSize() != (size, size):
                raise ValueError(f"Acoustic matrix {name} has an incompatible shape")
        if self.C.getSize() != (ns, np_):
            raise ValueError("C must have solid rows and pressure columns")
        if ns + np_ == 0:
            raise ValueError("The acoustic system has no active unknowns")


def _mass_lump(matrix: Any) -> np.ndarray:
    if matrix.getSize()[0] == 0:
        return np.empty(0, dtype=float)
    ones, lumped = matrix.createVecRight(), matrix.createVecLeft()
    try:
        ones.set(1.0)
        matrix.mult(ones, lumped)
        mass = lumped.array_r.copy()
        from mpi4py import MPI

        bad = bool(np.any(~np.isfinite(mass) | (mass <= 0)))
        if matrix.comm.tompi4py().allreduce(bad, op=MPI.LOR):
            raise ValueError("Acoustic field mass lumping produced non-positive entries")
        return mass
    finally:
        ones.destroy()
        lumped.destroy()


def coupled_rk4_bounds(ops: AcousticOperators) -> dict[str, float]:
    """Conservative spectral certificate for the *coupled*, damped RK4 system.

    Integrating the fluid equation once and setting p=z' gives symmetric mass
    and stiffness, PSD damping, and a skew velocity coupling G=[[0,C],[-C.T,0]].
    After mass normalization let k bound ||K||, d bound ||D||, and g bound ||G||.
    Every nonzero eigenvalue is in the left half-plane and satisfies
    |lambda|**2 <= (d+g)|lambda|+k. Thus the radius below bounds the full
    first-order spectrum. The original pressure system has the same nonzero
    eigenvalues. This argument also covers a stiffness nullspace.

    Rectangular coupling is normalized by both field masses before its 1/inf
    norm bound is taken; raw pressure/displacement row sums mix physical units
    and do not provide this certificate. No dense eigenproblem is assembled.
    """
    ms, mp = _mass_lump(ops.Ms), _mass_lump(ops.Mp)

    def bound(matrix: Any, mass: np.ndarray) -> float:
        return 0.0 if matrix is None or matrix.getSize()[0] == 0 else operator_bound(matrix, mass)

    k = max(bound(ops.Ks, ms), bound(ops.Kp, mp))
    d = max(bound(ops.Ds, ms), bound(ops.Dp, mp))
    g = 0.0
    if all(ops.C.getSize()):
        coupling = ops.C.copy()
        left, right = coupling.createVecLeft(), coupling.createVecRight()
        try:
            left.array[:] = 1.0 / np.sqrt(ms)
            right.array[:] = 1.0 / np.sqrt(mp)
            coupling.diagonalScale(left, right)
            g = np.sqrt(
                coupling.norm(PETSc.NormType.NORM_1) * coupling.norm(PETSc.NormType.NORM_INFINITY)
            )
        finally:
            left.destroy()
            right.destroy()
            coupling.destroy()
    radius = 0.5 * (d + g + np.hypot(d + g, 2.0 * np.sqrt(k)))
    return {
        "stiffness": float(k),
        "damping": float(d),
        "coupling": float(g),
        "radius": float(radius),
        "dt_limit": float(RK4_STABILITY_RADIUS / radius) if radius > 0 else np.inf,
    }


def _direct_solver(matrix: Any, *, indefinite: bool = False) -> Any | None:
    """Once-factorised LU for a constant operator, or ``None`` to stay iterative.

    Serial runs always factorise (PETSc LU, or MUMPS for symmetric-indefinite
    operators because it pivots). Parallel runs factorise with MUMPS below
    :data:`ITERATIVE_DOF_THRESHOLD`; larger systems or builds without MUMPS
    return ``None`` so the caller keeps its preconditioned Krylov solver.
    """
    parallel = matrix.comm.getSize() > 1
    mumps = bool(PETSc.Sys.hasExternalPackage("mumps"))
    if parallel and (not mumps or matrix.getSize()[0] >= ITERATIVE_DOF_THRESHOLD):
        return None
    solver = PETSc.KSP().create(matrix.comm)
    try:
        solver.setOperators(matrix)
        solver.setType(PETSc.KSP.Type.PREONLY)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        if mumps and (parallel or indefinite):
            pc.setFactorSolverType("mumps")
        # Factorise now: failures surface at setup, not in the first step.
        solver.setUp()
    except PETSc.Error:
        solver.destroy()
        if not parallel and not indefinite:
            raise
        return None
    return solver


def _linear_solver(matrix: Any, *, mass: bool = False) -> Any:
    """Solver for a constant SPD operator.

    Serial: LU. Parallel effective operators: a once-factorised MUMPS LU when
    affordable, else CG + block-Jacobi/ICC. Parallel mass matrices: CG + Jacobi.
    """
    if matrix.comm.getSize() == 1:
        solver = PETSc.KSP().create(matrix.comm)
        solver.setOperators(matrix)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        return solver
    if not mass:
        direct = _direct_solver(matrix)
        if direct is not None:
            return direct
    solver = PETSc.KSP().create(matrix.comm)
    solver.setOperators(matrix)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.JACOBI if mass else PETSc.PC.Type.BJACOBI)
    solver.setTolerances(rtol=1e-12, atol=1e-14, max_it=2000)
    if not mass:
        pc = solver.getPC()
        pc.setUp()
        for sub in pc.getBJacobiSubKSP():
            sub.getPC().setType(PETSc.PC.Type.ICC)
    return solver


def _solve_checked(solver: Any, rhs: Any, solution: Any) -> None:
    solver.solve(rhs, solution)
    if solver.getConvergedReason() <= 0:
        raise RuntimeError(
            f"Acoustic linear solve failed (PETSc reason {solver.getConvergedReason()})"
        )


def _inverse_sqrt_diagonal(matrix: Any) -> Any:
    from mpi4py import MPI

    scale = matrix.createVecLeft()
    matrix.getDiagonal(scale)
    bad = bool(np.any(~np.isfinite(scale.array_r) | (scale.array_r <= 0)))
    if matrix.comm.tompi4py().allreduce(bad, op=MPI.LOR):
        scale.destroy()
        raise ValueError("Acoustic field operator has a non-positive diagonal")
    scale.array[:] = 1.0 / np.sqrt(scale.array_r)
    return scale


class _ScaledMassSolver:
    """Normalize a field mass before factoring it, including very small SI entries."""

    def __init__(self, mass: Any) -> None:
        self.scale = _inverse_sqrt_diagonal(mass)
        self.matrix = mass.copy()
        self.matrix.diagonalScale(self.scale, self.scale)
        self.solver = _linear_solver(self.matrix, mass=True)
        self.rhs = mass.createVecLeft()
        self.solution = mass.createVecRight()

    def solve(self, rhs: Any, solution: Any) -> None:
        self.rhs.pointwiseMult(rhs, self.scale)
        self.solver.solve(self.rhs, self.solution)
        solution.pointwiseMult(self.solution, self.scale)

    def getConvergedReason(self) -> int:
        return self.solver.getConvergedReason()

    def destroy(self) -> None:
        for obj in (self.solver, self.matrix, self.rhs, self.solution, self.scale):
            obj.destroy()


class _CoupledState:
    def __init__(
        self,
        ops: AcousticOperators,
        forcing: Forcing | None,
        initial: Mapping[str, Any] | None,
        device: str,
    ) -> None:
        if device not in ("cpu", "auto"):
            raise NotImplementedError(
                "This PETSc acoustic integrator supports CPU execution (serial and MPI). "
                "Use CoupledArrayRK4 for serial CUDA explicit integration."
            )
        self.ops, self.forcing = ops, forcing
        self._objects: list[Any] = []
        self._ns, self._np = ops.Ms.getSize()[0], ops.Mp.getSize()[0]
        for matrix, names in (
            (ops.Ms, ("u", "v", "a", "_fs", "_rs", "_ws")),
            (ops.Mp, ("p", "pdot", "pddot", "pressure_integral", "_fp", "_rp", "_wp")),
        ):
            for name in names:
                setattr(self, name, self._vector(matrix))
        allowed = {"u", "v", "p", "pdot", "pressure_integral"}
        error: str | None = None
        try:
            if initial is None:
                initial = {}
            if not isinstance(initial, Mapping):
                raise ValueError("Acoustic initial fields must be a mapping")
            unknown = set(initial) - allowed
            if unknown:
                raise ValueError(f"Unknown acoustic initial fields: {sorted(map(str, unknown))}")
            for name, values in initial.items():
                target = getattr(self, name)
                if isinstance(values, PETSc.Vec):
                    if (
                        values.getSize() != target.getSize()
                        or values.getOwnershipRange() != target.getOwnershipRange()
                    ):
                        raise ValueError(f"Initial {name} must match its field distribution")
                    array = values.array_r
                else:
                    array = np.asarray(values, dtype=PETSc.ScalarType)
                if array.shape != target.array.shape:
                    raise ValueError(f"Initial {name} must contain its owned field entries")
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"Initial {name} contains non-finite entries")
                target.array[:] = array
        except (ValueError, TypeError, OverflowError, PETSc.Error) as exc:
            error = f"Invalid acoustic initial state: {exc}"
        errors = [e for e in ops.Ms.comm.tompi4py().allgather(error) if e is not None]
        if errors:
            self.destroy()
            raise ValueError(errors[0])

    def _vector(self, matrix: Any) -> Any:
        vector = matrix.createVecLeft()
        vector.set(0.0)
        self._objects.append(vector)
        return vector

    def _duplicate(self, vector: Any) -> Any:
        result = vector.duplicate()
        result.set(0.0)
        self._objects.append(result)
        return result

    def _forces(self, t: float) -> None:
        self._fs.set(0.0)
        self._fp.set(0.0)
        if self.forcing is not None:
            self.forcing(float(t), self._fs, self._fp)

    def _residuals(self, u: Any, v: Any, p: Any, pdot: Any, t: float) -> None:
        self._forces(t)
        self._fs.copy(self._rs)
        if self._ns:
            self.ops.Ks.mult(u, self._ws)
            self._rs.axpy(-1.0, self._ws)
            if self.ops.Ds is not None:
                self.ops.Ds.mult(v, self._ws)
                self._rs.axpy(-1.0, self._ws)
            if self._np:
                self.ops.C.mult(p, self._ws)
                self._rs.axpy(-1.0, self._ws)
        self._fp.copy(self._rp)
        if self._np:
            self.ops.Kp.mult(p, self._wp)
            self._rp.axpy(-1.0, self._wp)
            if self.ops.Dp is not None:
                self.ops.Dp.mult(pdot, self._wp)
                self._rp.axpy(-1.0, self._wp)

    @staticmethod
    def _check_dt(dt: float) -> None:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Acoustic time step must be finite and positive")

    def destroy(self) -> None:
        """Release states/work storage, leaving caller-owned operators intact."""
        for obj in reversed(self._objects):
            obj.destroy()
        self._objects.clear()


def _same_time(cached: float | None, t: float, dt: float) -> bool:
    return cached is not None and abs(t - cached) <= 1e-12 * max(abs(t), dt)


class CoupledRK4(_CoupledState):
    """Classical RK4 with separate row-sum masses and same-stage acceleration coupling.

    ``pressure_integral`` integrates pressure with RK4's stage weights; with
    initially resting fluid it permits v=-rho**-1 grad(integral(p dt)).
    Nonzero particle velocity needs a separate physical initial velocity field.

    The end-of-step acceleration ``a``/``pddot`` is reused as the first-stage
    derivative of the following step, so a step costs four, not five,
    operator evaluations. Reuse requires a matching time and unmodified state
    vectors (PETSc object states, agreed collectively across ranks).
    """

    def __init__(
        self,
        ops: AcousticOperators,
        forcing: Forcing | None = None,
        *,
        initial: Mapping[str, Any] | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(ops, forcing, initial, device)
        try:
            self._inv_ms, self._inv_mp = self._vector(ops.Ms), self._vector(ops.Mp)
            self._inv_ms.array[:] = 1.0 / _mass_lump(ops.Ms)
            self._inv_mp.array[:] = 1.0 / _mass_lump(ops.Mp)
            self.bounds = coupled_rk4_bounds(ops)
            self._state = (self.u, self.v, self.p, self.pdot)
            self._stage = tuple(self._duplicate(v) for v in self._state)
            self._total = tuple(self._duplicate(v) for v in self._state)
            self._stage_acceleration = (self._duplicate(self.a), self._duplicate(self.pddot))
            self._integral_total = self._duplicate(self.p)
            self._acceleration_key: tuple[float, tuple[int, ...]] | None = None
            self._end_acceleration(0.0)
        except BaseException:
            self.destroy()
            raise

    def stable_dt(self, safety: float = 1.0) -> float:
        if not np.isfinite(safety) or not 0 < safety <= 1:
            raise ValueError("RK4 safety factor must be in (0, 1]")
        return float(safety * self.bounds["dt_limit"])

    def _acceleration(
        self,
        u: Any,
        v: Any,
        p: Any,
        pdot: Any,
        t: float,
        au: Any,
        ap: Any,
    ) -> None:
        self._residuals(u, v, p, pdot, t)
        au.pointwiseMult(self._rs, self._inv_ms)
        if self._ns and self._np:
            self.ops.C.multTranspose(au, self._wp)
            self._rp.axpy(1.0, self._wp)
        ap.pointwiseMult(self._rp, self._inv_mp)

    def _vector_states(self) -> tuple[int, ...]:
        return tuple(v.stateGet() for v in (*self._state, self.a, self.pddot))

    def _end_acceleration(self, t: float) -> None:
        self._acceleration(self.u, self.v, self.p, self.pdot, t, self.a, self.pddot)
        self._acceleration_key = (float(t), self._vector_states())

    def _reuse_acceleration(self, t: float, dt: float) -> bool:
        key = self._acceleration_key
        valid = key is not None and _same_time(key[0], t, dt)
        valid = valid and key[1] == self._vector_states()
        comm = self.ops.Ms.comm
        if comm.getSize() > 1:
            # Every rank must agree, since the evaluation is collective.
            from mpi4py import MPI

            valid = comm.tompi4py().allreduce(valid, op=MPI.LAND)
        return bool(valid)

    def step(self, t: float, dt: float) -> None:
        self._check_dt(dt)
        if dt > self.stable_dt() * (1.0 + 1e-12):
            raise ValueError("Acoustic RK4 time step exceeds the coupled stability certificate")
        dv, dpdot = self._stage_acceleration
        stage = self._state
        for index, (fraction, weight) in enumerate(_RK4_STAGES):
            u, v, p, pdot = stage
            if index == 0 and self._reuse_acceleration(t, dt):
                au, ap = self.a, self.pddot
            else:
                self._acceleration(u, v, p, pdot, t + fraction * dt, dv, dpdot)
                au, ap = dv, dpdot
            # Position derivatives are the stage velocities themselves.
            derivatives = (v, au, pdot, ap)
            if index == 0:
                for total, derivative in zip(self._total, derivatives, strict=True):
                    derivative.copy(total)
                p.copy(self._integral_total)
            else:
                for total, derivative in zip(self._total, derivatives, strict=True):
                    total.axpy(weight, derivative)
                self._integral_total.axpy(weight, p)
            if index < 3:
                h = (0.5 if index < 2 else 1.0) * dt
                # Field order (u, v, p, pdot) advances each position before
                # the stage velocity it reads is overwritten.
                for temporary, original, derivative in zip(
                    self._stage, self._state, derivatives, strict=True
                ):
                    temporary.waxpy(h, derivative, original)
                stage = self._stage
        for state, total in zip(self._state, self._total, strict=True):
            state.axpy(dt / 6.0, total)
        self.pressure_integral.axpy(dt / 6.0, self._integral_total)
        self._end_acceleration(t + dt)


_MIRRORS = ("u", "v", "a", "p", "pdot", "pddot", "pressure_integral")


def _mirror(name: str) -> property:
    """PETSc host view of a device state, downloaded only when it is read."""

    def getter(self: CoupledArrayRK4) -> Any:
        if name in self._stale:
            self._sync_mirror(name)
        return self._mirrors[name]

    def setter(self: CoupledArrayRK4, value: Any) -> None:
        self._mirrors[name] = value

    return property(getter, setter, doc=f"Host PETSc mirror of ``{name}`` (read-only state view).")


class CoupledArrayRK4(_CoupledState):
    """Single-process acoustic RK4 on SciPy or a requested CuPy CUDA device.

    The operators and state are normalized by each field's lumped mass before
    conversion to the requested precision. This avoids storing tiny acoustic
    SI mass coefficients in float32 and balances the two physical fields.
    The stage equations are identical to :class:`CoupledRK4`, including the
    same-stage solid acceleration in the pressure equation and the reuse of
    the end-of-step acceleration as the next first-stage derivative.

    ``u``, ``v``, ``p``, ``pdot``, ``a``, ``pddot``, and ``pressure_integral``
    are PETSc host mirrors that are downloaded lazily, on first access after a
    step, and checked for finiteness then. Steps that are not sampled
    therefore perform no device-to-host transfer. A :class:`SeparableForcing`
    is uploaded once and every stage load is formed on the device from its
    scalar coefficients; any other forcing callback is evaluated on PETSc host
    vectors and uploaded per stage. Specify initial fields at construction;
    mirrors are read-only state views for callers. MPI uses
    :class:`CoupledRK4` instead of this serial backend.
    """

    u, v, a = _mirror("u"), _mirror("v"), _mirror("a")
    p, pdot, pddot = _mirror("p"), _mirror("pdot"), _mirror("pddot")
    pressure_integral = _mirror("pressure_integral")

    def __init__(
        self,
        ops: AcousticOperators,
        forcing: Forcing | None = None,
        *,
        initial: Mapping[str, Any] | None = None,
        device: str | DeviceInfo = "cpu",
        precision: str = "double",
    ) -> None:
        import scipy.sparse as sparse

        from wavesimfem.device import DeviceInfo, select_device
        from wavesimfem.solvers_gpu import make_backend

        if ops.Ms.comm.getSize() != 1 or ops.Mp.comm.getSize() != 1:
            raise ValueError("CoupledArrayRK4 requires one MPI rank; use CoupledRK4 for MPI")
        self.device_info = device if isinstance(device, DeviceInfo) else select_device(device)
        self.backend = make_backend(self.device_info, precision)
        self.precision = "single" if self.backend.dtype == np.dtype(np.float32) else "double"
        # The mirror properties store into these before the base class returns.
        self._mirrors: dict[str, Any] = {}
        self._stale: set[str] = set()
        super().__init__(ops, forcing, initial, "cpu")
        self.bounds = coupled_rk4_bounds(ops)
        self._sqrt_ms, self._sqrt_mp = np.sqrt(_mass_lump(ops.Ms)), np.sqrt(_mass_lump(ops.Mp))
        self._inv_sqrt_ms, self._inv_sqrt_mp = 1.0 / self._sqrt_ms, 1.0 / self._sqrt_mp

        def normalized(matrix: Any, left: np.ndarray, right: np.ndarray) -> Any:
            if matrix is None:
                return None
            indptr, indices, values = matrix.getValuesCSR()
            host = sparse.csr_matrix((values, indices, indptr), shape=matrix.getSize(), copy=True)
            host.eliminate_zeros()
            if host.nnz == 0 and matrix in (ops.Ds, ops.Dp):
                return None
            host = sparse.diags(left) @ host @ sparse.diags(right)
            if not np.isfinite(host.data).all():
                raise ValueError("Mass-normalized acoustic operator contains non-finite values")
            converted = self.backend.csr(host.tocsr())
            if not self.backend.all_finite(converted.data):
                raise ValueError("Acoustic operator exceeds the requested floating-point precision")
            return converted

        with self.backend.context():
            self._ks = normalized(ops.Ks, self._inv_sqrt_ms, self._inv_sqrt_ms)
            self._ds = normalized(ops.Ds, self._inv_sqrt_ms, self._inv_sqrt_ms)
            self._kp = normalized(ops.Kp, self._inv_sqrt_mp, self._inv_sqrt_mp)
            self._dp = normalized(ops.Dp, self._inv_sqrt_mp, self._inv_sqrt_mp)
            self._coupling = normalized(ops.C, self._inv_sqrt_ms, self._inv_sqrt_mp)
            self._coupling_t = self._coupling.T.tocsr()
            self._array_state = tuple(
                self.backend.from_host(vector.array_r * scale)
                for vector, scale in (
                    (self.u, self._sqrt_ms),
                    (self.v, self._sqrt_ms),
                    (self.p, self._sqrt_mp),
                    (self.pdot, self._sqrt_mp),
                )
            )
            self._array_stage, self._array_total = (
                tuple(self.backend.zeros(state.shape) for state in self._array_state)
                for _ in range(2)
            )
            self._array_stage_acceleration = (
                self.backend.zeros(self._ns),
                self.backend.zeros(self._np),
            )
            self._scratch = (self.backend.zeros(self._ns), self.backend.zeros(self._np))
            self._array_integral = self.backend.from_host(
                self.pressure_integral.array_r * self._sqrt_mp
            )
            self._array_integral_total = self.backend.zeros(self._np)
            self._array_a, self._array_pddot = (
                self.backend.zeros(self._ns),
                self.backend.zeros(self._np),
            )
            self._device_forcing = self._upload_forcing(forcing)
            self._mirror_sources = {
                "u": (self._array_state[0], self._inv_sqrt_ms),
                "v": (self._array_state[1], self._inv_sqrt_ms),
                "p": (self._array_state[2], self._inv_sqrt_mp),
                "pdot": (self._array_state[3], self._inv_sqrt_mp),
                "a": (self._array_a, self._inv_sqrt_ms),
                "pddot": (self._array_pddot, self._inv_sqrt_mp),
                "pressure_integral": (self._array_integral, self._inv_sqrt_mp),
            }
            self._array_acceleration(*self._array_state, 0.0, self._array_a, self._array_pddot)
            self._acceleration_time: float | None = 0.0
        self._stale = set(_MIRRORS)

    stable_dt = CoupledRK4.stable_dt

    def _upload_forcing(self, forcing: Forcing | None) -> tuple[tuple[Any, Any], ...] | None:
        """Mass-normalized device copies of separable source vectors."""
        if not isinstance(forcing, SeparableForcing):
            return None

        def upload(vector: Any, scale: np.ndarray) -> Any:
            if vector is None or not len(scale):
                return None
            values = vector.array_r
            if values.shape != scale.shape:
                raise ValueError("A separable source vector does not match its field layout")
            values = values * scale
            if not np.isfinite(values).all():
                raise ValueError("Mass-normalized acoustic source contains non-finite values")
            array = self.backend.from_host(values)
            if not self.backend.all_finite(array):
                raise ValueError("Acoustic source exceeds the requested floating-point precision")
            return array

        return tuple(
            (upload(solid, self._inv_sqrt_ms), upload(pressure, self._inv_sqrt_mp))
            for solid, pressure in zip(forcing.solid, forcing.pressure, strict=True)
        )

    def _array_acceleration(
        self, u: Any, v: Any, p: Any, pdot: Any, t: float, au: Any, ap: Any
    ) -> None:
        backend = self.backend
        if self._device_forcing is None:
            self._forces(t)
            au[:] = backend.from_host(self._fs.array_r * self._inv_sqrt_ms)
            ap[:] = backend.from_host(self._fp.array_r * self._inv_sqrt_mp)
        else:
            au.fill(0.0)
            ap.fill(0.0)
            scratch_s, scratch_p = self._scratch
            for coefficient, (solid, pressure) in zip(
                self.forcing.coefficients(t), self._device_forcing, strict=True
            ):
                coefficient = float(coefficient)
                if coefficient == 0.0:
                    continue
                if solid is not None:
                    backend.add_scaled(au, coefficient, solid, scratch_s)
                if pressure is not None:
                    backend.add_scaled(ap, coefficient, pressure, scratch_p)
        if self._ns:
            au -= backend.spmv(self._ks, u)
            if self._ds is not None:
                au -= backend.spmv(self._ds, v)
            if self._np:
                au -= backend.spmv(self._coupling, p)
        if self._np:
            ap -= backend.spmv(self._kp, p)
            if self._dp is not None:
                ap -= backend.spmv(self._dp, pdot)
            if self._ns:
                ap += backend.spmv(self._coupling_t, au)

    def _sync_mirror(self, name: str) -> None:
        state, scale = self._mirror_sources[name]
        with self.backend.context():
            host = self.backend.to_host(state)
        if not np.isfinite(host).all():
            raise FloatingPointError("Acoustic RK4 state contains non-finite values")
        # Rescale in PETSc precision, avoiding extra float32 rounding and
        # retaining representable SI outputs even for very small masses.
        self._mirrors[name].array[:] = host.astype(PETSc.ScalarType, copy=False) * scale
        self._stale.discard(name)

    def step(self, t: float, dt: float) -> None:
        self._check_dt(dt)
        if dt > self.stable_dt() * (1.0 + 1e-12):
            raise ValueError("Acoustic RK4 time step exceeds the coupled stability certificate")
        backend = self.backend
        scratch_s, scratch_p = self._scratch
        scratch = (scratch_s, scratch_s, scratch_p, scratch_p)
        dv, dpdot = self._array_stage_acceleration
        with backend.context():
            stage = self._array_state
            for index, (fraction, weight) in enumerate(_RK4_STAGES):
                u, v, p, pdot = stage
                if index == 0 and _same_time(self._acceleration_time, t, dt):
                    au, ap = self._array_a, self._array_pddot
                else:
                    self._array_acceleration(u, v, p, pdot, t + fraction * dt, dv, dpdot)
                    au, ap = dv, dpdot
                # Position derivatives are the stage velocities themselves.
                derivatives = (v, au, pdot, ap)
                if index == 0:
                    for total, derivative in zip(self._array_total, derivatives, strict=True):
                        total[...] = derivative
                    self._array_integral_total[...] = p
                else:
                    for total, derivative, work in zip(
                        self._array_total, derivatives, scratch, strict=True
                    ):
                        backend.add_scaled(total, weight, derivative, work)
                    backend.add_scaled(self._array_integral_total, weight, p, scratch_p)
                if index < 3:
                    h = (0.5 if index < 2 else 1.0) * dt
                    # Field order (u, v, p, pdot) advances each position before
                    # the stage velocity it reads is overwritten.
                    for temporary, original, derivative in zip(
                        self._array_stage, self._array_state, derivatives, strict=True
                    ):
                        backend.axpy(temporary, original, h, derivative)
                    stage = self._array_stage
            for state, total, work in zip(
                self._array_state, self._array_total, scratch, strict=True
            ):
                backend.add_scaled(state, dt / 6.0, total, work)
            backend.add_scaled(
                self._array_integral, dt / 6.0, self._array_integral_total, scratch_p
            )
            self._acceleration_time = None
            self._array_acceleration(*self._array_state, t + dt, self._array_a, self._array_pddot)
            self._acceleration_time = t + dt
        self._stale = set(_MIRRORS)

    def destroy(self) -> None:
        """Release state and device references without flushing shared CUDA pools."""
        super().destroy()
        self._stale.clear()
        for name in (
            "_ks",
            "_ds",
            "_kp",
            "_dp",
            "_coupling",
            "_coupling_t",
            "_array_state",
            "_array_stage",
            "_array_total",
            "_array_stage_acceleration",
            "_scratch",
            "_array_integral",
            "_array_integral_total",
            "_array_a",
            "_array_pddot",
            "_device_forcing",
            "_mirror_sources",
        ):
            if hasattr(self, name):
                delattr(self, name)


class CoupledNewmark(_CoupledState):
    """Monolithic Newmark with the acceleration coupling in its effective operator.

    The fluid row is multiplied by -1/a0 to symmetrize the lossless coupling;
    positive diagonal scaling balances the pressure/solid units. GMRES with
    field preconditioning solves this indefinite effective system. The default
    beta=1/4, gamma=1/2 is the second-order average-acceleration method.
    """

    def __init__(
        self,
        ops: AcousticOperators,
        dt: float,
        forcing: Forcing | None = None,
        *,
        initial: Mapping[str, Any] | None = None,
        device: str = "cpu",
        beta: float = 0.25,
        gamma: float = 0.5,
    ) -> None:
        self._check_dt(dt)
        if not np.isfinite(beta) or beta <= 0 or not np.isfinite(gamma) or gamma < 0.5:
            raise ValueError("Newmark requires finite beta > 0 and gamma >= 0.5")
        super().__init__(ops, forcing, initial, device)
        try:
            self.dt, self.beta, self.gamma = float(dt), float(beta), float(gamma)
            self._a0, self._a1 = 1.0 / (beta * dt * dt), gamma / (beta * dt)
            self._up, self._vp = self._duplicate(self.u), self._duplicate(self.v)
            self._pp, self._pvp = self._duplicate(self.p), self._duplicate(self.pdot)
            self._old_pressure = self._duplicate(self.p)
            self._solid_damping_input = self._duplicate(self.u)
            self._pressure_damping_input = self._duplicate(self.p)
            self._mass_solvers: list[_ScaledMassSolver | None] = []
            for mass in (ops.Ms, ops.Mp):
                solver = _ScaledMassSolver(mass) if mass.getSize()[0] else None
                if solver is not None:
                    self._objects.append(solver)
                self._mass_solvers.append(solver)
            self._initial_acceleration()
            self._setup_operator()
        except BaseException:
            self.destroy()
            raise

    def _initial_acceleration(self) -> None:
        self._residuals(self.u, self.v, self.p, self.pdot, 0.0)
        if self._ns:
            _solve_checked(self._mass_solvers[0], self._rs, self.a)
        if self._ns and self._np:
            self.ops.C.multTranspose(self.a, self._wp)
            self._rp.axpy(1.0, self._wp)
        if self._np:
            _solve_checked(self._mass_solvers[1], self._rp, self.pddot)

    def _setup_operator(self) -> None:
        self._effective: list[Any] = []
        self._scales: list[Any] = []
        for mass, stiffness, damping, size in (
            (self.ops.Ms, self.ops.Ks, self.ops.Ds, self._ns),
            (self.ops.Mp, self.ops.Kp, self.ops.Dp, self._np),
        ):
            if not size:
                self._effective.append(None)
                self._scales.append(None)
                continue
            operator = stiffness.copy()
            self._objects.append(operator)
            operator.axpy(self._a0, mass, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            if damping is not None:
                operator.axpy(
                    self._a1, damping, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN
                )
            scale = _inverse_sqrt_diagonal(operator)
            self._objects.append(scale)
            self._effective.append(operator)
            self._scales.append(scale)
        # In a coupled system balance the symmetric indefinite block. For a
        # single field retain its SPD effective operator and ordinary scaling.
        if self._ns and self._np:
            As, Ap = self._effective
            ss, sp = self._scales
            sp.scale(np.sqrt(self._a0))
            As.diagonalScale(ss, ss)
            Ap.scale(-1.0 / self._a0)
            Ap.diagonalScale(sp, sp)
            coupling = self.ops.C.copy()
            self._objects.append(coupling)
            coupling.diagonalScale(ss, sp)
            # PETSc's transpose() defaults to an in-place operation. Preserve
            # the rectangular upper block before creating its lower transpose.
            transpose = coupling.copy()
            self._objects.append(transpose)
            transpose.transpose()
            self._A = PETSc.Mat().createNest(
                [[As, coupling], [transpose, Ap]], comm=self.ops.Ms.comm
            )
            self._objects.append(self._A)
            self._A.assemble()
            self._solution_fields = [self._duplicate(self.u), self._duplicate(self.p)]
            self._rhs_fields = [self._duplicate(self.u), self._duplicate(self.p)]
            self._solution = PETSc.Vec().createNest(self._solution_fields, comm=self._A.comm)
            self._rhs = PETSc.Vec().createNest(self._rhs_fields, comm=self._A.comm)
            self._objects.extend((self._solution, self._rhs))
            # The operator is constant: factorise it once and back-substitute
            # each step. The balanced system is symmetric quasi-definite, which
            # MUMPS factorises stably. Each rank owns [solid rows, pressure
            # rows] of the converted matrix, so fields map to contiguous slices.
            parallel = self._A.comm.getSize() > 1
            mumps = bool(PETSc.Sys.hasExternalPackage("mumps"))
            self._solver = None
            if not parallel or (mumps and self._A.getSize()[0] < ITERATIVE_DOF_THRESHOLD):
                # PETSc's convert() defaults to an in-place operation; pass out=
                # so the Nest operator stays intact if factorisation falls back.
                monolithic = self._A.convert(PETSc.Mat.Type.AIJ, out=PETSc.Mat())
                self._solver = _direct_solver(monolithic, indefinite=True)
                if self._solver is not None:
                    self._objects.append(monolithic)
                    self._flat_solution, self._flat_rhs = monolithic.createVecs()
                    self._objects.extend((self._flat_solution, self._flat_rhs))
                    self._split = self._rhs_fields[0].getLocalSize()
                else:
                    monolithic.destroy()
            if self._solver is None:
                self._flat_solution = self._flat_rhs = None
                self._solver = PETSc.KSP().create(self._A.comm)
                self._solver.setOperators(self._A)
                self._solver.setType(PETSc.KSP.Type.GMRES)
                self._solver.setGMRESRestart(100)
                # Displacement errors are divided by dt**2 to recover acceleration.
                # A loose effective-solve tolerance therefore accumulates visible
                # MPI error even when the position residual appears converged.
                # 1e-12 stays far below discretisation error without asking
                # GMRES for machine precision, where it can stagnate.
                self._solver.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
                self._solver.setTolerances(rtol=1e-12, atol=0.0, max_it=2000)
                pc = self._solver.getPC()
                pc.setType(PETSc.PC.Type.FIELDSPLIT)
                pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
                pc.setUp()
                for sub in pc.getFieldSplitSubKSP():
                    sub.setType(PETSc.KSP.Type.PREONLY)
                    sub.getPC().setType(
                        PETSc.PC.Type.LU if self._A.comm.getSize() == 1 else PETSc.PC.Type.BJACOBI
                    )
        else:
            index = 0 if self._ns else 1
            self._A = self._effective[index]
            scale = self._scales[index]
            self._A.diagonalScale(scale, scale)
            self._solution = self._duplicate(self.u if self._ns else self.p)
            self._rhs = self._duplicate(self._solution)
            self._solver = _linear_solver(self._A)
        self._objects.append(self._solver)

    def step(self, t: float, dt: float) -> None:
        self._check_dt(dt)
        if not np.isclose(dt, self.dt, rtol=1e-12, atol=0.0):
            raise ValueError("CoupledNewmark has a fixed time step; rebuild it to change dt")
        self.p.copy(self._old_pressure)
        for position, velocity, acceleration, predictor, velocity_predictor in (
            (self.u, self.v, self.a, self._up, self._vp),
            (self.p, self.pdot, self.pddot, self._pp, self._pvp),
        ):
            position.copy(predictor)
            predictor.axpy(dt, velocity)
            predictor.axpy(dt * dt * (0.5 - self.beta), acceleration)
            velocity.copy(velocity_predictor)
            velocity_predictor.axpy(dt * (1.0 - self.gamma), acceleration)
        self._forces(t + dt)
        self._fs.copy(self._rs)
        self._fp.copy(self._rp)
        for mass, damping, predictor, velocity_predictor, residual, work, damping_input, size in (
            (
                self.ops.Ms,
                self.ops.Ds,
                self._up,
                self._vp,
                self._rs,
                self._ws,
                self._solid_damping_input,
                self._ns,
            ),
            (
                self.ops.Mp,
                self.ops.Dp,
                self._pp,
                self._pvp,
                self._rp,
                self._wp,
                self._pressure_damping_input,
                self._np,
            ),
        ):
            if size:
                mass.mult(predictor, work)
                residual.axpy(self._a0, work)
                if damping is not None:
                    damping_input.waxpy(-1.0 / self._a1, velocity_predictor, predictor)
                    damping.mult(damping_input, work)
                    residual.axpy(self._a1, work)
        if self._ns and self._np:
            self.ops.C.multTranspose(self._up, self._wp)
            self._rp.axpy(-self._a0, self._wp)
            self._rhs_fields[0].pointwiseMult(self._rs, self._scales[0])
            self._rhs_fields[1].pointwiseMult(self._rp, self._scales[1])
            self._rhs_fields[1].scale(-1.0 / self._a0)
            if self._flat_rhs is not None:
                split = self._split
                self._flat_rhs.array[:split] = self._rhs_fields[0].array_r
                self._flat_rhs.array[split:] = self._rhs_fields[1].array_r
                _solve_checked(self._solver, self._flat_rhs, self._flat_solution)
                solution = self._flat_solution.array_r
                self._solution_fields[0].array[:] = solution[:split]
                self._solution_fields[1].array[:] = solution[split:]
            else:
                _solve_checked(self._solver, self._rhs, self._solution)
            self.u.pointwiseMult(self._solution_fields[0], self._scales[0])
            self.p.pointwiseMult(self._solution_fields[1], self._scales[1])
        else:
            index = 0 if self._ns else 1
            residual, state = (self._rs, self.u) if self._ns else (self._rp, self.p)
            self._rhs.pointwiseMult(residual, self._scales[index])
            _solve_checked(self._solver, self._rhs, self._solution)
            state.pointwiseMult(self._solution, self._scales[index])
        for position, velocity, acceleration, predictor, velocity_predictor in (
            (self.u, self.v, self.a, self._up, self._vp),
            (self.p, self.pdot, self.pddot, self._pp, self._pvp),
        ):
            acceleration.waxpy(-1.0, predictor, position)
            acceleration.scale(self._a0)
            velocity.waxpy(self.gamma * dt, acceleration, velocity_predictor)
        self.pressure_integral.axpy(0.5 * dt, self._old_pressure)
        self.pressure_integral.axpy(0.5 * dt, self.p)
