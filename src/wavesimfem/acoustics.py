"""Restricted solid-displacement / fluid-pressure finite elements.

The parent mesh is conforming. Algebraic restrictions remove every DOF outside
its physical subdomain; parent spaces are used only for assembly and evaluation.
All matrices and field restrictions retain distributed PETSc ownership.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from itertools import pairwise
from typing import Any

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, geometry
from dolfinx.fem import petsc as fem_petsc
from dolfinx.mesh import locate_entities, locate_entities_boundary, meshtags
from mpi4py import MPI
from petsc4py import PETSc

from wavesimfem.materials import (
    compute_mesh_layout,
    mesh_vertical_tolerance,
    resolve_sensor_region,
    structure_from_config,
)
from wavesimfem.numerics import PointSensors
from wavesimfem.sources import _resolve_n_cycles


def acoustic_waveform(t: float, cfg: dict[str, Any]) -> tuple[float, float, float]:
    """Unit waveform and exact derivatives, with a C2 causal Ricker startup.

    The tiny pre-pulse Ricker tail is smoothly turned on over 0.25/f0, making
    prescribed pressure compatible with initially resting fluid. Tone bursts
    already have zero value and first two derivatives at their endpoints.
    """
    f = cfg["f_center"]
    if t < 0:
        return 0.0, 0.0, 0.0
    if cfg["source_type"] == "tone_burst":
        n = _resolve_n_cycles(cfg.get("n_cycles"))
        if t > n / f:
            return 0.0, 0.0, 0.0
        w, a = 2 * np.pi * f, 2 * np.pi * f / n
        env = 0.5 * (1 - np.cos(a * t))
        de, dde = 0.5 * a * np.sin(a * t), 0.5 * a * a * np.cos(a * t)
        s, c = np.sin(w * t), np.cos(w * t)
        return env * s, de * s + env * w * c, dde * s + 2 * de * w * c - env * w * w * s
    a = np.pi * f
    x = a * (t - 1.5 / f)
    e = np.exp(-x * x)
    g = (1 - 2 * x * x) * e
    dg = a * (4 * x**3 - 6 * x) * e
    ddg = a * a * (-8 * x**4 + 24 * x * x - 6) * e
    ramp = 0.25 / f
    if t < ramp:
        z = t / ramp
        r = 10 * z**3 - 15 * z**4 + 6 * z**5
        dr = (30 * z**2 - 60 * z**3 + 30 * z**4) / ramp
        ddr = (60 * z - 180 * z**2 + 120 * z**3) / ramp**2
        return r * g, dr * g + r * dg, ddr * g + 2 * dr * dg + r * ddg
    return g, dg, ddg


class RegionSensors(PointSensors):
    """Point evaluation on a specified material side, with unique MPI owners."""

    def __init__(self, domain: Any, points: np.ndarray, cells: np.ndarray) -> None:
        self.comm = domain.comm
        tree = geometry.bb_tree(domain, domain.topology.dim, entities=cells, padding=1e-12)
        candidates = geometry.compute_collisions_points(tree, points)
        hits = geometry.compute_colliding_cells(domain, candidates, points)
        local = np.array(
            [hits.links(i)[0] if len(hits.links(i)) else -1 for i in range(len(points))],
            dtype=np.int32,
        )
        owners = np.where(local >= 0, self.comm.rank, self.comm.size)
        self.comm.Allreduce(MPI.IN_PLACE, owners, op=MPI.MIN)
        if np.any(owners == self.comm.size):
            raise ValueError("Sensor points are outside the selected material region")
        self.indices = np.flatnonzero(owners == self.comm.rank)
        self.points = points[self.indices]
        self.cells = local[self.indices]
        self.count = len(points)


def _assemble(form: Any) -> Any:
    result = fem_petsc.assemble_matrix(fem.form(form))
    result.assemble()
    return result


def _vector(form: Any) -> Any:
    result = fem_petsc.assemble_vector(fem.form(form))
    result.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return result


def _restriction(space: Any, mask: np.ndarray) -> tuple[np.ndarray, Any]:
    im, bs = space.dofmap.index_map, space.dofmap.index_map_bs
    nodes = np.flatnonzero(mask[: im.size_local]).astype(np.int32)
    dofs = (nodes[:, None] * bs + np.arange(bs)).ravel().astype(np.int32)
    global_dofs = (im.local_to_global(nodes)[:, None] * bs + np.arange(bs)).ravel()
    return dofs, PETSc.IS().createGeneral(global_dofs.astype(PETSc.IntType), comm=space.mesh.comm)


def _in_regions(y: np.ndarray, regions: list[dict[str, Any]], tol: float) -> np.ndarray:
    mask = np.zeros(len(y), dtype=bool)
    for r in regions:
        mask |= (y >= r["y_bot"] - tol) & (y <= r["y_top"] + tol)
    return mask


def _cell_indicator(space: Any, cells: np.ndarray) -> Any:
    """DG0 function equal to one on ``cells`` (owned tags) and zero elsewhere."""
    indicator = fem.Function(space)
    indicator.x.array[space.dofmap.list[np.asarray(cells, dtype=np.int32), 0]] = 1.0
    indicator.x.scatter_forward()
    return indicator


def _source_coefficients(cfg: dict[str, Any], t: float) -> tuple[float, float, float]:
    """Multipliers ``(g, -g'', -g')`` of the three fixed load vectors at ``t``.

    ``g`` is the physical source history: ``traction_max`` times the elastic
    source signal for traction loads, otherwise ``source_amplitude`` times
    :func:`acoustic_waveform`. Its derivatives multiply the mass and damping
    corrections of an eliminated prescribed pressure.
    """
    if cfg.get("source_kind", "traction") == "traction":
        from wavesimfem.sources import source_signal

        g = float(cfg["traction_max"]) * float(
            source_signal(t, cfg["source_type"], cfg["f_center"], cfg.get("n_cycles"))
        )
        return g, 0.0, 0.0
    g, dg, ddg = (float(cfg["source_amplitude"]) * a for a in acoustic_waveform(t, cfg))
    return g, -ddg, -dg


@dataclass
class AcousticProblem:
    """Restricted operators, separable loads, sensors, and evaluation spaces.

    ``forcing`` is a :class:`~wavesimfem.acoustic_time.SeparableForcing`
    accepted by every coupled integrator. ``energy_operators`` hold the full
    fluid mass and stiffness on all fluid nodes, including prescribed ones,
    so physical energies are quadratic forms of the reconstructed fields.
    """

    cfg: dict[str, Any]
    summary: dict[str, Any]
    layout: Any
    domain: Any
    V: Any
    Q: Any
    operators: Any
    u_dofs: np.ndarray
    p_dofs: np.ndarray
    u_coords: np.ndarray
    u_components: np.ndarray
    p_coords: np.ndarray
    probes: RegionSensors
    sensor_x: np.ndarray
    sensor_y: float
    sensor_region: dict[str, Any]
    observable: str
    load_y: float
    prescribed: Any
    forcing: Any
    energy_operators: tuple[Any, Any]
    energy_dofs: np.ndarray
    _objects: list[Any] = field(default_factory=list)
    cell_tags: Any = None

    @property
    def ndofs_global(self) -> int:
        return self.operators.Ms.getSize()[0] + self.operators.Mp.getSize()[0]

    @property
    def n_owned_dofs(self) -> int:
        return len(self.u_dofs) + len(self.p_dofs)

    def destroy(self) -> None:
        seen = set()
        for obj in self._objects:
            if id(obj) not in seen:
                obj.destroy()
                seen.add(id(obj))
        self._objects.clear()


@dataclass
class _MeshContext:
    """Mesh, measures, and material partition shared by the assembly helpers."""

    cfg: dict[str, Any]
    summary: dict[str, Any]
    layout: Any
    domain: Any
    cell_tags: Any
    regions: list[dict[str, Any]]
    fluids: list[dict[str, Any]]
    solids: list[dict[str, Any]]
    tags: dict[int, int]
    tol: float
    faces: dict[str, tuple[int, int, float]]
    x: Any
    dx: Any
    ds: Any
    DG: Any

    def is_fluid(self, region: dict[str, Any]) -> bool:
        return any(region is r for r in self.fluids)

    def cells(self, regions: list[dict[str, Any]]) -> np.ndarray:
        found = [self.cell_tags.find(self.tags[id(r)]) for r in regions]
        return np.concatenate(found) if found else np.empty(0, dtype=np.int32)


def _region_band(ctx: _MeshContext, region: dict[str, Any]) -> Any:
    low = _constant(ctx, region["y_bot"] - ctx.tol)
    high = _constant(ctx, region["y_top"] + ctx.tol)
    return ufl.And(ufl.ge(ctx.x[1], low), ufl.le(ctx.x[1], high))


def _mesh_context(cfg: dict[str, Any], summary: dict[str, Any]) -> _MeshContext:
    from wavesimfem.solvers import _create_layered_mesh

    layout = compute_mesh_layout(cfg, summary)
    regions = summary["all_regions"]
    fluids = [r for r in regions if r.get("fluid_formulation") == "acoustic"]
    solids = [r for r in regions if not any(r is f for f in fluids)]
    domain, cell_tags = _create_layered_mesh(summary, layout)
    tol = mesh_vertical_tolerance(cfg, summary)
    tdim, fdim = domain.topology.dim, domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(tdim, fdim)
    faces = {
        "bottom": (1, 1, summary["y_min"]),
        "top": (2, 1, summary["y_max"]),
        "left": (3, 0, 0.0),
        "right": (4, 0, layout.plate_length),
    }
    all_facets, all_values = [], []
    for tag, axis, coordinate in faces.values():
        facets = locate_entities_boundary(
            domain, fdim, lambda z, a=axis, c=coordinate: np.isclose(z[a], c, atol=tol, rtol=0)
        )
        all_facets.extend(facets)
        all_values.extend([tag] * len(facets))
    order = np.argsort(all_facets)
    facet_tags = meshtags(
        domain,
        fdim,
        np.asarray(all_facets, dtype=np.int32)[order],
        np.asarray(all_values, dtype=np.int32)[order],
    )
    return _MeshContext(
        cfg=cfg,
        summary=summary,
        layout=layout,
        domain=domain,
        cell_tags=cell_tags,
        regions=regions,
        fluids=fluids,
        solids=solids,
        tags={id(r): tag for tag, r in enumerate(regions, 1)},
        tol=tol,
        faces=faces,
        x=ufl.SpatialCoordinate(domain),
        dx=ufl.Measure("dx", domain=domain, subdomain_data=cell_tags),
        ds=ufl.Measure("ds", domain=domain, subdomain_data=facet_tags),
        DG=fem.functionspace(domain, ("DG", 0)),
    )


def _interface_coupling(ctx: _MeshContext, V: Any, Q: Any) -> Any:
    """Assemble C = int_Gamma (v . n_solid) p over every solid/fluid interface."""
    fdim = ctx.domain.topology.dim - 1
    interfaces = []
    for low, high in pairwise(ctx.regions):
        if ctx.is_fluid(low) != ctx.is_fluid(high):
            yy = low["y_top"]
            interfaces.extend(
                locate_entities(
                    ctx.domain, fdim, lambda z, yy=yy: np.isclose(z[1], yy, atol=ctx.tol, rtol=0)
                )
            )
    interfaces = np.unique(interfaces).astype(np.int32)
    interface_tags = meshtags(
        ctx.domain, fdim, interfaces, np.ones(len(interfaces), dtype=np.int32)
    )
    dS = ufl.Measure("dS", domain=ctx.domain, subdomain_data=interface_tags)
    solid_flag = _cell_indicator(ctx.DG, ctx.cells(ctx.solids))
    v, p = ufl.TestFunction(V), ufl.TrialFunction(Q)
    normal = ufl.FacetNormal(ctx.domain)
    vs = solid_flag("+") * ufl.dot(v("+"), normal("+")) + solid_flag("-") * ufl.dot(
        v("-"), normal("-")
    )
    pf = (1 - solid_flag("+")) * p("+") + (1 - solid_flag("-")) * p("-")
    return _assemble(vs * pf * dS(1))


def _constant(ctx: _MeshContext, value: float) -> Any:
    """Form coefficient as a Constant: kernels are reused across parameter values."""
    return fem.Constant(ctx.domain, default_scalar_type(value))


def _sponge_indicator(ctx: _MeshContext) -> Any:
    """Quintic absorbing-layer profile, zero outside the lateral/half-space sponges."""
    x, summary, plate = ctx.x, ctx.summary, ctx.layout.plate_length
    length_value = ctx.cfg["absorb_length"]
    ind = ufl.as_ufl(0.0)
    if length_value > 0:
        length = _constant(ctx, length_value)
        right = _constant(ctx, plate - length_value)
        ind += ufl.conditional(ufl.lt(x[0], length), ((length - x[0]) / length) ** 5, 0.0)
        ind += ufl.conditional(ufl.gt(x[0], right), ((x[0] - right) / length) ** 5, 0.0)
    if summary["has_bottom_halfspace"]:
        depth_value = summary["bottom_halfspace"]["absorb_depth"]
        depth = _constant(ctx, depth_value)
        edge = _constant(ctx, summary["y_min"] + depth_value)
        ind += ufl.conditional(ufl.lt(x[1], edge), ((edge - x[1]) / depth) ** 5, 0.0)
    if summary["has_top_halfspace"]:
        depth_value = summary["top_halfspace"]["absorb_depth"]
        depth = _constant(ctx, depth_value)
        edge = _constant(ctx, summary["y_max"] - depth_value)
        ind += ufl.conditional(ufl.gt(x[1], edge), ((x[1] - edge) / depth) ** 5, 0.0)
    return ind


def _field_forms(ctx: _MeshContext, V: Any, Q: Any) -> list[Any]:
    """Bilinear forms ``[Ms, Ks, Ds, Mp, Kp, Dp]`` on the parent spaces."""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    p, w = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    dx, ind = ctx.dx, _sponge_indicator(ctx)
    alpha = _constant(ctx, ctx.cfg["absorb_alpha"])
    beta = _constant(ctx, ctx.cfg["absorb_beta"])
    zero = fem.Constant(ctx.domain, default_scalar_type(0.0))
    ms = zero * ufl.dot(u, v) * dx
    ks, ds_form = ms, ms
    mp = zero * p * w * dx
    kp, dp = mp, mp

    def eps(z: Any) -> Any:
        return ufl.sym(ufl.grad(z))

    def voigt_stress(z: Any, constants: Any) -> Any:
        c11, c12, c16, c22, c26, c66 = (_constant(ctx, c) for c in constants)
        tensor = ufl.as_matrix([[c11, c12, c16], [c12, c22, c26], [c16, c26, c66]])
        e = eps(z)
        stress = tensor * ufl.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])
        return ufl.as_matrix([[stress[0], stress[2]], [stress[2], stress[1]]])

    for tag, reg in enumerate(ctx.regions, 1):
        if ctx.is_fluid(reg):
            m = p * w / _constant(ctx, reg["K"])
            k = ufl.dot(ufl.grad(p), ufl.grad(w)) / _constant(ctx, reg["rho"])
            mp += m * dx(tag)
            kp += k * dx(tag)
            dp += ind * (alpha * m + beta * k) * dx(tag)
        else:
            m = _constant(ctx, reg["rho"]) * ufl.dot(u, v)
            if "C" in reg:
                stress = voigt_stress(u, reg["C"])
            else:
                lmbda, mu = _constant(ctx, reg["lmbda"]), _constant(ctx, reg["mu"])
                stress = lmbda * ufl.div(u) * ufl.Identity(2) + 2 * mu * eps(u)
            k = ufl.inner(stress, eps(v))
            ms += m * dx(tag)
            ks += k * dx(tag)
            ds_form += ind * (alpha * m + beta * k) * dx(tag)
            if reg["material_model"] == "viscoelastic":
                ds_form += ufl.inner(voigt_stress(u, reg["Cv"]), eps(v)) * dx(tag)

    # Exterior pressure faces: rigid (natural), radiation (impedance damping),
    # or pressure release (eliminated in _pressure_constraints).
    for face, (tag, _, _) in ctx.faces.items():
        if ctx.cfg.get("acoustic_boundaries", {}).get(face, "rigid") == "radiation":
            for reg in ctx.fluids:
                impedance = ufl.conditional(
                    _region_band(ctx, reg),
                    _constant(ctx, 1.0 / (reg["rho"] * reg["c_fluid"])),
                    0.0,
                )
                dp += impedance * p * w * ctx.ds(tag)
    return [ms, ks, ds_form, mp, kp, dp]


def _pressure_constraints(
    ctx: _MeshContext, coords_p: np.ndarray, active: np.ndarray
) -> np.ndarray:
    """Active pressure nodes on pressure-release faces (homogeneous Dirichlet)."""
    constrained = np.zeros(len(coords_p), dtype=bool)
    for face, (_, axis, coordinate) in ctx.faces.items():
        if ctx.cfg.get("acoustic_boundaries", {}).get(face, "rigid") == "pressure_release":
            constrained |= np.isclose(coords_p[:, axis], coordinate, atol=ctx.tol, rtol=0) & active
    return constrained


def _source_location(ctx: _MeshContext) -> tuple[str, str, float]:
    """Resolve ``(source_kind, source_face, source_y)``; never relocate a user load."""
    cfg, summary, tol = ctx.cfg, ctx.summary, ctx.tol
    source_kind = cfg.get("source_kind", "traction")
    source_face = cfg.get("source_boundary", "auto")
    if source_face == "auto":
        source_face = (
            "interface" if source_kind == "traction" and summary["has_top_halfspace"] else "top"
        )
    source_y = cfg.get("source_y")
    if source_face == "interface":
        source_y = summary["H_plate"] if source_y is None else source_y
        if not any(np.isclose(source_y, r["y_top"], atol=tol, rtol=0) for r in ctx.regions[:-1]):
            raise ValueError("source_y must identify an internal material interface")
        if source_kind != "traction":
            raise ValueError(
                "Internal coupled interfaces accept solid traction sources; acoustic boundary drives must be exterior"
            )
    elif source_face in ctx.faces:
        _, axis, coordinate = ctx.faces[source_face]
        if source_y is not None and (
            axis == 0 or not np.isclose(source_y, coordinate, atol=tol, rtol=0)
        ):
            # Never silently relocate a user-specified load height.
            raise ValueError(
                f"source_y={source_y:g} m does not apply to source_boundary {source_face!r}; "
                "use source_boundary: interface for an internal load or remove source_y"
            )
        source_y = coordinate if axis == 1 else 0.5 * (summary["y_min"] + summary["y_max"])
    else:
        raise ValueError(f"Unknown source_boundary {source_face!r}")
    return source_kind, source_face, float(source_y)


def _source_forms(
    ctx: _MeshContext,
    V: Any,
    Q: Any,
    location: tuple[str, str, float],
    coords_p: np.ndarray,
    constrained: np.ndarray,
) -> tuple[Any, Any, Any, np.ndarray, float]:
    """Unit-amplitude loads ``(ls, lp)``, prescribed pressure, its nodes, and load height."""
    cfg, tol, x, domain = ctx.cfg, ctx.tol, ctx.x, ctx.domain
    source_kind, source_face, source_y = location
    fdim = domain.topology.dim - 1
    v, w = ufl.TestFunction(V), ufl.TestFunction(Q)
    source_region = cfg.get("source_region")
    target = [r for r in ctx.regions if source_region is None or r["name"] == source_region]
    if not target:
        raise ValueError(f"Unknown source_region {source_region!r}")
    # Same footprint as the elastic solver: Gaussian standard deviation
    # sigma = excitation_width / 3, i.e. exp(-(x - x0)^2 / (2 sigma^2)).
    sigma = cfg["excitation_width"] / 3.0
    profile = (
        ufl.exp(
            -((x[0] - _constant(ctx, cfg["source_x"])) ** 2) / (2.0 * _constant(ctx, sigma) ** 2)
        )
        if source_face not in ("left", "right")
        else 1.0
    )
    zero = fem.Constant(domain, default_scalar_type(0.0))
    ls = zero * v[0] * ctx.dx
    lp = zero * w * ctx.dx
    prescribed = fem.Function(Q)
    drive = np.zeros(len(coords_p), dtype=bool)

    def touching(fluid: bool) -> list[dict[str, Any]]:
        return [
            r
            for r in target
            if ctx.is_fluid(r) == fluid
            and (
                source_face in ("left", "right") or r["y_bot"] - tol <= source_y <= r["y_top"] + tol
            )
        ]

    if source_kind == "acoustic_volume":
        if source_region is None or len(target) != 1 or not ctx.is_fluid(target[0]):
            raise ValueError("acoustic_volume requires an acoustic source_region")
        lp += profile * w * ctx.dx(ctx.tags[id(target[0])])
        source_y = 0.5 * (target[0]["y_bot"] + target[0]["y_top"])
    elif source_kind == "traction":
        eligible = touching(fluid=False)
        if not eligible:
            raise ValueError(
                "A traction source must act on a solid or explicitly regularized_elastic region; choose an acoustic source_kind for fluid"
            )
        fx, fy = cfg["source_fx"], cfg["source_fy"]
        norm = float(np.hypot(fx, fy))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("source_fx and source_fy must not both be zero")
        direction = ufl.as_vector([_constant(ctx, fx / norm), _constant(ctx, fy / norm)])
        if source_face == "interface":
            facet = locate_entities(
                domain, fdim, lambda z: np.isclose(z[1], source_y, atol=tol, rtol=0)
            )
            tags = meshtags(domain, fdim, facet, np.ones(len(facet), dtype=np.int32))
            # Select the desired material trace explicitly, even at bonded solids.
            select = _cell_indicator(ctx.DG, ctx.cells(eligible[:1]))
            trace = select("+") * v("+") + select("-") * v("-")
            ls += (
                profile
                * ufl.dot(direction, trace)
                * ufl.Measure("dS", domain=domain, subdomain_data=tags)(1)
            )
        else:
            for reg in eligible:
                ls += (
                    ufl.conditional(_region_band(ctx, reg), 1.0, 0.0)
                    * profile
                    * ufl.dot(direction, v)
                    * ctx.ds(ctx.faces[source_face][0])
                )
    elif source_kind in ("acoustic_pressure", "acoustic_normal_acceleration"):
        eligible = touching(fluid=True)
        if not eligible:
            raise ValueError("Acoustic boundary source does not touch an acoustic fluid")
        if ctx.cfg.get("acoustic_boundaries", {}).get(source_face, "rigid") != "rigid":
            raise ValueError(
                "A driven pressure face must not also specify pressure_release or radiation"
                if source_kind == "acoustic_pressure"
                else "A prescribed normal-acceleration face cannot also specify pressure release or radiation"
            )
        _, axis, coordinate = ctx.faces[source_face]
        if source_kind == "acoustic_pressure":
            drive = np.isclose(coords_p[:, axis], coordinate, atol=tol, rtol=0) & _in_regions(
                coords_p[:, 1], eligible, tol
            )
            if domain.comm.allreduce(bool(np.any(drive & constrained)), op=MPI.LOR):
                raise ValueError(
                    "Driven pressure intersects a pressure-release boundary; prescribe compatible boundary data on shared nodes"
                )
            values = (
                np.exp(-((coords_p[:, 0] - cfg["source_x"]) ** 2) / (2.0 * sigma**2))
                if axis == 1
                else np.ones(len(coords_p))
            )
            prescribed.x.array[drive] = values[drive]
            prescribed.x.scatter_forward()
        else:
            for reg in eligible:
                lp += (
                    ufl.conditional(_region_band(ctx, reg), 1.0, 0.0)
                    * profile
                    * w
                    * ctx.ds(ctx.faces[source_face][0])
                )
    else:
        raise ValueError(f"Unknown source_kind {source_kind!r}")
    return ls, lp, prescribed, drive, float(source_y)


def _restricted_load(full: Any, dofs: np.ndarray, template: Any) -> Any:
    result = template.duplicate()
    result.array[:] = full.array_r[dofs]
    return result


def _lifted_load(matrix: Any, prescribed: Any, rows: np.ndarray, template: Any) -> Any:
    work = matrix.createVecLeft()
    try:
        matrix.mult(prescribed.x.petsc_vec, work)
        return _restricted_load(work, rows, template)
    finally:
        work.destroy()


def _sensor_layout(
    ctx: _MeshContext,
) -> tuple[np.ndarray, float, str, dict[str, Any], RegionSensors]:
    """Sensor positions, observable, material side, and point evaluator."""
    cfg, summary = ctx.cfg, ctx.summary
    sensor_x = (
        cfg["source_x"]
        + cfg["sensor_offset"]
        + np.arange(int(cfg["n_sensors"])) * cfg["sensor_pitch"]
    )
    sensor_y, observable, selected = resolve_sensor_region(cfg, summary)
    points = np.column_stack([sensor_x, np.full(len(sensor_x), sensor_y), np.zeros(len(sensor_x))])
    probes = RegionSensors(ctx.domain, points, ctx.cells([selected]))
    return sensor_x, sensor_y, observable, selected, probes


def build_acoustic_problem(
    cfg: dict[str, Any], summary: dict[str, Any] | None = None
) -> AcousticProblem:
    """Assemble the physical field blocks, removing inactive/Dirichlet DOFs."""
    import contextlib

    from wavesimfem.acoustic_time import AcousticOperators, SeparableForcing
    from wavesimfem.solvers import _drop_zero_entries

    ctx = _mesh_context(cfg, summary or structure_from_config(cfg))
    degree = int(cfg["degree"])
    V = fem.functionspace(ctx.domain, ("Q", degree, (2,)))
    Q = fem.functionspace(ctx.domain, ("Q", degree))
    coords_p = Q.tabulate_dof_coordinates()
    active_p = _in_regions(coords_p[:, 1], ctx.fluids, ctx.tol)
    constrained = _pressure_constraints(ctx, coords_p, active_p)
    location = _source_location(ctx)
    ls, lp, prescribed, drive, source_y = _source_forms(ctx, V, Q, location, coords_p, constrained)
    constrained |= drive

    coords_u = V.tabulate_dof_coordinates()
    # Objects that the problem will own: released here only if building fails.
    owned: list[Any] = []
    try:
        with contextlib.ExitStack() as temporary:

            def temp(obj: Any) -> Any:
                temporary.callback(obj.destroy)
                return obj

            u_dofs, ui = _restriction(V, _in_regions(coords_u[:, 1], ctx.solids, ctx.tol))
            temp(ui)
            p_dofs, pi = _restriction(Q, active_p & ~constrained)
            temp(pi)
            energy_dofs, ei = _restriction(Q, active_p)
            temp(ei)
            loads = (temp(_vector(ls)), temp(_vector(lp)))

            restricted: list[Any] = []
            tpl_s = tpl_p = lumped_mp = energy_mass = energy_stiffness = None
            lift_c = lift_kp = lift_mp = lift_dp = None
            pending: list[Any] = []
            try:
                for i, form in enumerate(_field_forms(ctx, V, Q)):
                    with contextlib.ExitStack() as mat_stack:
                        full_mat = _assemble(form)
                        mat_stack.callback(full_mat.destroy)
                        rows = ui if i < 3 else pi
                        sub = full_mat.createSubMatrix(rows, rows)
                        if i in (2, 5) and sub.getSize()[0] > 0:
                            raw = sub
                            try:
                                sub = _drop_zero_entries(raw)
                            finally:
                                raw.destroy()
                        restricted.append(sub)
                        owned.append(sub)
                        if i == 0:
                            tpl_s = temp(sub.createVecRight())
                        elif i == 3:
                            tpl_p = temp(sub.createVecRight())
                            ones_q = full_mat.createVecRight()
                            mat_stack.callback(ones_q.destroy)
                            full_lumped_mp = full_mat.createVecLeft()
                            mat_stack.callback(full_lumped_mp.destroy)
                            ones_q.set(1.0)
                            full_mat.mult(ones_q, full_lumped_mp)
                            lumped_mp = _restricted_load(full_lumped_mp, p_dofs, tpl_p)
                            owned.append(lumped_mp)
                            energy_mass = full_mat.createSubMatrix(ei, ei)
                            owned.append(energy_mass)
                            lift_mp = _lifted_load(full_mat, prescribed, p_dofs, tpl_p)
                            pending.append(lift_mp)
                        elif i == 4:
                            energy_stiffness = full_mat.createSubMatrix(ei, ei)
                            owned.append(energy_stiffness)
                            lift_kp = _lifted_load(full_mat, prescribed, p_dofs, tpl_p)
                            pending.append(lift_kp)
                        elif i == 5:
                            lift_dp = _lifted_load(full_mat, prescribed, p_dofs, tpl_p)
                            pending.append(lift_dp)

                with contextlib.ExitStack() as mat_stack:
                    coupling_full = _interface_coupling(ctx, V, Q)
                    mat_stack.callback(coupling_full.destroy)
                    coupling = coupling_full.createSubMatrix(ui, pi)
                    owned.append(coupling)
                    lift_c = _lifted_load(coupling_full, prescribed, u_dofs, tpl_s)
                    pending.append(lift_c)

                ops = AcousticOperators(*restricted, coupling, lumped_mp=lumped_mp)
                energy = (energy_mass, energy_stiffness)

                solid_0 = _restricted_load(loads[0], u_dofs, tpl_s)
                pending.append(solid_0)
                solid_0.axpy(-1.0, lift_c)
                pending.remove(lift_c)
                lift_c.destroy()

                press_0 = _restricted_load(loads[1], p_dofs, tpl_p)
                pending.append(press_0)
                press_0.axpy(-1.0, lift_kp)
                pending.remove(lift_kp)
                lift_kp.destroy()

                kept: list[Any] = []
                for vector in (solid_0, press_0, lift_mp, lift_dp):
                    pending.remove(vector)
                    if vector.norm(PETSc.NormType.NORM_INFINITY) > 0.0:
                        kept.append(vector)
                        owned.append(vector)
                    else:
                        vector.destroy()
                        kept.append(None)
                forcing = SeparableForcing(
                    solid=(kept[0], None, None),
                    pressure=(kept[1], kept[2], kept[3]),
                    coefficients=partial(_source_coefficients, cfg),
                    lumped_pressure=(kept[1], None, kept[3]),
                )
            except BaseException:
                for vec in reversed(pending):
                    vec.destroy()
                raise

        sensor_x, sensor_y, observable, selected, probes = _sensor_layout(ctx)
        result = AcousticProblem(
            cfg=cfg,
            summary=ctx.summary,
            layout=ctx.layout,
            domain=ctx.domain,
            V=V,
            Q=Q,
            operators=ops,
            u_dofs=u_dofs,
            p_dofs=p_dofs,
            u_coords=np.repeat(coords_u[: V.dofmap.index_map.size_local], 2, axis=0)[u_dofs],
            u_components=u_dofs % 2,
            p_coords=coords_p[p_dofs],
            probes=probes,
            sensor_x=sensor_x,
            sensor_y=sensor_y,
            sensor_region=selected,
            observable=observable,
            load_y=source_y,
            prescribed=prescribed,
            forcing=forcing,
            energy_operators=energy,
            energy_dofs=energy_dofs,
            cell_tags=ctx.cell_tags,
        )
    except BaseException:
        for obj in reversed(owned):
            obj.destroy()
        raise
    # Same order as before: restricted operators, coupling, energy, sources.
    result._objects = owned
    return result


class AcousticSampler:
    """Map distributed active states to physical fields and material-side sensors."""

    def __init__(self, problem: AcousticProblem) -> None:
        self.problem = problem
        self.displacement = fem.Function(problem.V, name="solid_displacement")
        self.solid_velocity = fem.Function(problem.V, name="solid_velocity")
        self.pressure = fem.Function(problem.Q, name="acoustic_pressure")
        self.integral = fem.Function(problem.Q)
        self.fluid_velocity = None
        self.velocity_expression = None
        self._has_solid = problem.operators.Ms.getSize()[0] > 0
        self._has_fluid = problem.operators.Mp.getSize()[0] > 0
        self._has_energy = problem.energy_operators[0].getSize()[0] > 0
        self._probe_cells = np.empty(0, dtype=np.int32)
        if problem.observable == "particle_velocity":
            W = fem.functionspace(problem.domain, ("DG", int(problem.cfg["degree"]), (2,)))
            self.fluid_velocity = fem.Function(W, name="fluid_particle_velocity")
            rho = fem.Constant(problem.domain, default_scalar_type(problem.sensor_region["rho"]))
            self.velocity_expression = fem.Expression(
                -ufl.grad(self.integral) / rho,
                W.element.interpolation_points,
            )
            if len(problem.probes.cells):
                self._probe_cells = np.unique(problem.probes.cells).astype(np.int32)
        mass = problem.energy_operators[0]
        self._energy_vectors = (mass.createVecRight(), mass.createVecLeft())

    def update(
        self,
        stepper: Any,
        t: float,
        prescribed_integral: float,
        *,
        need_sensors: bool = True,
        need_energy: bool = True,
        need_vtx: bool = True,
    ) -> np.ndarray:
        p = self.problem
        need_u = (need_vtx and self._has_solid) or (need_sensors and p.observable == "displacement")
        need_v = need_sensors and p.observable == "velocity"
        need_p = (
            (need_vtx and self._has_fluid)
            or (need_energy and self._has_energy)
            or (need_sensors and p.observable == "pressure")
        )
        need_i = (need_energy and self._has_energy) or (
            need_sensors and p.observable == "particle_velocity"
        )
        has_prescribed = p.cfg["source_kind"] == "acoustic_pressure"
        for active, f, values, dofs in (
            (need_u, self.displacement, stepper.u if need_u else None, p.u_dofs),
            (need_v, self.solid_velocity, stepper.v if need_v else None, p.u_dofs),
            (need_p, self.pressure, stepper.p if need_p else None, p.p_dofs),
            (need_i, self.integral, stepper.pressure_integral if need_i else None, p.p_dofs),
        ):
            if not active:
                continue
            f.x.array[:] = 0.0
            f.x.array[dofs] = values.array_r
            if has_prescribed and f is self.pressure:
                f.x.array[:] += (
                    p.cfg["source_amplitude"]
                    * acoustic_waveform(t, p.cfg)[0]
                    * p.prescribed.x.array
                )
            elif has_prescribed and f is self.integral:
                f.x.array[:] += prescribed_integral * p.prescribed.x.array
            f.x.scatter_forward()
        if not need_sensors:
            return np.empty((0, 1 if p.observable == "pressure" else 2), dtype=float)
        if p.observable == "particle_velocity":
            if len(self._probe_cells):
                self.fluid_velocity.interpolate(self.velocity_expression, cells0=self._probe_cells)
            self.fluid_velocity.x.scatter_forward()
            selected = self.fluid_velocity
        else:
            selected = {
                "displacement": self.displacement,
                "velocity": self.solid_velocity,
                "pressure": self.pressure,
            }[p.observable]
        return p.probes.sample(selected)

    def fluid_energy(self) -> tuple[float, float]:
        """Physical pressure and reconstructed particle-velocity energy [J/m].

        ``0.5 p.Mp p`` and ``0.5 I.Kp I`` with the fluid mass/stiffness on all
        fluid nodes, which equal the integrals of ``p**2/(2K)`` and
        ``|grad I|**2/(2 rho)``. Temporal fields are aligned. Conservation is
        expected only in undamped, closed, source-free problems; sponge
        regions are numerical truncation.
        """
        mass, stiffness = self.problem.energy_operators
        if mass.getSize()[0] == 0:
            return 0.0, 0.0
        x, y = self._energy_vectors
        dofs = self.problem.energy_dofs
        energies = []
        for function, matrix in ((self.pressure, mass), (self.integral, stiffness)):
            x.array[:] = function.x.array[dofs]
            matrix.mult(x, y)
            energies.append(0.5 * float(x.dot(y)))
        return energies[0], energies[1]

    def destroy(self) -> None:
        for vector in self._energy_vectors:
            vector.destroy()


class _DeviceAcousticRecorder:
    """Bounded on-device ring buffer for serial explicit acoustic sensors and fluid energy.

    Keeps both point-sensor evaluation and quadratic fluid-energy accumulation
    resident on the array backend (SciPy or CuPy) so recorded steps require no
    per-step device-to-host synchronization, mirror rescaling, or DOLFINx
    function scatter.
    """

    def __init__(
        self,
        problem: AcousticProblem,
        stepper: Any,
        sampler: AcousticSampler,
        local: list[np.ndarray],
        fluid_energy: np.ndarray,
        n_samples: int,
        chunk_bytes: int,
    ) -> None:
        import scipy.sparse as sp

        from wavesimfem.numerics import point_evaluation_matrix

        def _petsc_to_csr(mat: Any) -> sp.csr_matrix:
            indptr, indices, values = mat.getValuesCSR()
            return sp.csr_matrix((values, indices, indptr), shape=mat.getSize(), copy=True)

        self._backend = stepper.backend
        self._local = local
        self._fluid_energy = fluid_energy
        self._cfg = problem.cfg
        self._observable = problem.observable
        self._has_prescribed = problem.cfg["source_kind"] == "acoustic_pressure"
        self._ncomp = 1 if self._observable == "pressure" else 2
        self._pending = 0
        self._row = 0

        if self._observable in ("displacement", "velocity"):
            template_fn = (
                sampler.displacement
                if self._observable == "displacement"
                else sampler.solid_velocity
            )
            dofs = problem.u_dofs
            inv_sqrt = stepper._inv_sqrt_ms
            self._state = stepper._array_state[0 if self._observable == "displacement" else 1]
        else:
            template_fn = sampler.pressure
            dofs = problem.p_dofs
            inv_sqrt = stepper._inv_sqrt_mp
            self._state = stepper._array_state[2]

        s_full = point_evaluation_matrix(
            template_fn, problem.probes.points, problem.probes.cells
        ).tocsr()
        s_full.data[np.abs(s_full.data) <= 1e-14] = 0.0
        s_full.eliminate_zeros()
        s_host = (s_full[:, dofs] @ sp.diags(inv_sqrt, shape=(len(dofs), len(dofs)))).tocsr()
        s_host.eliminate_zeros()
        self._n_sensor_rows = s_host.shape[0]

        backend = self._backend
        with backend.context():
            self._sensor_op = backend.csr(s_host)
            if self._sensor_op.nnz > 0 and not backend.all_finite(self._sensor_op.data):
                raise ValueError(
                    "Acoustic sensor operator exceeds the requested floating-point precision"
                )
            self._s_prescribed: Any = None
            self._sensor_scratch: Any = None
            if self._has_prescribed and self._observable == "pressure":
                s_pres_host = np.asarray(s_full @ problem.prescribed.x.array, dtype=np.float64)
                if np.any(s_pres_host != 0.0):
                    s_pres_dev = backend.from_host(s_pres_host)
                    if not backend.all_finite(s_pres_dev):
                        raise ValueError(
                            "Acoustic sensor operator exceeds the requested floating-point precision"
                        )
                    self._s_prescribed = s_pres_dev
                    self._sensor_scratch = backend.zeros(self._n_sensor_rows)

            self._mp_op: Any = None
            self._kp_op: Any = None
            self._m_cross: Any = None
            self._k_cross: Any = None
            self._m_dd = 0.0
            self._k_dd = 0.0
            self._p_state = stepper._array_state[2]
            self._i_state = stepper._array_integral
            self._energy_ring: Any = None
            self._energy_dd: np.ndarray | None = None
            self._energy_ready = True

            if sampler._has_energy:
                np_ = stepper._np
                if np_ > 0:
                    inv_mp = sp.diags(stepper._inv_sqrt_mp, shape=(np_, np_))
                    mp_norm = (inv_mp @ _petsc_to_csr(problem.operators.Mp) @ inv_mp).tocsr()
                    kp_norm = (inv_mp @ _petsc_to_csr(problem.operators.Kp) @ inv_mp).tocsr()
                    mp_norm.eliminate_zeros()
                    kp_norm.eliminate_zeros()
                    self._mp_op = backend.csr(mp_norm)
                    self._kp_op = backend.csr(kp_norm)
                    if (self._mp_op.nnz > 0 and not backend.all_finite(self._mp_op.data)) or (
                        self._kp_op.nnz > 0 and not backend.all_finite(self._kp_op.data)
                    ):
                        self._energy_ready = False
                if self._energy_ready and self._has_prescribed:
                    d_e = np.asarray(
                        problem.prescribed.x.array[problem.energy_dofs], dtype=np.float64
                    )
                    m_e = _petsc_to_csr(problem.energy_operators[0])
                    k_e = _petsc_to_csr(problem.energy_operators[1])
                    md = m_e @ d_e
                    kd = k_e @ d_e
                    self._m_dd = float(np.dot(d_e, md))
                    self._k_dd = float(np.dot(d_e, kd))
                    if np_ > 0:
                        free_in_energy = np.searchsorted(problem.energy_dofs, problem.p_dofs)
                        m_cross_h = md[free_in_energy] * stepper._inv_sqrt_mp
                        k_cross_h = kd[free_in_energy] * stepper._inv_sqrt_mp
                        if np.any(m_cross_h != 0.0):
                            self._m_cross = backend.from_host(m_cross_h)
                            if not backend.all_finite(self._m_cross):
                                self._energy_ready = False
                        if self._energy_ready and np.any(k_cross_h != 0.0):
                            self._k_cross = backend.from_host(k_cross_h)
                            if not backend.all_finite(self._k_cross):
                                self._energy_ready = False

            energy_cols = 2 if (sampler._has_energy and self._energy_ready) else 0
            sample_bytes = max(1, (self._n_sensor_rows + energy_cols) * backend.dtype.itemsize)
            chunk_rows = int(min(max(1, n_samples - 1), max(1, chunk_bytes // sample_bytes)))
            self._ring = backend.zeros((chunk_rows, self._n_sensor_rows))
            if energy_cols:
                self._energy_ring = backend.zeros((chunk_rows, 2))
                if self._has_prescribed and (self._m_dd != 0.0 or self._k_dd != 0.0):
                    self._energy_dd = np.zeros((chunk_rows, 2), dtype=np.float64)

    @property
    def ready(self) -> bool:
        return self._energy_ready

    def record(self, t: float, boundary_integral: float) -> None:
        backend = self._backend
        xp = backend.xp
        dt_scalar = backend.dtype.type
        g_val = (
            float(self._cfg["source_amplitude"] * acoustic_waveform(t, self._cfg)[0])
            if self._has_prescribed
            else 0.0
        )
        with backend.context():
            if self._sensor_op.nnz > 0:
                sample = backend.spmv(self._sensor_op, self._state)
            else:
                sample = backend.zeros(self._n_sensor_rows)
            if self._s_prescribed is not None and g_val != 0.0:
                backend.add_scaled(sample, g_val, self._s_prescribed, self._sensor_scratch)
            self._ring[self._pending] = sample

            if self._energy_ring is not None:
                if self._mp_op is not None:
                    p_dev = self._p_state
                    i_dev = self._i_state
                    ep = dt_scalar(0.5) * xp.dot(p_dev, backend.spmv(self._mp_op, p_dev))
                    ek = dt_scalar(0.5) * xp.dot(i_dev, backend.spmv(self._kp_op, i_dev))
                    if self._m_cross is not None and g_val != 0.0:
                        ep = ep + dt_scalar(g_val) * xp.dot(self._m_cross, p_dev)
                    if self._k_cross is not None and boundary_integral != 0.0:
                        ek = ek + dt_scalar(boundary_integral) * xp.dot(self._k_cross, i_dev)
                    self._energy_ring[self._pending, 0] = ep
                    self._energy_ring[self._pending, 1] = ek
                else:
                    self._energy_ring[self._pending, 0] = dt_scalar(0.0)
                    self._energy_ring[self._pending, 1] = dt_scalar(0.0)
                if self._energy_dd is not None:
                    self._energy_dd[self._pending, 0] = 0.5 * g_val * g_val * self._m_dd
                    self._energy_dd[self._pending, 1] = (
                        0.5 * boundary_integral * boundary_integral * self._k_dd
                    )
        self._pending += 1
        self._row += 1
        if self._pending == self._ring.shape[0]:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        count = self._pending
        first = self._row - count + 1
        with self._backend.context():
            values = self._backend.to_host(self._ring[:count])
            energy = (
                self._backend.to_host(self._energy_ring[:count])
                if self._energy_ring is not None
                else None
            )
        if self._ncomp == 1:
            self._local[0][first : self._row + 1] = values
        else:
            self._local[0][first : self._row + 1] = values[:, 0::2]
            self._local[1][first : self._row + 1] = values[:, 1::2]
        if energy is not None:
            out = energy.astype(np.float64, copy=False)
            if self._energy_dd is not None:
                out = out + self._energy_dd[:count]
            self._fluid_energy[first : self._row + 1] = out
        self._pending = 0


@dataclass
class AcousticSolveResult:
    """Physical sensor histories, indexed by their archive field names."""

    times: np.ndarray
    histories: dict[str, np.ndarray]
    dt: float
    n_steps: int
    setup_seconds: float
    loop_seconds: float
    runtime: dict[str, Any]
    fluid_energy: np.ndarray


def solve_acoustic(
    problem: AcousticProblem,
    *,
    solver_type: str | None = None,
    device: Any = None,
    history_store: Any = None,
) -> AcousticSolveResult:
    """Integrate an assembled problem without publishing files or owning its mesh.

    A caller-supplied HistoryStore owns disk-backed returned arrays. Otherwise
    returned histories live in memory. Public solve_implicit/solve_explicit
    dispatch here for field-aware problems.

    ``device`` follows the elastic solver policy: a resolved ``DeviceInfo``
    selects the single-process explicit array engine; ``None`` or a device
    name is resolved from the configuration. Implicit and distributed runs use
    PETSc on the CPU in double precision, and reject an explicit accelerator.
    """
    import contextlib
    import os
    import time

    from wavesimfem.acoustic_time import CoupledArrayRK4, CoupledNewmark, CoupledRK4
    from wavesimfem.device import DeviceInfo
    from wavesimfem.materials import implicit_time_step, sampled_time_grid
    from wavesimfem.reporting import log
    from wavesimfem.solvers import _progress, _resolve_device
    from wavesimfem.storage import HISTORY_CHUNK_BYTES, HistoryStore, all_finite_history

    cfg, summary, comm = problem.cfg, problem.summary, problem.domain.comm
    solver_type = solver_type or cfg["solver_type"]
    if isinstance(device, DeviceInfo):
        if solver_type != "explicit" or comm.size > 1:
            raise ValueError(
                "A device selects the single-process explicit array engine; implicit and "
                "distributed acoustic runs use PETSc on the CPU (pass device=None)"
            )
    else:
        device = _resolve_device(
            {**cfg, "device": device or cfg["device"]}, solver_type, size=comm.size, quiet=True
        )
    array_path = device is not None
    started = time.perf_counter()
    with contextlib.ExitStack() as stack:
        if solver_type == "explicit":
            stepper = (
                CoupledArrayRK4(
                    problem.operators,
                    forcing=problem.forcing,
                    device=device,
                    precision=cfg["precision"],
                )
                if array_path
                else CoupledRK4(problem.operators, forcing=problem.forcing)
            )
            stack.callback(stepper.destroy)
            dt_limit = stepper.stable_dt()
            requested = cfg["dt"] or cfg["CFL"] * dt_limit
            if requested > dt_limit * (1 + 1e-12):
                raise ValueError(
                    f"Requested dt={requested:g} exceeds the coupled RK4 stability limit {dt_limit:g}"
                )
            dt, steps, stride = sampled_time_grid(cfg, problem.layout, requested)
        else:
            dt_limit = None
            requested = cfg["dt"] or implicit_time_step(
                problem.layout.mesh_frequency, cfg["steps_per_period"]
            )
            dt, steps, stride = sampled_time_grid(cfg, problem.layout, requested)
            stepper = CoupledNewmark(
                problem.operators,
                dt,
                forcing=problem.forcing,
                beta=cfg["beta_val"],
                gamma=cfg["gamma_val"],
            )
            stack.callback(stepper.destroy)
        sampler = AcousticSampler(problem)
        stack.callback(sampler.destroy)
        ncomp = 1 if problem.observable == "pressure" else 2
        times = np.arange(steps // stride + 1) * (stride * dt)
        histories = history_store or stack.enter_context(HistoryStore())
        local = [histories.zeros((len(times), len(problem.probes.indices))) for _ in range(ncomp)]
        initial = sampler.update(stepper, 0.0, 0.0, need_vtx=bool(cfg["write_vtx"]))
        fluid_energy = np.zeros((len(times), 2))
        fluid_energy[0] = sampler.fluid_energy()
        for j, h in enumerate(local):
            h[0] = initial[:, j]
        writers = []
        if cfg["write_vtx"]:
            from dolfinx.io import VTXWriter

            out_dir = cfg.get("output_dir") or f"results_wavesim_{solver_type}"
            if comm.rank == 0:
                os.makedirs(out_dir, exist_ok=True)
            comm.Barrier()
            # Distinct files prevent differing element value shapes being conflated.
            fields = (
                [(sampler.pressure, "acoustic_pressure.bp")]
                if problem.operators.Mp.getSize()[0]
                else []
            )
            if problem.operators.Ms.getSize()[0]:
                fields.append((sampler.displacement, "solid_displacement.bp"))
            for function, filename in fields:
                writer = VTXWriter(comm, os.path.join(out_dir, filename), [function], engine="BP4")
                stack.callback(writer.close)
                writer.write(0.0)
                writers.append(writer)
        recorder: _DeviceAcousticRecorder | None = None
        if (
            array_path
            and not writers
            and problem.observable in ("pressure", "displacement", "velocity")
            and len(problem.probes.indices) > 0
        ):
            candidate = _DeviceAcousticRecorder(
                problem,
                stepper,
                sampler,
                local,
                fluid_energy,
                len(times),
                HISTORY_CHUNK_BYTES,
            )
            if candidate.ready:
                recorder = candidate

        if comm.rank == 0:
            log(
                f"  Active DOFs: {problem.operators.Ms.getSize()[0]} solid, {problem.operators.Mp.getSize()[0]} pressure"
            )
            log(
                f"  Sensor observable: {problem.observable}; material side: {problem.sensor_region['name']}"
            )
            log(
                f"  Source: {cfg['source_kind']} at y={problem.load_y:g} m; sensors at y={problem.sensor_y:g} m"
            )
            log(f"  Time step: {dt:.6g} s; steps: {steps}; record stride: {stride}")
        setup = time.perf_counter() - started
        loop_start = time.perf_counter()
        boundary_integral = 0.0
        save_stride = max(1, steps // cfg["target_snapshots"])
        report_every = max(1, steps // 10)
        for i in range(steps):
            t = i * dt
            stepper.step(t, dt)
            if cfg["source_kind"] == "acoustic_pressure":
                amplitude = cfg["source_amplitude"]
                if solver_type == "explicit":
                    boundary_integral += (
                        amplitude
                        * dt
                        / 6
                        * (
                            acoustic_waveform(t, cfg)[0]
                            + 4 * acoustic_waveform(t + dt / 2, cfg)[0]
                            + acoustic_waveform(t + dt, cfg)[0]
                        )
                    )
                else:
                    boundary_integral += (
                        amplitude
                        * dt
                        / 2
                        * (acoustic_waveform(t, cfg)[0] + acoustic_waveform(t + dt, cfg)[0])
                    )
            record = (i + 1) % stride == 0
            snapshot = bool(writers and ((i + 1) % save_stride == 0 or i + 1 == steps))
            if recorder is not None and record and not snapshot:
                recorder.record((i + 1) * dt, boundary_integral)
            elif record or snapshot:
                values = sampler.update(
                    stepper,
                    (i + 1) * dt,
                    boundary_integral,
                    need_sensors=record,
                    need_energy=record,
                    need_vtx=snapshot,
                )
                if record:
                    fluid_energy[(i + 1) // stride] = sampler.fluid_energy()
                    for j, h in enumerate(local):
                        h[(i + 1) // stride] = values[:, j]
                if snapshot:
                    for writer in writers:
                        writer.write((i + 1) * dt)
            if (i + 1) % report_every == 0 or i + 1 == steps:
                local_bad = not stepper.all_finite()
                if comm.allreduce(local_bad, op=MPI.LOR):
                    raise FloatingPointError(f"Non-finite coupled acoustic state at step {i + 1}")
                if comm.rank == 0 and (i + 1) % report_every == 0:
                    _progress(i + 1, steps, loop_start)
        if recorder is not None:
            recorder.flush()
        elapsed = time.perf_counter() - loop_start
        if comm.allreduce(
            not all(all_finite_history(h) for h in local) or not np.isfinite(fluid_energy).all(),
            op=MPI.LOR,
        ):
            raise FloatingPointError("Non-finite acoustic sensor histories")
        keys = {
            "pressure": ("sensor_data_pressure",),
            "displacement": ("sensor_data_x", "sensor_data_y"),
            "velocity": ("sensor_data_vx", "sensor_data_vy"),
            "particle_velocity": ("sensor_data_vx", "sensor_data_vy"),
        }[problem.observable]
        engine = stepper.device_info.backend if array_path else "petsc"
        runtime: dict[str, Any] = {
            "sample_stride": stride,
            "integrator": "RK4" if solver_type == "explicit" else "Newmark",
            "engine": engine,
            "backend": engine,
            "device": stepper.device_info.device if array_path else "cpu",
            "precision": cfg["precision"] if array_path else "double",
            "dt_limit": dt_limit,
            "fluid_formulation": "acoustic"
            if summary.get("has_acoustic")
            else ("regularized_elastic" if summary.get("has_fluid") else "none"),
            "solid_dofs": problem.operators.Ms.getSize()[0],
            "pressure_dofs": problem.operators.Mp.getSize()[0],
        }
        if solver_type == "implicit":
            runtime["linear_solver"] = stepper.linear_solver_description
        return AcousticSolveResult(
            times,
            dict(zip(keys, local, strict=True)),
            dt,
            steps,
            setup,
            elapsed,
            runtime,
            fluid_energy,
        )


def run_acoustic_simulation(cfg: dict[str, Any], summary: dict[str, Any]) -> tuple[str, str]:
    """Assemble, integrate, and publish explicitly named physical observables."""
    import contextlib
    import os
    import time

    from wavesimfem.analysis import plot_quick_overview
    from wavesimfem.materials import preflight_estimate
    from wavesimfem.reporting import log, print_structure_table, warn
    from wavesimfem.solvers import (
        _archive_fields,
        _publish_results,
        _resolve_device,
        _runtime_record,
    )
    from wavesimfem.storage import HISTORY_CHUNK_BYTES, HistoryStore

    comm = MPI.COMM_WORLD
    wall_started = time.perf_counter()
    # Same policy as the elastic solver, resolved before the costly assembly.
    device = _resolve_device(cfg, cfg["solver_type"], size=comm.size)
    if comm.rank == 0:
        log(
            f"WaveSimFEM: solid displacement / acoustic pressure ({cfg['solver_type']}, {comm.size} ranks)"
        )
        print_structure_table(summary, rank=0, f0=cfg["f_center"])
        if summary.get("has_legacy_fluid"):
            warn(
                "Explicitly selected regularized_elastic fluid regions retain artificial shear and continuous elastic displacement; results in those regions remain a legacy approximation."
            )
        os.makedirs(cfg["output_dir"], exist_ok=True)
    comm.Barrier()
    preflight_estimate(cfg, rank=comm.rank)
    started = time.perf_counter()
    problem = build_acoustic_problem(cfg, summary)
    assembly_seconds = time.perf_counter() - started
    with contextlib.ExitStack() as stack:
        stack.callback(problem.destroy)
        histories = stack.enter_context(
            HistoryStore(cfg["history_storage"], directory=cfg["output_dir"])
        )
        result = solve_acoustic(problem, device=device, history_store=histories)
        result.setup_seconds += assembly_seconds
        collected = {}
        for name, local in result.histories.items():
            out = (
                histories.zeros((len(result.times), problem.probes.count), dtype=local.dtype)
                if comm.rank == 0 and histories.mode == "disk"
                else None
            )
            collected[name] = problem.probes.gather(local, out=out, chunk_bytes=HISTORY_CHUNK_BYTES)
        path = os.path.join(cfg["output_dir"], "sensor_data.npz")
        if comm.rank == 0:
            wall = time.perf_counter() - wall_started
            loop = max(result.loop_seconds, 1e-12)
            label = cfg["solver_type"].upper()
            log(f"\n{'═' * 58}\n  Performance Summary ({label}, acoustic)\n{'═' * 58}")
            log(f"  Total wall time:     {wall:.2f} s")
            log(f"  ├─ Setup / factorise: {result.setup_seconds:.2f} s")
            log(f"  ├─ Time-step loop:    {result.loop_seconds:.2f} s")
            log(f"  └─ Per-step average:  {loop / result.n_steps * 1e3:.3f} ms/step")
            log(f"  Throughput:           {result.n_steps / loop:.0f} steps/s\n{'═' * 58}")
            unit = {
                "pressure": "Pa",
                "displacement": "m",
                "velocity": "m/s",
                "particle_velocity": "m/s",
            }[problem.observable]
            cfg["sensor_y"], cfg["sensor_region"] = problem.sensor_y, problem.sensor_region["name"]
            cfg["sensor_observable"], cfg["load_y"] = problem.observable, problem.load_y
            cfg["runtime"] = _runtime_record(problem, result)
            archive = {
                **_archive_fields(problem),
                **collected,
                "schema_version": 2,
                "times": result.times,
                "dt": result.dt,
                "sample_stride": result.runtime["sample_stride"],
                "run_id": cfg["runtime"]["run_id"],
                "sensor_observable": problem.observable,
                "sensor_units": unit,
                "sensor_region": problem.sensor_region["name"],
                "sensor_components": np.array(
                    ["pressure"] if problem.observable == "pressure" else ["x", "y"]
                ),
                "source_kind": cfg["source_kind"],
                "source_units": cfg.get("source_units", "Pa"),
                "initial_state": "rest",
                "fluid_pressure_energy": result.fluid_energy[:, 0],
                "fluid_kinetic_energy": result.fluid_energy[:, 1],
                "energy_units": "J/m",
                "energy_description": "Physical acoustic pressure and particle-velocity energy per unit out-of-plane length; velocity integrated from rest. Sponge-domain energy is affected by numerical truncation.",
            }
            _publish_results(cfg, archive, cfg["output_dir"])
            if cfg["quick_plot"] and cfg["run_postprocess"]:
                plot_quick_overview(
                    result.times,
                    problem.sensor_x,
                    list(collected.values())[-1],
                    cfg["f_center"],
                    cfg["source_x"],
                    os.path.join(cfg["output_dir"], "quick_results.png"),
                    source_type=cfg["source_type"],
                    n_cycles=cfg.get("n_cycles"),
                    usetex=cfg["usetex"],
                    observable=problem.observable,
                )
    comm.Barrier()
    return path, cfg["output_dir"]
