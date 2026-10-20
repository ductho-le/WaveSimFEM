"""Console reporting: package logger, verbosity control, and summary tables.

All user-facing progress output goes through the ``wavesimfem`` logger so
that applications can silence, redirect, or capture it with the standard
:mod:`logging` machinery. By default messages are written to the *current*
``sys.stdout`` (resolved at emit time, so pytest/IPython capture works).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from typing import Any

from wavesimfem.units import fmt_frequency, fmt_length, length_unit

LOGGER_NAME = "wavesimfem"
logger = logging.getLogger(LOGGER_NAME)


class _StdoutHandler(logging.Handler):
    """Write plain messages to whatever ``sys.stdout`` is at emit time."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stream = sys.stdout
            stream.write(self.format(record) + "\n")
            stream.flush()
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


if not any(isinstance(h, _StdoutHandler) for h in logger.handlers):
    _handler = _StdoutHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def set_verbosity(level: int | str) -> None:
    """Set the package log level (e.g. ``"WARNING"`` for quiet runs)."""
    logger.setLevel(level.upper() if isinstance(level, str) else level)


def log(msg: str = "") -> None:
    """Emit an informational progress message."""
    logger.info(msg)


def warn(msg: str) -> None:
    """Emit a warning message."""
    logger.warning(f"  WARNING: {msg}")


def format_banner_box(lines: Sequence[str], min_width: int = 60) -> str:
    """Format lines inside a double-bordered Unicode box."""
    max_len = max((len(line) for line in lines), default=0)
    inner_width = max(max_len + 4, min_width)
    top = "╔" + "═" * inner_width + "╗"
    bot = "╚" + "═" * inner_width + "╝"
    body = [f"║{f'  {line}'.ljust(inner_width)}║" for line in lines]
    return "\n" + "\n".join([top, *body, bot])


def print_structure_table(summary: dict[str, Any], rank: int = 0, f0: float | None = None) -> None:
    """Print the normalized structure summary table (rank 0 only).

    Lengths are shown in a unit chosen from the thinnest region, so seismic
    (m) and ultrasonic (mm) models both read naturally.
    """
    if rank != 0:
        return

    regions = summary["all_regions"]
    th_unit = length_unit(min((r["thickness"] for r in regions), default=1.0))
    widths = [5, 27, 12, 15, 11, 11, 11, 14]
    w_no, w_name, w_th, w_mod, w_rho, w_cl, w_ct, w_role = widths
    top = "┌" + "┬".join("─" * c for c in widths) + "┐"
    mid = "├" + "┼".join("─" * c for c in widths) + "┤"
    bot = "└" + "┴".join("─" * c for c in widths) + "┘"
    log("\n" + top)
    log(
        f"│ {'No.':<{w_no - 2}} │ {'Region / Layer Name':<{w_name - 2}} │ "
        f"{f'Thick [{th_unit.label}]':<{w_th - 2}} │ {'Model':<{w_mod - 2}} │ "
        f"{'ρ [kg/m³]':<{w_rho - 2}} │ {'c_L [m/s]':<{w_cl - 2}} │ "
        f"{'c_T [m/s]':<{w_ct - 2}} │ {'Domain Role':<{w_role - 2}} │"
    )
    log(mid)
    for i, r in enumerate(regions):
        model_display = (
            "acoustic"
            if r.get("is_acoustic")
            else "legacy fluid"
            if r.get("is_legacy_fluid")
            else r["material_model"]
        )
        ct_display = (
            f"{'— (fluid)':<{w_ct - 2}}"
            if r.get("is_fluid", False)
            else f"{r['c_T']:<{w_ct - 2}.1f}"
        )
        log(
            f"│ {i + 1:<{w_no - 2}} │ {r['name'][: w_name - 2]:<{w_name - 2}} │ "
            f"{th_unit(r['thickness']):<{w_th - 2}.4g} │ "
            f"{model_display[: w_mod - 2]:<{w_mod - 2}} │ "
            f"{r['rho']:<{w_rho - 2}.1f} │ {r['c_L']:<{w_cl - 2}.1f} │ "
            f"{ct_display} │ {r['role'][: w_role - 2]:<{w_role - 2}} │"
        )
    log(bot)
    if summary.get("has_acoustic", False):
        log(
            "  Acoustic fluid: scalar pressure, zero shear stiffness, normal coupling with tangential slip."
        )
    if summary.get("has_legacy_fluid", False):
        warn(
            "Legacy regularized-elastic fluid approximation: artificial shear stiffness "
            "and tangential displacement continuity."
        )

    n_plate = len(summary["plate_layers"])
    mode_labels = {
        "1_layer": "1 layer (free plate in vacuum)",
        "N_layers": f"N layers ({n_plate} layers in vacuum)",
        "halfspace": "Halfspace (truncated semi-infinite medium, surface at y = 0)",
        "plate_on_halfspace": f"N layers over a halfspace ({n_plate} plate layers on substrate)",
        "plate_in_halfspaces": (
            f"N layers between 2 halfspaces ({n_plate} plate layers between "
            "substrate & superstrate)"
        ),
    }
    log(
        "  Configuration mode: "
        + mode_labels.get(summary["structure_mode"], summary["structure_mode"])
    )

    def _span(lo: float, hi: float) -> str:
        unit = length_unit(max(abs(lo), abs(hi)) or 1.0)
        return f"[{unit(lo):.4g}, {unit(hi):.4g}] {unit.label}"

    if summary["structure_mode"] == "halfspace":
        hs = summary["bottom_halfspace"]
        c_R = summary.get("c_R")
        h_sponge = hs["absorb_depth"]
        h_clean = summary["H_domain"] - h_sponge
        f0 = f0 or 1.0e6
        lambda_ref = (c_R or summary["c_wave_min"]) / f0
        is_fluid_hs = hs.get("is_fluid", False)
        wav_lbl = "λ_f" if is_fluid_hs else "λ_R"
        log(
            f"  Halfspace exterior surface: y = 0 (total depth {fmt_length(summary['H_domain'])}, "
            f"y ∈ {_span(summary['y_min'], 0.0)})"
        )
        if is_fluid_hs:
            log(f"  Fluid halfspace acoustic sound speed: c_f = {summary['c_L_min']:.1f} m/s")
            log(
                f"  Acoustic wavelength:    λ_f ≈ {fmt_length(lambda_ref)} (at f₀ = {fmt_frequency(f0)})"
            )
        else:
            c_r_msg = f", c_R = {c_R:.1f} m/s (Rayleigh surface wave)" if c_R else ""
            log(
                f"  Wave speeds: c_L = {summary['c_L_min']:.1f} m/s, "
                f"c_T = {summary['c_T_min']:.1f} m/s{c_r_msg}"
            )
            log(
                f"  Rayleigh wavelength:    λ_R ≈ {fmt_length(lambda_ref)} (at f₀ = {fmt_frequency(f0)})"
            )
        log(
            f"  Clean halfspace depth:  {fmt_length(h_clean)} ({h_clean / lambda_ref:.1f} {wav_lbl})"
        )
        log(
            f"  Bottom absorbing sponge: {fmt_length(h_sponge)} ({h_sponge / lambda_ref:.1f} {wav_lbl}), "
            f"y ∈ {_span(summary['y_min'], summary['y_min'] + h_sponge)}"
        )
        log("  Default acquisition reference: exterior surface y = 0\n")
        return

    log(
        f"  Plate thickness H_plate = {fmt_length(summary['H_plate'])} "
        f"(y ∈ {_span(0.0, summary['H_plate'])})"
    )
    log(
        f"  Computational domain y  = {_span(summary['y_min'], summary['y_max'])} "
        f"(total height {fmt_length(summary['H_domain'])})"
    )
    plate = summary["plate_layers"]
    fluid_layers = [lyr for lyr in plate if lyr.get("is_fluid", False)]
    if len(fluid_layers) < len(plate):
        log(
            f"  Plate wave speeds: c_L = [{summary['c_L_plate_min']:.1f}, "
            f"{summary['c_L_plate_max']:.1f}] m/s, c_T = [{summary['c_T_plate_min']:.1f}, "
            f"{summary['c_T_plate_max']:.1f}] m/s"
        )
    if fluid_layers:
        speeds = [r["c_L"] for r in fluid_layers]
        log(f"  Plate fluid acoustic speed: c_f = [{min(speeds):.1f}, {max(speeds):.1f}] m/s")

    def _hs_type(hs: dict[str, Any]) -> str:
        if hs.get("is_fluid", False):
            return f"fluid (c_f = {hs['c_L']:.1f} m/s)"
        return f"solid (c_L = {hs['c_L']:.1f}, c_T = {hs['c_T']:.1f} m/s)"

    if summary["has_bottom_halfspace"]:
        hs_b = summary["bottom_halfspace"]
        lo = summary["y_min"]
        log(
            f"  Bottom substrate:        {hs_b['name']} [{_hs_type(hs_b)}], sponge y ∈ "
            f"{_span(lo, lo + hs_b['absorb_depth'])} (depth {fmt_length(hs_b['absorb_depth'])})"
        )
    if summary["has_top_halfspace"]:
        hs_t = summary["top_halfspace"]
        hi = summary["y_max"]
        log(
            f"  Top superstrate:         {hs_t['name']} [{_hs_type(hs_t)}], sponge y ∈ "
            f"{_span(hi - hs_t['absorb_depth'], hi)} (depth {fmt_length(hs_t['absorb_depth'])})"
        )
    top_reg = summary["all_regions"][-1]
    if summary["has_top_halfspace"]:
        loc_str = f"plate-superstrate interface at y = {fmt_length(summary['H_plate'])}"
    else:
        top_type = "fluid" if top_reg.get("is_fluid", False) else "solid"
        loc_str = (
            f"top surface of '{top_reg['name']}' [{top_type}] at y = {fmt_length(summary['y_max'])}"
        )
    log(f"  Default acquisition reference: {loc_str}\n")
