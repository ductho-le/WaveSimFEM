"""Acoustic source conservation and incompatible boundary-data checks."""

from __future__ import annotations

from math import erf

import numpy as np
import pytest

from wavesimfem.acoustic_time import CoupledNewmark, CoupledRK4
from wavesimfem.acoustics import build_acoustic_problem
from wavesimfem.config import parse_and_resolve_config


def _source_config():
    return {
        "solver_type": "explicit",
        "device": "cpu",
        "CFL": 0.5,
        "duration": 6e-6,
        "sensing_length": 0.01,
        "absorb_length": 0.0,
        "absorb_alpha": 0.0,
        "absorb_beta": 0.0,
        "f_center": 1e5,
        "source_type": "tone_burst",
        "n_cycles": 2,
        "source_x": 0.004,
        "excitation_width": 0.006,
        "source_kind": "acoustic_volume",
        "source_amplitude": 1.5e6,
        "source_region": "water",
        "sensor_y": 0.0005,
        "sensor_offset": 0.001,
        "sensor_pitch": 0.001,
        "n_sensors": 3,
        "nx": 16,
        "ny_per_region": [2, 2],
        "degree": 1,
        "structure_mode": "N_layers",
        "layers": [
            {"name": "water", "thickness": 0.001, "rho": 1000.0, "c": 1500.0},
            {"name": "oil", "thickness": 0.001, "rho": 800.0, "c": 1200.0},
        ],
        "run_postprocess": False,
        "write_vtx": False,
    }


@pytest.mark.parametrize("method", [CoupledRK4, CoupledNewmark])
def test_volume_source_satisfies_closed_cavity_integrated_pressure_balance(method):
    """Integral(p/K)'' = Integral(q) across two fluids with unequal K and rho.

    The oracle integrates the specified Gaussian in space using erf and the
    two-cycle tone burst in time using its three sine harmonics. It does not
    reuse the production waveform or forcing vector as its reference.
    """
    cfg = parse_and_resolve_config(custom_cfg=_source_config())
    problem = build_acoustic_problem(cfg)
    end, steps = cfg["duration"], 240
    dt = end / steps
    stepper = (
        method(problem.operators, dt, forcing=problem.forcing)
        if method is CoupledNewmark
        else method(problem.operators, forcing=problem.forcing)
    )
    weighted = problem.operators.Mp.createVecLeft()
    try:
        for i in range(steps):
            stepper.step(i * dt, dt)
        # exp(-(x-c)^2 / (2 sigma^2)) = exp(-((x-c)/width)^2) with width = sqrt(2) sigma.
        width, center = np.sqrt(2.0) * cfg["excitation_width"] / 3.0, cfg["source_x"]
        spatial_integral = (
            0.001
            * width
            * np.sqrt(np.pi)
            / 2
            * (erf((0.01 - center) / width) - erf(-center / width))
        )
        omega = 2 * np.pi * cfg["f_center"]
        envelope_omega = omega / 2
        frequencies = np.array([omega, omega + envelope_omega, omega - envelope_omega])
        coefficients = np.array([0.5, -0.25, -0.25])
        first_integral = np.sum(coefficients * (1 - np.cos(frequencies * end)) / frequencies)
        second_integral = np.sum(
            coefficients * (end / frequencies - np.sin(frequencies * end) / frequencies**2)
        )
        tolerance = 8e-5 if method is CoupledNewmark else 2e-8
        for state, temporal_integral in (
            (stepper.p, second_integral),
            (stepper.pdot, first_integral),
        ):
            problem.operators.Mp.mult(state, weighted)
            expected = cfg["source_amplitude"] * spatial_integral * temporal_integral
            assert weighted.sum() == pytest.approx(expected, rel=tolerance, abs=0.0)
    finally:
        weighted.destroy()
        stepper.destroy()
        problem.destroy()


@pytest.mark.parametrize(
    "source_kind,boundary_kind,message",
    [
        ("acoustic_pressure", "pressure_release", "driven pressure face"),
        ("acoustic_pressure", "radiation", "driven pressure face"),
        ("acoustic_normal_acceleration", "radiation", "normal-acceleration face"),
        ("acoustic_normal_acceleration", "pressure_release", "normal-acceleration face"),
    ],
)
def test_acoustic_boundary_drive_rejects_incompatible_boundary_condition(
    source_kind, boundary_kind, message
):
    raw = _source_config()
    raw.update(
        source_kind=source_kind,
        source_region="oil",
        source_boundary="top",
        acoustic_boundaries={"top": boundary_kind},
    )
    cfg = parse_and_resolve_config(custom_cfg=raw)
    with pytest.raises(ValueError, match=message):
        build_acoustic_problem(cfg)


def test_driven_pressure_rejects_incompatible_release_data_at_shared_corner():
    raw = _source_config()
    raw.update(
        source_kind="acoustic_pressure",
        source_region="oil",
        source_boundary="top",
        acoustic_boundaries={"left": "pressure_release"},
    )
    cfg = parse_and_resolve_config(custom_cfg=raw)
    with pytest.raises(ValueError, match="Driven pressure intersects a pressure-release boundary"):
        build_acoustic_problem(cfg)
