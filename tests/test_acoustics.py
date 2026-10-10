"""Acoustic verification against independent continuum reference solutions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
_spec = importlib.util.spec_from_file_location(
    "acoustic_reference", TOOLS / "acoustic_reference.py"
)
reference = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = reference
_spec.loader.exec_module(reference)
_spec = importlib.util.spec_from_file_location("verify_acoustics", TOOLS / "verify_acoustics.py")
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)


def test_normal_reference_reduces_to_impedance_theory():
    result = reference.fluid_solid_coefficients(1000, 1500, 2000, 2500, 1200, 0)
    reflection, _, _, _ = reference.fluid_impedance_coefficients(1000, 1500, 2000, 2500)
    assert result["reflection"] == pytest.approx(reflection, abs=1e-14)
    assert result["velocity_s"] == 0
    assert result["velocity_p"] == pytest.approx(2 / (1000 * 1500 + 2000 * 2500))


@pytest.mark.parametrize("angle", [5, 20, 35, 50])
def test_oblique_reference_has_slip_zero_shear_and_energy_balance(angle):
    result = reference.fluid_solid_coefficients(1000, 1500, 2000, 2500, 1200, np.deg2rad(angle))
    assert abs(result["normal_velocity_residual"]) < 1e-20
    assert abs(result["normal_traction_residual"]) < 1e-12
    assert abs(result["tangential_traction"]) < 1e-12
    assert abs(result["tangential_slip"]) > 1e-8
    assert result["reflected_energy"] + result["p_energy"] + result["s_energy"] == pytest.approx(
        1, abs=1e-12
    )


def test_scholte_and_plate_references_have_physical_limits():
    c = reference.scholte_speed(1000, 1500, 2000, 2500, 1200)
    assert c == pytest.approx(1009.0411173762864, abs=1e-7)
    assert 0 < c < min(1500, 1200)
    # A0 tends to zero as k*h tends to zero, and speeds up with k*h.
    speeds = [
        reference.fluid_loaded_plate_speed(k, 0.2, 1000, 1500, 2000, 2500, 1200) for k in (1, 2, 4)
    ]
    assert 0 < speeds[0] < speeds[1] < speeds[2] < c


def test_rigid_cavity_has_one_pressure_field_and_constant_null_mode():
    problem = verify.build(verify.benchmark_config([verify.fluid()], 3, [3], degree=1))
    try:
        operators = verify.operators(problem)
        assert operators["Mp"].shape == (16, 16)
        assert operators["Ms"] is None or operators["Ms"].shape == (0, 0)
        assert len(problem.u_coords) == 0
        assert np.linalg.norm(operators["Kp"] @ np.ones(16)) < 1e-13
        assert np.all(operators["Mp"].diagonal() > 0)
    finally:
        problem.destroy()


@pytest.mark.slow
def test_cavity_spatial_convergence():
    assert verify.cavity_check()["status"] == "pass"


@pytest.mark.slow
def test_cavity_temporal_convergence():
    assert verify.temporal_check()["status"] == "pass"


@pytest.mark.slow
def test_pressure_release_and_normal_radiation():
    assert verify.boundary_check()["status"] == "pass"


@pytest.mark.slow
def test_density_contrast_transmission():
    assert verify.fluid_fluid_check()["status"] == "pass"


@pytest.mark.slow
def test_normal_fluid_solid_matches_reference_and_legacy():
    assert verify.scattering_check()["status"] == "pass"


@pytest.mark.slow
def test_oblique_fluid_solid_matches_independent_mode_conversion():
    assert verify.scattering_check(np.deg2rad(20))["status"] == "pass"


@pytest.mark.slow
def test_scholte_speed_and_localization_converge():
    assert verify.scholte_check()["status"] == "pass"


@pytest.mark.slow
def test_fluid_loaded_plate_dispersion():
    assert verify.plate_check()["status"] == "pass"
