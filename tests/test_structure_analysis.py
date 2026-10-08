"""Physical applicability checks for structure-aware report references."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from guidedwavefem.config import flatten_config
from guidedwavefem.materials import compute_rayleigh_speed
from guidedwavefem.structure_analysis import reference_dispersion, structure_profile

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXAMPLES = [
    ("01_plate_aluminum_lamb.yaml", "free_isotropic_plate"),
    ("02_layers_bonded_joint.yaml", "layered_plate"),
    ("03_halfspace_steel_rayleigh.yaml", "solid_halfspace"),
    ("04_plate_on_halfspace_soil_bedrock.yaml", "layered_halfspace"),
    ("05_layers_bone_qus.yaml", "solid_fluid_waveguide"),
    ("06_plate_in_halfspaces_immersed_water.yaml", "fluid_loaded_plate"),
]


def _profile(cfg, **metadata):
    return structure_profile({"simulation_config_yaml": yaml.safe_dump(cfg), **metadata})


def _isotropic():
    return dict(
        name="Solid", thickness=0.003, material_model="isotropic", E=70e9, nu=0.33, rho=2700
    )


@pytest.mark.parametrize("filename,family", EXAMPLES)
@pytest.mark.parametrize("flat", [False, True])
def test_shipped_structures_identified_from_full_archived_provenance(filename, family, flat):
    config = yaml.safe_load((CONFIGS / filename).read_text())
    profile = _profile(flatten_config(config) if flat else config)
    assert profile["family"] == family
    assert profile["provenance"] == "archived_config"
    assert profile["lamb_applicable"] == (family == "free_isotropic_plate")
    assert profile["regions"]
    assert not any("Love" in name or "SH" in name for name in profile["wave_families"])
    curves = reference_dispersion(profile, [0, 1e5, 2e5])
    assert {curve["label"] for curve in curves if curve["kind"] == "lamb"} == (
        {"A0", "S0"} if family == "free_isotropic_plate" else set()
    )
    json.dumps(profile, allow_nan=False)
    json.dumps(curves, allow_nan=False)


def test_free_plate_references_have_correct_low_frequency_limits():
    profile = _profile(dict(structure_mode="1_layer", single_layer=_isotropic()))
    curves = {c["label"]: c for c in reference_dispersion(profile, [1000, 2000])}
    plate = profile["homogeneous_plate"]
    c_l, c_t = plate["c_L_m_s"], plate["c_T_m_s"]
    extensional = 2 * c_t * np.sqrt(1 - (c_t / c_l) ** 2)
    np.testing.assert_allclose(curves["S0"]["phase_velocity_m_s"], extensional, rtol=1e-3)
    flexural = curves["A0"]["phase_velocity_m_s"]
    assert flexural[1] / flexural[0] == pytest.approx(np.sqrt(2), rel=0.01)
    for curve in curves.values():
        np.testing.assert_allclose(
            np.asarray(curve["wavenumber_rad_m"]) * curve["phase_velocity_m_s"],
            2 * np.pi * np.asarray(curve["frequency_hz"]),
        )


def test_halfspace_rayleigh_reference_is_subshear_and_not_lamb():
    profile = _profile(dict(structure_mode="halfspace", halfspace=_isotropic()))
    refs = {r["kind"]: r for r in profile["references"]}
    medium = profile["regions"][0]
    expected = compute_rayleigh_speed(medium["c_T_m_s"], c_L=medium["c_L_m_s"])
    assert refs["rayleigh"]["speed_m_s"] == pytest.approx(expected)
    assert refs["rayleigh"]["speed_m_s"] < medium["c_T_m_s"]
    assert not profile["lamb_applicable"]
    assert any("surface localization" in statement for statement in profile["interpretation"])


def test_layered_rayleigh_values_are_constituent_guides_only():
    config = yaml.safe_load((CONFIGS / EXAMPLES[3][0]).read_text())
    profile = _profile(config)
    refs = [r for r in profile["references"] if "rayleigh" in r["kind"]]
    assert len(refs) == 2
    assert all(r["kind"] == "constituent_rayleigh" for r in refs)
    assert all("not a layered-system branch" in r["applicability"] for r in refs)
    assert any("not" in s and "bounds" in s for s in profile["interpretation"])


def test_fluid_halfspace_never_uses_numerical_shear_as_a_physical_reference():
    water = dict(name="Water", thickness=0.03, material_model="fluid", rho=1000, c=1500)
    profile = _profile(dict(structure_mode="halfspace", halfspace=water))
    assert profile["family"] == "fluid_halfspace"
    assert profile["wave_families"] == ["Acoustic P waves"]
    assert len(profile["references"]) == 1
    assert profile["references"][0]["speed_m_s"] == pytest.approx(1500)
    assert profile["references"][0]["kind"] == "acoustic"
    assert profile["regions"][0]["numerical_shear_speed_m_s"] > 0
    assert not profile["lamb_applicable"]


def test_fluid_loaded_plate_guides_are_radiation_thresholds_not_free_lamb():
    config = yaml.safe_load((CONFIGS / EXAMPLES[5][0]).read_text())
    profile = _profile(config, sensor_y=config["structure"]["layers"][0]["thickness"])
    assert not profile["lamb_applicable"]
    assert len([r for r in profile["references"] if r["kind"] == "acoustic_radiation"]) == 2
    assert len(profile["sensor_regions"]) == 2  # interface belongs to both regions
    assert any("tangential continuity" in s for s in profile["exclusions"])


def test_anisotropic_axial_reference_uses_christoffel_eigenvalues():
    C = [25e9, 5e9, 2e9, 20e9, 1e9, 6e9]
    region = dict(name="Crystal", thickness=0.003, material_model="anisotropic", C=C, rho=2000)
    profile = _profile(dict(structure_mode="1_layer", single_layer=region))
    expected = np.sqrt(np.linalg.eigvalsh([[C[0], C[2]], [C[2], C[5]]]) / 2000)
    actual = sorted(r["speed_m_s"] for r in profile["references"])
    np.testing.assert_allclose(actual, expected)
    assert all(r["kind"] == "quasi_bulk" for r in profile["references"])
    assert profile["regions"][0]["c_L_m_s"] is None
    assert not profile["lamb_applicable"]


@pytest.mark.parametrize("halfspace", [False, True])
def test_viscoelastic_material_is_not_automatically_titled_anisotropic(halfspace):
    # Isotropic elastic tensor with proportional isotropic Kelvin-Voigt loss.
    region = dict(
        name="Lossy solid",
        thickness=0.01,
        material_model="viscoelastic",
        rho=2000,
        C=[20e9, 8e9, 0, 20e9, 0, 6e9],
        Cv=[20, 8, 0, 20, 0, 6],
    )
    cfg = (
        dict(structure_mode="halfspace", halfspace=region)
        if halfspace
        else dict(structure_mode="1_layer", single_layer=region)
    )
    profile = _profile(cfg)
    assert "Viscoelastic" in profile["title"]
    assert not profile["lamb_applicable"]
    assert all("elastic" in r["label"] for r in profile["references"])
    assert any("frequency-dependent" in s for s in profile["interpretation"])
    assert any("intrinsic" in s for s in profile["exclusions"])


def test_legacy_material_uncertainty_does_not_invent_anisotropy():
    profile = structure_profile(dict(structure_mode="halfspace", c_L=6000, c_T=3000))
    assert "anisotropic" not in profile["family"]
    assert profile["provenance"] == "legacy"
    assert not profile["lamb_applicable"]
    assert profile["references"] == []


def test_legacy_plate_cannot_authorize_free_plate_mode_curves():
    profile = structure_profile(
        dict(
            structure_mode="1_layer",
            material_model="isotropic",
            plate_thickness=0.003,
            c_L=6000,
            c_T=3000,
            rho=2700,
        )
    )
    assert not profile["lamb_applicable"]
    assert {r["kind"] for r in profile["references"]} == {"bulk"}
    assert not any(r["kind"] == "lamb" for r in reference_dispersion(profile, [1000]))


def test_legacy_multiple_layers_do_not_inherit_top_material_properties():
    profile = structure_profile(
        dict(
            structure_mode="N_layers",
            all_regions_names=["Bottom", "Top"],
            all_regions_models=["isotropic", "isotropic"],
            all_regions_roles=["Plate L1", "Plate L2"],
            all_regions_thicknesses=[0.002, 0.001],
            c_L=6000,
            c_T=3000,
            rho=2700,
            plate_thickness=0.003,
        )
    )
    assert profile["regions"][0]["c_L_m_s"] is None
    assert profile["regions"][1]["c_L_m_s"] == 6000
    assert {r["region"] for r in profile["references"]} == {"Top"}


@pytest.mark.parametrize(
    "raw", ["[not, a, mapping]", "structure: []", "structure: {mode: 1_layer}"]
)
def test_invalid_present_provenance_is_not_silently_replaced_with_legacy_guesses(raw):
    with pytest.raises(ValueError):
        structure_profile(
            dict(simulation_config_yaml=raw, material_model="isotropic", c_L=6000, c_T=3000)
        )


@pytest.mark.parametrize("frequencies", [[-1, 2], [0, np.nan], [[1, 2]]])
def test_invalid_reference_frequencies_fail_clearly(frequencies):
    profile = _profile(dict(structure_mode="1_layer", single_layer=_isotropic()))
    with pytest.raises(ValueError):
        reference_dispersion(profile, frequencies)
