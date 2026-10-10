"""Physical material, configuration and observable contracts for pressure acoustics."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from wavesimfem.config import dump_config, parse_and_resolve_config
from wavesimfem.materials import (
    compute_mesh_layout,
    normalize_structure,
    preflight_estimate,
    restricted_field_dof_counts,
    structure_from_config,
)
from wavesimfem.structure_analysis import structure_profile
from wavesimfem.wave_reports import write_structure_report

WATER = dict(name="Water", thickness=0.01, rho=1000, c=1500, material_model="fluid")
SOLID = dict(name="Solid", thickness=0.01, rho=2700, E=70e9, nu=0.33, material_model="isotropic")


def _config():
    return {
        "solver": {"type": "explicit", "CFL": 0.5, "duration": 2e-4},
        "geometry": {"sensing_length": 0.3, "absorb_length": 0.1},
        "structure": {"mode": "1_layer", "single_layer": dict(WATER)},
        "mesh": {"nx": 10, "ny_per_region": [2], "degree": 2},
        "excitation": {
            "f_center": 1e5,
            "source_kind": "acoustic_volume",
            "source_amplitude": 1e6,
            "source_region": "Water",
        },
        "sensors": {"observable": "pressure", "n_sensors": 3},
        "postprocess": {"run_postprocess": False},
    }


@pytest.mark.parametrize("properties", [{"c_sound": 1500}, {"K": 2.25e9}, {"E": 1.35e9, "nu": 0.4}])
def test_acoustic_material_has_only_physical_fluid_properties(properties):
    material = {k: v for k, v in WATER.items() if k != "c"}
    summary = normalize_structure("1_layer", single_layer_input=dict(material, **properties))
    fluid = summary["all_regions"][0]
    assert fluid["fluid_formulation"] == "acoustic"
    assert fluid["K"] == pytest.approx(2.25e9)
    assert fluid["c_fluid"] == fluid["c_L"] == fluid["c_physical_min"] == pytest.approx(1500)
    assert not {"E", "nu", "mu", "lmbda", "c_T", "fluid_shear_reg"}.intersection(fluid)
    assert summary["has_acoustic"] and not summary["has_legacy_fluid"]
    assert summary["c_T_min"] is None and summary["c_T_plate_top"] is None


def test_legacy_regularization_is_explicit_and_acoustic_reg_parameter_is_rejected():
    with pytest.raises(ValueError, match=r"remove it.*regularized_elastic"):
        normalize_structure("1_layer", single_layer_input=dict(WATER, fluid_shear_reg=1e-8))
    summary = normalize_structure(
        "1_layer", single_layer_input=dict(WATER, fluid_formulation="regularized_elastic")
    )
    fluid = summary["all_regions"][0]
    assert fluid["mu"] == pytest.approx(1e-8 * fluid["K"])
    assert fluid["c_T"] > 0 and fluid["is_legacy_fluid"]
    assert summary["has_legacy_fluid"] and not summary["has_acoustic"]


def test_acoustic_config_round_trip_preserves_source_units_and_boundaries(tmp_path):
    raw = _config()
    raw["excitation"].update(source_fx=0, source_fy=0)
    raw["acoustic_boundaries"] = {"top": "pressure_release", "right": "radiation"}
    cfg = parse_and_resolve_config(custom_cfg=raw)
    assert cfg["source_units"] == "s^-2"
    assert cfg["acoustic_boundaries"]["bottom"] == "rigid"
    assert cfg["acoustic_boundaries"]["top"] == "pressure_release"
    assert cfg["sensor_observable"] == "pressure"
    path = tmp_path / "config.yaml"
    dump_config(cfg, str(path))
    reloaded = parse_and_resolve_config(config_path=str(path))
    assert reloaded["source_units"] == cfg["source_units"]
    assert reloaded["acoustic_boundaries"] == cfg["acoustic_boundaries"]


@pytest.mark.parametrize("boundary,coordinate", [("left", 0.0), ("right", 0.5)])
@pytest.mark.parametrize("count", [None, 3])
def test_lateral_source_is_resolved_before_sensor_layout(boundary, coordinate, count):
    raw = _config()
    raw["excitation"].update(source_kind="acoustic_normal_acceleration", source_boundary=boundary)
    raw["sensors"]["n_sensors"] = count
    cfg = parse_and_resolve_config(custom_cfg=raw)
    assert cfg["source_x"] == coordinate
    first = cfg["source_x"] + cfg["sensor_offset"]
    last = first + (cfg["n_sensors"] - 1) * cfg["sensor_pitch"]
    assert 0 <= first < last <= 0.5
    if boundary == "right":
        assert last < cfg["source_x"]
    assert parse_and_resolve_config(custom_cfg=cfg) == cfg


@pytest.mark.parametrize("boundary", ["left", "right"])
def test_lateral_source_rejects_an_explicit_off_boundary_coordinate(boundary):
    raw = _config()
    raw["excitation"].update(
        source_kind="acoustic_normal_acceleration", source_boundary=boundary, source_x=0.15
    )
    with pytest.raises(ValueError, match=f"source_boundary: {boundary} requires source_x"):
        parse_and_resolve_config(custom_cfg=raw)


def test_acoustic_cavity_can_disable_lateral_sponges():
    raw = _config()
    raw["geometry"]["absorb_length"] = 0
    raw["absorbing_boundaries"] = {"alpha": 0, "beta": 0}
    cfg = parse_and_resolve_config(custom_cfg=raw)
    assert cfg["absorb_length"] == 0
    assert set(cfg["acoustic_boundaries"].values()) == {"rigid"}


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("excitation", "source_amplitude", None, "requires.*source_amplitude"),
        ("excitation", "source_amplitude", "1 Pa", "must be a number"),
        ("excitation", "source_region", None, "requires source_region"),
        ("excitation", "source_region", "Unknown", "name one material region"),
        ("excitation", "source_kind", "elastic_fluid", "source_kind"),
        ("sensors", "observable", "fluid_displacement", "sensor_observable"),
        ("acoustic_boundaries", "top", "traction_free", "boundary types"),
        ("acoustic_boundaries", "interface", "rigid", "Unknown acoustic boundary"),
    ],
)
def test_acoustic_config_rejects_ambiguous_or_unsupported_inputs(section, key, value, match):
    cfg = _config()
    cfg.setdefault(section, {})[key] = value
    with pytest.raises(ValueError, match=match):
        parse_and_resolve_config(custom_cfg=cfg)


def test_preflight_counts_restricted_fields_and_never_certifies_uncoupled_acoustic_dt():
    raw = _config()
    raw["structure"] = {
        "mode": "N_layers",
        "layers": [dict(WATER), dict(SOLID), dict(WATER, name="Water 2")],
    }
    raw["mesh"]["ny_per_region"] = [2, 3, 4]
    cfg = parse_and_resolve_config(custom_cfg=raw)
    summary = structure_from_config(cfg)
    layout = compute_mesh_layout(cfg, summary)
    counts = restricted_field_dof_counts(summary, layout, 2)
    # Each water component has its own boundary rows; the solid owns only its seven rows.
    assert counts == {"displacement": 2 * 21 * 7, "pressure": 21 * (5 + 9)}
    estimate = preflight_estimate(cfg, rank=1)
    assert estimate["ndofs_est"] == sum(counts.values())
    assert estimate["sensor_components_est"] == 1
    assert estimate["dt_element_limit"] is None
    assert estimate["dt_stability_status"] == "advisory_uncoupled_requires_assembled_check"


def test_old_archive_stays_legacy_and_new_acoustic_archive_has_no_regularization_warning():
    config = {"structure_mode": "1_layer", "single_layer": dict(WATER)}
    data = {"simulation_config_yaml": yaml.safe_dump(config)}
    old = structure_profile(data)
    new = structure_profile(
        dict(data, schema_version=2, sensor_observable="pressure", sensor_units="Pa")
    )
    assert old["has_legacy_fluid"] and not old["has_acoustic"]
    assert old["regions"][0]["numerical_shear_speed_m_s"] > 0
    assert new["has_acoustic"] and not new["has_legacy_fluid"]
    assert "numerical_shear_speed_m_s" not in new["regions"][0]
    assert not any("Regularized" in text for text in new["exclusions"])


@pytest.mark.parametrize("observable", ["pressure", "velocity", "particle_velocity"])
def test_reports_use_selected_observable_units_and_withhold_scalar_polarization(
    tmp_path, monkeypatch, observable
):
    captured = []
    monkeypatch.setattr(
        "wavesimfem.wave_reports._save", lambda fig, *args: captured.append(fig) or []
    )
    t = np.arange(512) * 1e-7
    x = 0.02 + np.arange(21) * 1e-3
    phase = 2 * np.pi * 1e5 * (t[:, None] - x[None, :] / 1500)
    data = dict(
        schema_version=2,
        times=t,
        sensor_x=x,
        sensor_y=0.005,
        source_x=0.0,
        f_center=1e5,
        disp_f_max=1.5e5,
        c_fluid=1500,
        c_L=1500,
        sensor_observable=observable,
        sensor_units="Pa" if observable == "pressure" else "m/s",
        simulation_config_yaml=yaml.safe_dump(
            {"structure_mode": "1_layer", "single_layer": dict(WATER)}
        ),
    )
    if observable == "pressure":
        data["sensor_data_pressure"] = 100 * np.sin(phase)
    else:
        data["sensor_data_vx"], data["sensor_data_vy"] = np.cos(phase), np.sin(phase)
    archive = tmp_path / "sensors.npz"
    np.savez(archive, **data)
    report = write_structure_report(archive, formats=("png",), usetex=False)
    assert report["summary"]["sensor_observable"] == observable
    assert report["summary"]["sensor_units"] == data["sensor_units"]
    if observable == "pressure":
        assert report["summary"]["selected_component"] == "pressure"
        assert report["summary"]["coherent_polarization_samples"] == 0
        assert "scalar observable" in report["polarization_method"]["reason"]
        assert any("Pressure" in ax.get_title() for ax in captured[1].axes)
        assert not any("Horizontal" in ax.get_title() for ax in captured[1].axes)
    readme = (tmp_path / "wave_physics" / "README.md").read_text()
    assert f"`{observable}` [{data['sensor_units']}]" in readme
    with (tmp_path / "wave_physics" / "report.json").open() as stream:
        assert json.load(stream)["summary"]["sensor_units"] == data["sensor_units"]
    for figure in captured:
        figure.clear()
