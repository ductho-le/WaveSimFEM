"""Configuration schema, resolution, and structure-validation tests."""

from __future__ import annotations

import argparse
import copy
import math

import numpy as np
import pytest

from wavesimfem.config import (
    ABSORB_ALPHA_REF,
    ABSORB_BETA_REF,
    LEGACY_DISPERSION_KEYS,
    MAX_AUTO_SENSORS,
    dump_config,
    flatten_config,
    load_config,
    parse_and_resolve_config,
)
from wavesimfem.materials import sizing_wavelength, structure_from_config
from wavesimfem.sources import DEFAULT_N_CYCLES

ALUMINUM = {"thickness": 1e-3, "rho": 2700.0, "E": 70e9, "nu": 0.33, "material_model": "isotropic"}


def _base(**overrides):
    cfg = {
        "solver": {"type": "explicit", "CFL": 0.8},
        "mesh": {"nx": 3213, "ny_per_region": [7]},
        "geometry": {"sensing_length": 0.3, "absorb_length": 0.1},
        "excitation": {"f_center": 1e6},
        "structure": {"mode": "1_layer", "single_layer": dict(ALUMINUM)},
    }
    for dotted, value in overrides.items():
        section, key = dotted.split("__")
        cfg.setdefault(section, {})[key] = value
    return cfg


def test_valid_base_resolves_with_defaults():
    cfg = parse_and_resolve_config(custom_cfg=_base())
    assert cfg["device"] == "auto"
    assert cfg["precision"] == "double"
    assert (cfg["nx"], cfg["ny_per_region"], cfg["CFL"]) == (3213, [7], 0.8)
    summary = structure_from_config(cfg)
    wavelength = sizing_wavelength(cfg, summary, 1e6)
    assert cfg["excitation_width"] == pytest.approx(wavelength / 2)
    assert cfg["source_x"] == pytest.approx(0.1 + wavelength)
    assert cfg["sensor_pitch"] == pytest.approx(
        min(wavelength / 8, sizing_wavelength(cfg, summary, 3e6) / 4)
    )
    assert cfg["steps_per_period"] == 40 and cfg["sensor_sample_rate"] is None
    assert cfg["absorb_alpha"] == pytest.approx(ABSORB_ALPHA_REF)  # f0 = 1 MHz reference
    assert cfg["absorb_beta"] == pytest.approx(ABSORB_BETA_REF)
    assert cfg["source_type"] == "ricker" and cfg["n_cycles"] is None
    assert cfg["figure_formats"] == ["png", "pdf", "svg", "eps"]
    assert cfg["usetex"] is False


@pytest.mark.parametrize("value,expected", [(False, False), (True, True), ("AUTO", "auto")])
def test_latex_setting_is_preserved(value, expected):
    cfg = parse_and_resolve_config(custom_cfg=_base(output__usetex=value))
    assert cfg["usetex"] == expected


def test_auto_layout_scales_with_wavelength():
    """Scaling every length by s and f0 by 1/s scales the automatic layout by s."""
    keys = ("excitation_width", "source_x", "sensor_offset", "sensor_pitch")
    small = parse_and_resolve_config(custom_cfg=_base())
    scale = 1000.0
    big = _base(excitation__f_center=1e6 / scale)
    big["geometry"] = {"sensing_length": 0.3 * scale, "absorb_length": 0.1 * scale}
    big["structure"]["single_layer"]["thickness"] = 1e-3 * scale
    big = parse_and_resolve_config(custom_cfg=big)
    for key in keys:
        assert big[key] == pytest.approx(scale * small[key]), key
    assert big["n_sensors"] == small["n_sensors"]


def test_auto_layout_fits_short_domains():
    """A wavelength much longer than the domain still yields a valid sensor array."""
    cfg = parse_and_resolve_config(custom_cfg=_base(excitation__f_center=15.0))
    sensing, absorb = 0.3, 0.1
    assert cfg["excitation_width"] == pytest.approx(sensing / 10)
    assert cfg["source_x"] == pytest.approx(absorb + sensing / 8)
    assert cfg["sensor_pitch"] == pytest.approx(sensing / 100)
    last = cfg["source_x"] + cfg["sensor_offset"] + (cfg["n_sensors"] - 1) * cfg["sensor_pitch"]
    assert last <= absorb + sensing + 1e-12


def test_auto_sensor_count_is_capped():
    cfg = _base()
    cfg["geometry"] = {"sensing_length": 10.0, "absorb_length": 0.1}
    assert parse_and_resolve_config(custom_cfg=cfg)["n_sensors"] == MAX_AUTO_SENSORS


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("solver__steps_per_period", 2, "steps_per_period"),
        ("sensors__sample_rate", "-1 MHz", "sample_rate"),
        ("sensors__sample_rate", "fast", "sample_rate"),
    ],
)
def test_time_sampling_options_are_validated(key, value, match):
    with pytest.raises(ValueError, match=match):
        parse_and_resolve_config(custom_cfg=_base(**{key: value}))


def test_sample_rate_accepts_units():
    cfg = parse_and_resolve_config(custom_cfg=_base(sensors__sample_rate="20 MHz"))
    assert cfg["sensor_sample_rate"] == pytest.approx(20e6)


@pytest.mark.parametrize("f0", [15.0, 50e3, 5e6])
def test_sponge_damping_scales_with_frequency(f0):
    cfg = parse_and_resolve_config(custom_cfg=_base(excitation__f_center=f0))
    assert cfg["absorb_alpha"] / f0 == pytest.approx(ABSORB_ALPHA_REF / 1e6)
    assert cfg["absorb_beta"] * f0 == pytest.approx(ABSORB_BETA_REF * 1e6)


def test_explicit_sponge_damping_is_kept():
    cfg = parse_and_resolve_config(
        custom_cfg=_base(absorbing_boundaries__alpha=3.0, absorbing_boundaries__beta=0.0)
    )
    assert (cfg["absorb_alpha"], cfg["absorb_beta"]) == (3.0, 0.0)


def test_default_source_clears_sponge_by_excitation_width():
    cfg = _base(excitation__excitation_width=3.0)
    cfg["geometry"] = {"sensing_length": 100.0, "absorb_length": 60.0}
    cfg["sensors"] = {"sensor_offset": 10.0, "sensor_pitch": 1.0, "n_sensors": 50}
    assert parse_and_resolve_config(custom_cfg=cfg)["source_x"] == pytest.approx(63.0)


def test_hierarchical_device_and_precision_are_honoured():
    cfg = parse_and_resolve_config(
        custom_cfg=_base(solver__device="CUDA", solver__precision="Single")
    )
    assert cfg["device"] == "cuda:0"
    assert cfg["precision"] == "single"


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("solver__device", "gpu", "Invalid device"),
        ("solver__precision", "half", "precision"),
        ("solver__type", "leapfrog", "solver_type"),
        ("solver__CFL", -1, "CFL"),
        ("solver__CFL", 1.5, "CFL"),
        ("excitation__f_center", 0, "f_center"),
        ("excitation__f_center", "fast", "f_center"),
        ("sensors__sensor_pitch", 0, "sensor_pitch"),
        ("sensors__n_sensors", 1, "n_sensors"),
        ("sensors__n_sensors", 3.5, "n_sensors"),
        ("sensors__sensor_y", 0.5, "sensor_y"),
        ("mesh__degree", 0, "degree"),
        ("mesh__degree", True, "degree"),
        ("excitation__source_x", 10.0, "source_x"),
        ("postprocess__f_max_disp_MHz", -1.0, "postprocess.f_max"),
        ("postprocess__f_max", "-1 MHz", "nonnegative"),
        ("postprocess__f_max", "2 parsecs", "unknown frequency unit"),
        ("postprocess__k_max", "3 MHz", "unknown wavenumber unit"),
        ("postprocess__k_max", True, "boolean"),
        ("excitation__source_type", "chirp", "source_type"),
        ("excitation__n_cycles", 5, "only applies"),
        ("output__figure_formats", ["png", "tiff"], "figure format"),
        ("output__usetex", "maybe", "usetex"),
    ],
)
def test_invalid_values_are_rejected(key, value, match):
    with pytest.raises(ValueError, match=match):
        parse_and_resolve_config(custom_cfg=_base(**{key: value}))


@pytest.mark.parametrize("key", ["nx", "ny_per_region"])
@pytest.mark.parametrize("missing", [False, True])
def test_mesh_counts_are_required(key, missing):
    cfg = _base()
    if missing:
        del cfg["mesh"][key]
    else:
        cfg["mesh"][key] = None
    with pytest.raises(ValueError, match=rf"must specify 'mesh\.{key}'"):
        parse_and_resolve_config(custom_cfg=cfg)


@pytest.mark.parametrize("value", [0, -1, 3.5, True, "4", float("nan"), float("inf")])
def test_longitudinal_count_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="nx"):
        parse_and_resolve_config(custom_cfg=_base(mesh__nx=value))


@pytest.mark.parametrize(
    "value",
    [
        [],
        4,
        "4",
        (4,),
        {"plate": 4},
        [0],
        [-1],
        [True],
        [1.5],
        ["4"],
        [float("nan")],
        [float("inf")],
    ],
)
def test_region_counts_must_be_a_list_of_positive_integers(value):
    with pytest.raises(ValueError, match="ny_per_region"):
        parse_and_resolve_config(custom_cfg=_base(mesh__ny_per_region=value))


def test_integer_counts_are_normalized_without_mutating_input():
    cfg = _base(mesh__nx=3213.0, mesh__ny_per_region=[7.0])
    resolved = parse_and_resolve_config(custom_cfg=cfg)
    assert type(resolved["nx"]) is int
    assert type(resolved["ny_per_region"][0]) is int
    assert type(cfg["mesh"]["ny_per_region"][0]) is float


@pytest.mark.parametrize("counts", [[4], [4, 5, 6]])
def test_region_counts_must_match_the_normalized_structure(counts):
    cfg = _base(mesh__ny_per_region=counts)
    cfg["structure"] = {"mode": "N_layers", "layers": [dict(ALUMINUM), dict(ALUMINUM)]}
    with pytest.raises(ValueError, match="ny_per_region"):
        parse_and_resolve_config(custom_cfg=cfg)


@pytest.mark.parametrize("key", ["elems_per_wavelength", "min_elems_x", "min_elems_y"])
@pytest.mark.parametrize("sectioned", [False, True])
def test_removed_automatic_mesh_keys_explain_migration(key, sectioned):
    cfg = _base()
    (cfg["mesh"] if sectioned else cfg)[key] = 8
    with pytest.raises(ValueError, match=r"no longer supported.*mesh\.nx.*mesh\.ny_per_region"):
        parse_and_resolve_config(custom_cfg=cfg)


def test_removed_regional_minimum_explains_migration():
    cfg = _base()
    cfg["structure"]["single_layer"]["min_elems_y"] = 4
    with pytest.raises(ValueError, match=r"min_elems_y.*ny_per_region"):
        parse_and_resolve_config(custom_cfg=cfg)


@pytest.mark.parametrize("dt", [None, 1e-9])
@pytest.mark.parametrize("missing", [False, True])
def test_explicit_cfl_is_required_even_with_a_dt_override(dt, missing):
    cfg = _base(solver__dt=dt, solver__CFL=None)
    if missing:
        del cfg["solver"]["CFL"]
    with pytest.raises(ValueError, match=r"must specify 'solver\.CFL'"):
        parse_and_resolve_config(custom_cfg=cfg)


def test_explicit_cfl_can_be_supplied_as_a_cli_override():
    cfg = _base()
    del cfg["solver"]["CFL"]
    resolved = parse_and_resolve_config(custom_cfg=cfg, args=argparse.Namespace(cfl=0.5))
    assert resolved["CFL"] == 0.5


def test_implicit_cfl_is_optional():
    cfg = _base(solver__type="implicit")
    del cfg["solver"]["CFL"]
    assert parse_and_resolve_config(custom_cfg=cfg)["CFL"] is None


@pytest.mark.parametrize("value", [0, -1, 1.1, True, float("nan"), float("inf")])
def test_implicit_cfl_is_validated_when_supplied(value):
    with pytest.raises(ValueError, match="CFL"):
        parse_and_resolve_config(custom_cfg=_base(solver__type="implicit", solver__CFL=value))


@pytest.mark.parametrize("n_cycles", [0, 2.5, True, "five"])
def test_tone_burst_cycles_are_strict(n_cycles):
    with pytest.raises(ValueError, match="n_cycles"):
        parse_and_resolve_config(
            custom_cfg=_base(excitation__source_type="tone_burst", excitation__n_cycles=n_cycles)
        )


def test_tone_burst_defaults_and_normalisation():
    cfg = parse_and_resolve_config(custom_cfg=_base(excitation__source_type="Tone-Burst"))
    assert cfg["source_type"] == "tone_burst"
    assert cfg["n_cycles"] == DEFAULT_N_CYCLES
    cfg = parse_and_resolve_config(
        custom_cfg=_base(excitation__source_type="tone_burst", excitation__n_cycles=3.0)
    )
    assert cfg["n_cycles"] == 3 and isinstance(cfg["n_cycles"], int)


def test_dispersion_limits_accept_si_numbers_and_unit_strings():
    cfg = parse_and_resolve_config(
        custom_cfg=_base(
            postprocess__f_min="100 kHz",
            postprocess__f_max=2e6,
            postprocess__k_min=0,
            postprocess__k_max="3 rad/mm",
        )
    )
    assert cfg["disp_f_min"] == pytest.approx(1e5)
    assert cfg["disp_f_max"] == pytest.approx(2e6)
    assert cfg["disp_k_min"] == 0.0
    assert cfg["disp_k_max"] == pytest.approx(3e3)


def test_legacy_dispersion_keys_are_converted():
    cfg = parse_and_resolve_config(
        custom_cfg=_base(postprocess__f_max_disp_MHz=1.5, postprocess__k_max_disp=2.0)
    )
    assert cfg["disp_f_max"] == pytest.approx(1.5e6)
    assert cfg["disp_k_max"] == pytest.approx(2e3)
    assert not set(LEGACY_DISPERSION_KEYS) & set(cfg)


def test_legacy_and_new_dispersion_keys_conflict():
    with pytest.raises(ValueError, match="only one of"):
        parse_and_resolve_config(
            custom_cfg=_base(postprocess__f_max="1 MHz", postprocess__f_max_disp_MHz=1.0)
        )


def test_dispersion_limits_must_be_ordered():
    with pytest.raises(ValueError, match=r"postprocess\.f_min < postprocess\.f_max"):
        parse_and_resolve_config(
            custom_cfg=_base(postprocess__f_min="2 MHz", postprocess__f_max="1 MHz")
        )


def test_automatic_dispersion_limits():
    cfg = parse_and_resolve_config(custom_cfg=_base(sensors__sensor_pitch=1e-3))
    c_t = (ALUMINUM["E"] / (2 * (1 + ALUMINUM["nu"])) / ALUMINUM["rho"]) ** 0.5
    assert cfg["disp_f_min"] == 0.0 and cfg["disp_f_max"] == pytest.approx(2e6)
    assert cfg["disp_k_min"] == 0.0
    assert cfg["disp_k_max"] == pytest.approx(min(2 * math.pi * 2e6 / c_t, math.pi / 1e-3))


def test_cli_dispersion_override_beats_legacy_file_key():
    args = argparse.Namespace(f_max="750 kHz")
    cfg = parse_and_resolve_config(custom_cfg=_base(postprocess__f_max_disp_MHz=1.0), args=args)
    assert cfg["disp_f_max"] == pytest.approx(7.5e5)


@pytest.mark.parametrize("value", ["false", "yes", 0, 1])
def test_booleans_are_strict(value):
    with pytest.raises(ValueError, match="true or false"):
        parse_and_resolve_config(custom_cfg=_base(output__write_vtx=value))


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_structure_report_switch_is_strict(value):
    with pytest.raises(ValueError, match=r"structure_analysis.*true or false"):
        parse_and_resolve_config(custom_cfg=_base(postprocess__structure_analysis=value))


def test_structure_report_is_default_and_can_be_disabled():
    assert parse_and_resolve_config(custom_cfg=_base())["structure_analysis"] is True
    cfg = parse_and_resolve_config(custom_cfg=_base(postprocess__structure_analysis=False))
    assert cfg["structure_analysis"] is False


def test_unknown_section_key_suggests_correction():
    with pytest.raises(ValueError, match=r"Unknown key 'solver\.devcie'.*'device'"):
        flatten_config(_base(solver__devcie="cpu"))


def test_unknown_top_level_key_suggests_correction():
    with pytest.raises(ValueError, match=r"Unknown configuration key 'f_centre'.*'f_center'"):
        flatten_config({"f_centre": 1e6})


def test_duplicate_definition_is_rejected():
    cfg = _base()
    cfg["f_center"] = 2e6
    with pytest.raises(ValueError, match="defined twice"):
        flatten_config(cfg)


def test_section_must_be_mapping():
    with pytest.raises(ValueError, match="must be a mapping"):
        flatten_config({"solver": "explicit"})


def test_cli_overrides_take_precedence():
    args = argparse.Namespace(solver="implicit", device="cpu", cfl=0.25, no_postprocess=True)
    cfg = parse_and_resolve_config(custom_cfg=_base(), args=args)
    assert cfg["solver_type"] == "implicit"
    assert cfg["device"] == "cpu"
    assert cfg["CFL"] == 0.25
    assert cfg["run_postprocess"] is False


def test_halfspace_aliases():
    cfg = _base()
    cfg["mesh"]["ny_per_region"] = [8, 7]
    cfg["structure"] = {
        "mode": "plate_on_halfspace",
        "single_layer": dict(ALUMINUM),
        "substrate": {"thickness": 2e-3, "rho": 1000.0, "c": 1480.0, "material_model": "fluid"},
    }
    resolved = parse_and_resolve_config(custom_cfg=cfg)
    assert resolved["bottom_halfspace"]["c"] == 1480.0
    assert "substrate" not in resolved

    cfg["structure"]["bottom_halfspace"] = cfg["structure"]["substrate"]
    with pytest.raises(ValueError, match="only one of"):
        parse_and_resolve_config(custom_cfg=cfg)


def test_resolved_config_round_trips(tmp_path):
    """A dumped (provenance-carrying) configuration reloads and re-resolves."""
    cfg = parse_and_resolve_config(custom_cfg=_base())
    cfg["runtime"] = {"dt": 1e-9}
    cfg["load_y"] = 1e-3
    path = tmp_path / "simulation_config.yaml"
    dump_config(cfg, str(path))
    again = parse_and_resolve_config(str(path))
    assert again["runtime"]["dt"] == 1e-9
    for key in ("f_center", "device", "precision", "source_x", "single_layer"):
        assert again[key] == cfg[key]


def test_resolution_is_idempotent():
    once = parse_and_resolve_config(custom_cfg=_base())
    twice = parse_and_resolve_config(custom_cfg=copy.deepcopy(once))
    assert twice == once


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_dump_load_formats(tmp_path, suffix):
    path = tmp_path / f"cfg{suffix}"
    dump_config({"a": 1.5, "b": [1, 2], "c": True, "_internal": "dropped"}, str(path))
    assert load_config(str(path)) == {"a": 1.5, "b": [1, 2], "c": True}


def test_yaml_json_toml_resolve_identically(tmp_path):
    toml_text = """
[solver]
type = "explicit"
CFL = 0.8

[mesh]
nx = 3213
ny_per_region = [7]

[geometry]
sensing_length = 0.3
absorb_length = 0.1

[excitation]
source_type = "tone_burst"
n_cycles = 3
f_center = 1e6

[structure]
mode = "1_layer"

[structure.single_layer]
thickness = 1e-3
rho = 2700.0
E = 70e9
nu = 0.33
material_model = "isotropic"

[postprocess]
f_max = "2 MHz"
"""
    (tmp_path / "cfg.toml").write_text(toml_text, encoding="utf-8")
    base = _base(excitation__source_type="tone_burst", excitation__n_cycles=3)
    base["postprocess"] = {"f_max": "2 MHz"}
    dump_config(base, str(tmp_path / "cfg.yaml"))
    dump_config(base, str(tmp_path / "cfg.json"))
    resolved = [
        {
            k: v
            for k, v in parse_and_resolve_config(str(tmp_path / f"cfg{ext}")).items()
            if k[0] != "_"
        }
        for ext in (".toml", ".yaml", ".json")
    ]
    assert resolved[0] == resolved[1] == resolved[2]
    assert resolved[0]["n_cycles"] == 3 and resolved[0]["disp_f_max"] == pytest.approx(2e6)


def test_load_config_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "missing.yaml"))
    bad = tmp_path / "cfg.ini"
    bad.write_text("x=1")
    with pytest.raises(ValueError, match="Unsupported"):
        load_config(str(bad))
    scalar = tmp_path / "cfg.yaml"
    scalar.write_text("42\n")
    with pytest.raises(TypeError, match="mapping"):
        load_config(str(scalar))


# ---------------------------------------------------------------------------
# Structure / material regions
# ---------------------------------------------------------------------------
def _structure(**kwargs):
    return structure_from_config({"structure_mode": kwargs.pop("mode", "1_layer"), **kwargs})


def test_unknown_region_field_suggests_correction():
    layer = dict(ALUMINUM, thicknes=1e-3)
    del layer["thickness"]
    with pytest.raises(ValueError, match=r"thicknes.*'thickness'"):
        _structure(single_layer=layer)


def test_missing_region_field_is_reported():
    layer = dict(ALUMINUM)
    del layer["rho"]
    with pytest.raises(ValueError, match="rho"):
        _structure(single_layer=layer)


def test_one_layer_mode_rejects_multiple_layers():
    with pytest.raises(ValueError, match="1_layer"):
        _structure(layers=[dict(ALUMINUM), dict(ALUMINUM)])


def test_viscous_tensor_requires_viscoelastic_model():
    layer = dict(ALUMINUM, Cv=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="viscoelastic"):
        _structure(single_layer=layer)


def test_absorb_depth_only_on_halfspaces():
    with pytest.raises(ValueError, match="absorb_depth"):
        _structure(single_layer=dict(ALUMINUM, absorb_depth=1e-4))


def test_material_model_is_inferred():
    layer = dict(ALUMINUM)
    del layer["material_model"]
    summary = _structure(single_layer=layer)
    assert summary["plate_layers"][0]["material_model"] == "isotropic"


def test_unknown_structure_mode():
    with pytest.raises(ValueError, match="mode"):
        _structure(mode="sandwich", single_layer=dict(ALUMINUM))


def test_anisotropic_layer_bounds_match_isotropic_equivalent():
    """An isotropic tensor entered as ``anisotropic`` gives the exact shear speed and a
    rigorous upper bound for the fastest speed (Kelvin eigenvalue bounds)."""
    E, nu, rho = ALUMINUM["E"], ALUMINUM["nu"], ALUMINUM["rho"]
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    tensor = [lam + 2 * mu, lam, 0.0, lam + 2 * mu, 0.0, mu]
    cfg = _base()
    cfg["structure"]["single_layer"] = {
        "thickness": 1e-3,
        "rho": rho,
        "material_model": "anisotropic",
        "C": tensor,
    }
    aniso = structure_from_config(parse_and_resolve_config(custom_cfg=cfg))["all_regions"][0]
    iso = structure_from_config(parse_and_resolve_config(custom_cfg=_base()))["all_regions"][0]
    assert aniso["c_T"] == pytest.approx(iso["c_T"], rel=1e-12)
    assert aniso["c_L"] >= iso["c_L"]
    assert np.allclose(aniso["C_mat"], aniso["C_mat"].T)


def test_readme_yaml_example_is_valid_config():
    """The 'Configuration at a Glance' YAML snippet in README.md must parse and validate."""
    import re
    from pathlib import Path

    import yaml

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```yaml\n(.*?)```", readme, flags=re.DOTALL)
    assert blocks, "Expected at least one ```yaml block in README.md"
    for block in blocks:
        resolved = parse_and_resolve_config(custom_cfg=yaml.safe_load(block))
        assert resolved["structure_mode"] == "1_layer"
        assert resolved["n_sensors"] == 64


def test_sensor_placement_and_observable_compatibility_checked_at_config_time():
    """Incompatible sensor_region, sensor_y, and sensor_observable fail during config validation."""
    from wavesimfem.materials import preflight_estimate

    water = {"name": "water", "thickness": 1e-3, "rho": 1000.0, "c_sound": 1500.0}
    solid = dict(ALUMINUM, name="aluminum", thickness=1e-3)

    # Solid-solid stack: default sensor_y is top (aluminum), so sensor_region='bottom' cannot match
    two_solids = _base()
    two_solids["structure"] = {
        "mode": "N_layers",
        "layers": [dict(ALUMINUM, name="bottom", thickness=1e-3), solid],
    }
    two_solids["mesh"]["ny_per_region"] = [2, 2]
    two_solids.setdefault("sensors", {})["region"] = "bottom"
    with pytest.raises(ValueError, match="sensor_region 'bottom' does not contain or touch"):
        parse_and_resolve_config(custom_cfg=two_solids)

    # Even on an unvalidated raw dict, preflight_estimate must not misclassify empty candidates as pressure
    unvalidated = parse_and_resolve_config(custom_cfg=two_solids, validate_simulation=False)
    unvalidated.update(duration=1e-5, max_frequency=3e5, source_x=0.0, n_sensors=8)
    assert preflight_estimate(unvalidated, rank=1)["sensor_components_est"] == 2

    # Coupled water + solid: displacement inside water-only height fails immediately
    coupled = _base()
    coupled["structure"] = {"mode": "N_layers", "layers": [water, solid]}
    coupled["mesh"]["ny_per_region"] = [2, 2]
    coupled.setdefault("sensors", {}).update(sensor_y=0.5e-3, observable="displacement")
    with pytest.raises(ValueError, match="supports sensor_observable 'displacement'"):
        parse_and_resolve_config(custom_cfg=coupled)

    # Fluid-fluid interface with particle_velocity requires sensors.region
    two_fluids = _base()
    two_fluids["structure"] = {
        "mode": "N_layers",
        "layers": [water, dict(water, name="oil", rho=850.0, c_sound=1400.0)],
    }
    two_fluids["mesh"]["ny_per_region"] = [2, 2]
    two_fluids["excitation"] = {
        "f_center": 1e5,
        "source_kind": "acoustic_volume",
        "source_region": "water",
        "source_amplitude": 1.0,
    }
    two_fluids.setdefault("sensors", {}).update(sensor_y=1e-3, observable="particle_velocity")
    with pytest.raises(ValueError, match=r"requires sensors\.region"):
        parse_and_resolve_config(custom_cfg=two_fluids)
