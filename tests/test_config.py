"""Configuration schema, resolution, and structure-validation tests."""

from __future__ import annotations

import argparse
import copy
import math

import pytest

from guidedwavefem.config import (
    ABSORB_ALPHA_REF,
    ABSORB_BETA_REF,
    LEGACY_DISPERSION_KEYS,
    dump_config,
    flatten_config,
    load_config,
    parse_and_resolve_config,
)
from guidedwavefem.materials import structure_from_config
from guidedwavefem.sources import DEFAULT_N_CYCLES

ALUMINUM = {"thickness": 1e-3, "rho": 2700.0, "E": 70e9, "nu": 0.33, "material_model": "isotropic"}


def _base(**overrides):
    cfg = {
        "solver": {"type": "explicit"},
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
    assert cfg["source_x"] == pytest.approx(0.1 + 0.005)
    assert cfg["absorb_alpha"] == pytest.approx(ABSORB_ALPHA_REF)  # f0 = 1 MHz reference
    assert cfg["absorb_beta"] == pytest.approx(ABSORB_BETA_REF)
    assert cfg["source_type"] == "ricker" and cfg["n_cycles"] is None
    assert cfg["figure_formats"] == ["png", "pdf", "svg", "eps"]


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
