"""Duplicate-key validation must preserve YAML merge and alias semantics."""

from __future__ import annotations

import pytest
import yaml

from wavesimfem.config import load_config


def _load(tmp_path, source):
    path = tmp_path / "config.yaml"
    path.write_text(source, encoding="utf-8")
    return load_config(str(path))


@pytest.mark.parametrize(
    "source,key",
    [
        ("solver: {CFL: 0.5, CFL: 1.0}", "CFL"),
        ("solver: {<<: {CFL: 0.5, CFL: 1.0}}", "CFL"),
        ("solver: {<<: {<<: {CFL: 0.5, CFL: 1.0}}}", "CFL"),
        ("solver: {<<: [{CFL: 0.5}, {CFL: 0.8, CFL: 1.0}]}", "CFL"),
        ("solver: {<<: {CFL: 0.5}, <<: {CFL: 1.0}}", "<<"),
        ("layers: [{<<: &base {E: 70e9, E: 7e9}}, *base]", "E"),
        ("settings: {true: 1, 1: 2}", "1"),
        ("settings: {=: 1, '=': 2}", "="),
    ],
)
def test_duplicate_original_keys_are_rejected(tmp_path, source, key):
    with pytest.raises(ValueError, match="Duplicate key") as exc_info:
        _load(tmp_path, source)
    message = str(exc_info.value)
    assert key in message
    assert "first defined at" in message
    assert "line 1" in message


@pytest.mark.parametrize(
    "source",
    [
        "solver: {<<: {CFL: 0.5}, CFL: 0.8}",
        "layers: [{<<: &derived {<<: {E: 70e9}, E: 7e9}}, *derived]",
        (
            "base: &base {CFL: 0.5, steps_per_period: 40}\n"
            "solver: {<<: [*base, {CFL: 1.0, duration: 0.1}], CFL: 0.8}"
        ),
        "solver: {<<: [{CFL: 0.5}, {CFL: 1.0}]}",
        "solver: {<<: {CFL: 0.5}, '<<': literal}",
        "settings: {=: 1, other: 2}",
        "settings: &settings {<<: *settings, value: 1}",
        "settings: &settings {<<: &other {<<: *settings, value: 2}, value: 1}",
    ],
)
def test_legitimate_merges_match_safe_loader(tmp_path, source):
    assert _load(tmp_path, source) == yaml.safe_load(source)


def test_reused_derived_anchor_preserves_alias_identity(tmp_path):
    cfg = _load(tmp_path, "layers: [{<<: &derived {<<: {E: 70e9}, E: 7e9}}, *derived]")
    assert cfg["layers"] == [{"E": "7e9"}, {"E": "7e9"}]
    assert cfg["layers"][0] is not cfg["layers"][1]
    # References to the anchor itself still share the same constructed object.
    cfg = _load(tmp_path, "layers: [&derived {<<: {E: 70e9}, E: 7e9}, *derived]")
    assert cfg["layers"][0] is cfg["layers"][1]


def test_merge_sequence_uses_first_mapping_precedence(tmp_path):
    cfg = _load(tmp_path, "solver: {<<: [{CFL: 0.5}, {CFL: 1.0, duration: 0.1}]}")
    assert cfg == {"solver": {"CFL": 0.5, "duration": 0.1}}


def test_cyclic_alias_is_visited_once(tmp_path):
    cfg = _load(tmp_path, "settings: &settings {self: *settings, value: 1}")
    assert cfg["settings"]["self"] is cfg["settings"]
    assert cfg["settings"]["value"] == 1


def test_duplicate_mapping_reachable_through_alias_cycle_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate key 'value'"):
        _load(tmp_path, "settings: &settings {self: *settings, value: 1, value: 2}")


@pytest.mark.parametrize("source", ["settings: {[key]: value}", "settings: {{key: value}: 1}"])
def test_non_scalar_mapping_keys_are_rejected(tmp_path, source):
    with pytest.raises(ValueError, match="Invalid mapping key"):
        _load(tmp_path, source)


@pytest.mark.parametrize("source", ["solver: {<<: 1}", "solver: {<<: [1]}"])
def test_invalid_merge_sources_keep_clear_yaml_errors(tmp_path, source):
    with pytest.raises(ValueError, match="Invalid YAML") as exc_info:
        _load(tmp_path, source)
    assert "expected a mapping" in str(exc_info.value)
