"""Preflight stability certificates and the limits of automatic arrival windows."""

from types import SimpleNamespace

import pytest

from guidedwavefem.config import parse_and_resolve_config
from guidedwavefem.materials import (
    compute_mesh_layout,
    duration_sizing_notes,
    preflight_estimate,
    sampled_time_grid,
    sizing_group_velocity,
    structure_from_config,
)
from guidedwavefem.reporting import logger


def plate_config(**updates):
    values = dict(
        solver_type="implicit",
        sensing_length=0.3,
        absorb_length=0.05,
        f_center=250e3,
        structure_mode="1_layer",
        single_layer=dict(thickness=0.003, rho=2700.0, E=70e9, nu=0.33),
        nx=663,
        ny_per_region=[5],
        CFL=0.8,
        run_postprocess=False,
    )
    values.update(updates)
    return parse_and_resolve_config(custom_cfg=values)


@pytest.mark.parametrize("factor,expect_warning", [(0.999, False), (1.0, True), (1.001, True)])
def test_higher_mode_cutoff_is_reported(factor, expect_warning):
    cfg = plate_config()
    summary = structure_from_config(cfg)
    plate = summary["all_regions"][0]
    cutoff = plate["c_T"] / (2 * plate["thickness"])
    cfg["max_frequency"] = cutoff * factor
    basis, warnings = duration_sizing_notes(cfg, summary)
    assert "A0" in basis
    assert bool(warnings) is expect_warning
    if expect_warning:
        assert "higher Lamb modes" in warnings[0]
        assert "solver.duration" in warnings[0]
        assert "solver.min_group_velocity" in warnings[0]
        assert "no finite window" in warnings[0]


def test_automatic_duration_does_not_claim_higher_mode_coverage():
    cfg = plate_config()
    automatic = preflight_estimate(cfg, rank=1)
    # An A1 packet at 528.6 kHz has an independently calculated group velocity
    # near 1024 m/s; its 250 mm flight takes ~244 us. The A0 window is shorter.
    assert automatic["duration"] < 0.25 / 1024
    assert "A0" in automatic["duration_basis"]
    assert any("may truncate" in warning for warning in automatic["warnings"])


@pytest.mark.parametrize(
    "updates,basis",
    [(dict(duration=0.002), "user-specified"), (dict(min_group_velocity=500), "user-supplied")],
)
def test_explicit_arrival_target_resolves_warning(updates, basis):
    cfg = plate_config(**updates)
    estimates = preflight_estimate(cfg, rank=1)
    assert basis in estimates["duration_basis"]
    assert not estimates["warnings"]
    if "duration" in updates:
        assert estimates["duration"] == updates["duration"]
    else:
        summary = structure_from_config(cfg)
        assert sizing_group_velocity(cfg, summary) <= updates["min_group_velocity"]
        assert estimates["duration"] >= cfg["sensing_length"] / updates["min_group_velocity"]


def test_higher_mode_warning_survives_quiet_logging(capsys):
    previous_level = logger.level
    try:
        logger.setLevel("WARNING")
        preflight_estimate(plate_config())
    finally:
        logger.setLevel(previous_level)
    output = capsys.readouterr().out
    assert "WARNING:" in output and "higher Lamb modes" in output
    assert "PREFLIGHT NUMERICAL ESTIMATE" not in output


def test_higher_mode_warning_is_only_emitted_on_rank_zero(capsys):
    estimates = preflight_estimate(plate_config(), rank=1)
    assert estimates["warnings"]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_analysis_band_never_changes_user_mesh_counts(solver):
    cfg = plate_config(solver_type=solver, f_center=1e6, max_frequency=600e3, duration=2e-5)
    estimates = preflight_estimate(cfg, rank=1)
    assert estimates["mesh_frequency"] == 600e3
    assert (estimates["nx"], estimates["ny_per_region"]) == (663, [5])
    assert not estimates["warnings"]


@pytest.mark.parametrize("requested_dt", [1e-8, 7.5e-8, 1e-7])
def test_certified_explicit_request_uses_requested_grid(monkeypatch, requested_dt):
    # The element certificate is 100 ns. At CFL=0.5 the automatic step is 50 ns,
    # but a user request between 50 and 100 ns remains certified and is honored.
    monkeypatch.setattr("guidedwavefem.materials.explicit_time_step_estimate", lambda *_: 5e-8)
    cfg = plate_config(solver_type="explicit", dt=requested_dt, duration=2e-5, CFL=0.5)
    estimates = preflight_estimate(cfg, rank=1)
    expected = sampled_time_grid(
        cfg, compute_mesh_layout(cfg, structure_from_config(cfg)), requested_dt
    )
    assert estimates["dt_element_limit"] == pytest.approx(1e-7)
    assert estimates["dt_stability_status"] == "certified_by_element_bound"
    assert estimates["dt_requested"] == requested_dt
    assert estimates["dt_est"] == expected[0]
    assert estimates["nsteps_est"] == expected[1]
    assert estimates["sample_stride_est"] == expected[2]
    assert not estimates["warnings"]


def test_uncertified_explicit_request_warns_and_costs_conservatively(monkeypatch, capsys):
    monkeypatch.setattr("guidedwavefem.materials.explicit_time_step_estimate", lambda *_: 5e-8)
    cfg = plate_config(solver_type="explicit", dt=10e-6, duration=2e-5)
    estimates = preflight_estimate(cfg)
    assert estimates["dt_stability_status"] == "requires_assembled_check"
    assert estimates["dt_requested"] == 10e-6
    assert estimates["dt_est"] <= 5e-8
    assert estimates["nsteps_est"] >= 400
    assert (
        estimates["n_samples_est"] == estimates["nsteps_est"] // estimates["sample_stride_est"] + 1
    )
    output = capsys.readouterr().out
    assert "WARNING:" in output
    assert "assembled-operator check may reject" in output
    assert "does not prove instability" in output


def test_implicit_request_does_not_apply_explicit_certificate(monkeypatch):
    def must_not_run(*_args):
        pytest.fail("Explicit stability estimate must not constrain implicit Newmark")

    monkeypatch.setattr("guidedwavefem.materials.explicit_time_step_estimate", must_not_run)
    estimates = preflight_estimate(plate_config(dt=10e-6, duration=2e-5), rank=1)
    assert estimates["dt_element_limit"] is None
    assert estimates["dt_stability_status"] == "not_applicable"
    assert estimates["nsteps_est"] == 2
    assert not estimates["warnings"]


def test_preflight_rounds_step_count_to_complete_sample_stride():
    cfg = plate_config(
        f_center=1e6,
        duration=1e-5,
        dt=7e-9,
        sensor_sample_rate=23e6,
    )
    estimates = preflight_estimate(cfg, rank=1)
    # Initial ceil(T / dt) = 1429. The six-step recording stride requires 1434
    # solver steps, yielding 239 intervals and 240 uniformly spaced samples.
    assert estimates["nsteps_est"] == 1434
    assert estimates["sample_stride_est"] == 6
    assert estimates["n_samples_est"] == 240
    assert estimates["dt_est"] == pytest.approx(1e-5 / 1434)
    assert estimates["dt_est"] * estimates["nsteps_est"] == pytest.approx(1e-5)


def test_shared_time_grid_preserves_postprocess_sample_requirement():
    cfg = dict(f_center=1e6, run_postprocess=True)
    with pytest.raises(ValueError, match="at least 32 time samples"):
        sampled_time_grid(cfg, SimpleNamespace(duration=1e-5, mesh_frequency=3e6), 1e-6)


@pytest.mark.parametrize("mode", [None, "memory", "disk", " DISK "])
def test_history_storage_configuration(mode):
    cfg = plate_config(history_storage=mode)
    expected = "memory" if mode is None else mode.strip().lower()
    assert cfg["history_storage"] == expected
    cfg = parse_and_resolve_config(
        custom_cfg={"output": {"history_storage": mode}}, validate_simulation=False
    )
    assert cfg["history_storage"] == expected


@pytest.mark.parametrize("mode", [True, 0, [], "file", "memmory"])
def test_invalid_history_storage_is_rejected(mode):
    with pytest.raises(ValueError, match="history_storage"):
        plate_config(history_storage=mode)


def test_disk_history_costs_scale_on_disk_with_bounded_measurement_buffers():
    cfg = plate_config(
        duration=0.2,
        dt=1e-7,
        n_sensors=1001,
        sensor_pitch=1e-4,
        sensor_sample_rate=1e7,
        history_storage="disk",
    )
    disk = preflight_estimate(cfg, rank=1)
    memory = preflight_estimate(dict(cfg, history_storage="memory"), rank=1)
    assert disk["history_size_mb"] == memory["history_mem_mb"]
    assert disk["history_disk_mb"] == 2 * disk["history_size_mb"]
    assert disk["archive_payload_mb"] > disk["history_size_mb"]
    assert disk["history_mem_mb"] == 0
    assert disk["transfer_mem_mb"] <= 16
    assert disk["collection_mem_mb"] <= 16
    assert disk["timestamp_mem_mb"] > 0
    assert disk["total_mem_est_mb"] < memory["total_mem_est_mb"]


def test_disk_history_costs_allow_a_row_larger_than_chunk_budget(monkeypatch):
    monkeypatch.setattr("guidedwavefem.materials.HISTORY_CHUNK_BYTES", 8)
    cfg = plate_config(duration=1e-3, history_storage="disk")
    disk = preflight_estimate(cfg, rank=1)
    row_mb = 2 * cfg["n_sensors"] * 8 / 1024**2
    assert disk["transfer_mem_mb"] >= row_mb
    assert disk["collection_mem_mb"] >= row_mb
