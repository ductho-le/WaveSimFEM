"""Integration checks for structure-aware reports using analytical wave records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from wavesimfem import wave_reports
from wavesimfem.units import displacement_unit, length_unit

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def captured_figures(monkeypatch):
    figures = []

    def save(fig, base, formats, dpi):
        figures.append(fig)
        return [str(base) + "." + extension for extension in formats]

    monkeypatch.setattr(wave_reports, "_save", save)
    return figures


def _archive(path, *, config=None, leftward=False, source_x=None, **fields):
    times = np.arange(512) * 1e-7
    positions = np.arange(48) * 0.001
    frequency = 390_625.0
    direction = -1 if leftward else 1
    phase = 2 * np.pi * frequency * (times[:, None] - direction * positions[None, :] / 2500)
    values = dict(
        times=times,
        sensor_x=positions,
        sensor_data_x=1e-9 * np.cos(phase),
        sensor_data_y=2e-9 * np.sin(phase),
        f_center=frequency,
        source_x=(0.06 if leftward else -0.01) if source_x is None else source_x,
        structure_mode="halfspace",
        material_model="isotropic",
        c_L=6000.0,
        c_T=3000.0,
        c_R=2800.0,
        total_depth=0.05,
        sensor_y=0.0,
        y_min=-0.05,
    )
    if config is not None:
        values["simulation_config_yaml"] = yaml.safe_dump(config)
    values.update(fields)
    np.savez(path, **values)
    return values


def _run(path, **kwargs):
    return wave_reports.write_structure_report(path, formats=("png",), usetex=False, **kwargs)


def _rows(path, name):
    with (path / "wave_physics" / (name + ".csv")).open() as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize(
    "pattern,family,diagnostic",
    [
        ("01_*.yaml", "free_isotropic_plate", "polarization"),
        ("02_*.yaml", "layered_plate", "polarization"),
        ("03_*.yaml", "solid_halfspace", "polarization"),
        ("04_*.yaml", "layered_halfspace", "wavelength_thickness"),
        ("05_*.yaml", "solid_fluid_waveguide", "fluid_sound_line"),
        ("06_*.yaml", "fluid_loaded_plate", "fluid_sound_line"),
    ],
)
def test_structure_selects_matching_analysis_and_persists_measurements(
    tmp_path, captured_figures, pattern, family, diagnostic
):
    config = yaml.safe_load(next(CONFIGS.glob(pattern)).read_text())
    archive = tmp_path / "wave.npz"
    _archive(archive, config=config, provenance="Synthetic analytical verification")
    result = _run(archive)
    assert result["profile"]["family"] == family
    assert result["summary"]["structure_diagnostic"] == diagnostic
    assert len(captured_figures) == len(result["figures"]) == 2
    assert all(
        len([ax for ax in fig.axes if ax.get_title().startswith(("(a)", "(b)", "(c)", "(d)"))]) == 4
        for fig in captured_figures
    )
    assert all(not fig.texts for fig in captured_figures)
    assert all(not ax.texts for fig in captured_figures for ax in fig.axes)
    assert all(fig.get_figwidth() == pytest.approx(190 / 25.4) for fig in captured_figures)
    # Power and coherence need distinct bars; the two displacement maps share one.
    assert len(captured_figures[0].axes) == 6
    assert len(captured_figures[1].axes) == 5
    output = tmp_path / "wave_physics"
    assert json.loads((output / "report.json").read_text()) == result
    readme = (output / "README.md").read_text()
    assert "Synthetic analytical verification" in readme
    assert "190 mm export width and 10 pt built-in serif/mathtext" in readme
    assert "Grey cells fail signal or wavelength support" in readme
    assert result["interpretation"] in readme
    assert "no receiver-by-receiver amplitude normalization" in readme
    for region in result["profile"]["regions"]:
        assert region["name"] in readme

    for name in ("phase_velocity", "polarization", "spectral_decay", "directionality"):
        assert _rows(tmp_path, name)
    with np.load(output / "spectral_arrays.npz", allow_pickle=False) as exported:
        assert exported["coherence"].shape == exported["support_mask"].shape
        assert exported["coherence"].shape == (
            len(exported["frequencies_hz"]),
            len(exported["phase_velocities_m_s"]),
        )
        assert np.isnan(exported["ridge_velocity_m_s"]).sum() > 0
    theoretical = {r["label"] for r in result["references"] if r["kind"] == "lamb"}
    assert theoretical == ({"A0", "S0"} if family == "free_isotropic_plate" else set())


@pytest.mark.parametrize("leftward", [False, True])
def test_directional_content_recovers_outgoing_wave(tmp_path, captured_figures, leftward):
    archive = tmp_path / "wave.npz"
    _archive(archive, leftward=leftward)
    result = _run(archive)
    assert result["summary"]["direction"].startswith("-x" if leftward else "+x")
    assert result["summary"]["accepted_velocity_samples"] > 0
    rows = _rows(tmp_path, "directionality")
    dominant = min(rows, key=lambda r: abs(float(r["frequency_hz"]) - 390_625))
    assert float(dominant["opposite_direction_power_fraction"]) < 1e-4
    velocity = _rows(tmp_path, "phase_velocity")
    dominant = min(velocity, key=lambda r: abs(float(r["frequency_hz"]) - 390_625))
    assert dominant["accepted"] == "True"
    assert float(dominant["phase_velocity_m_s"]) == pytest.approx(2500, rel=0.015)


def test_internal_source_selects_one_side_but_propagation_keeps_all_receivers(
    tmp_path, captured_figures
):
    archive = tmp_path / "wave.npz"
    fields = _archive(archive, source_x=0.017)
    result = _run(archive)
    expected = np.flatnonzero(fields["sensor_x"] > 0.017)
    assert result["summary"]["selected_receiver_count"] == len(expected)
    assert result["summary"]["total_receiver_count"] == 48
    assert result["summary"]["direction"] == "+x from source"
    with np.load(tmp_path / "wave_physics" / "spectral_arrays.npz") as exported:
        np.testing.assert_array_equal(exported["receiver_indices"], expected)
    for ax in [
        ax
        for ax in captured_figures[1].axes
        if any(word in ax.get_title() for word in ("Horizontal", "Vertical"))
    ]:
        assert np.asarray(ax.collections[0].get_array()).shape == (48, 512)
    offsets = fields["sensor_x"] - fields["source_x"]
    unit = length_unit(np.max(np.abs(offsets)))
    axis = next(ax for ax in captured_figures[1].axes if "Horizontal" in ax.get_title())
    assert axis.get_ylim()[0] < unit(offsets[0])
    assert axis.get_ylim()[1] > unit(offsets[-1])


def test_source_without_three_receivers_on_either_side_cannot_qualify_ridge(
    tmp_path, captured_figures
):
    archive = tmp_path / "wave.npz"
    fields = _archive(archive, source_x=0.002)
    fields["sensor_x"] = fields["sensor_x"][:5]
    for key in ("sensor_data_x", "sensor_data_y"):
        fields[key] = fields[key][:, :5]
    np.savez(archive, **fields)
    result = _run(archive)
    assert result["summary"]["accepted_velocity_samples"] == 0
    assert "No one-sided aperture" in result["summary"]["direction"]
    assert {r["status"] for r in _rows(tmp_path, "phase_velocity")} == {"no one-sided aperture"}
    with np.load(tmp_path / "wave_physics" / "spectral_arrays.npz") as exported:
        assert not exported["support_mask"].any()


@pytest.mark.parametrize("missing_x", [False, True])
def test_zero_or_missing_horizontal_component_does_not_invent_polarization(
    tmp_path, captured_figures, missing_x
):
    archive = tmp_path / "wave.npz"
    fields = _archive(archive)
    if missing_x:
        fields.pop("sensor_data_x")
    else:
        fields["sensor_data_x"][:] = 0
    np.savez(archive, **fields)
    result = _run(archive)
    assert result["summary"]["selected_component"] == "y"
    assert result["summary"]["coherent_polarization_samples"] == 0
    assert all(r["vertical_horizontal_ratio"] == "" for r in _rows(tmp_path, "polarization"))
    horizontal = next(ax for ax in captured_figures[1].axes if "Horizontal" in ax.get_title())
    assert not horizontal.texts
    assert "Horizontal component absent" in (tmp_path / "wave_physics" / "README.md").read_text()
    assert (
        "Horizontal component absent or without dynamic signal."
        in result["summary"]["unavailable_measurements"]
    )


def test_horizontal_only_wave_is_analyzed_and_maps_share_physical_scale(tmp_path, captured_figures):
    archive = tmp_path / "wave.npz"
    fields = _archive(archive)
    fields["sensor_data_y"][:] = 0
    np.savez(archive, **fields)
    result = _run(archive)
    assert result["summary"]["selected_component"] == "x"
    assert result["summary"]["accepted_velocity_samples"] > 0
    maps = [
        ax.collections[0]
        for ax in captured_figures[1].axes
        if any(word in ax.get_title() for word in ("Horizontal", "Vertical"))
    ]
    assert maps[0].get_clim() == maps[1].get_clim()
    assert maps[0].norm is maps[1].norm
    assert maps[1].colorbar.norm is maps[0].norm
    assert maps[1].colorbar.extend == "neither"
    unit = displacement_unit(np.max(abs(fields["sensor_data_x"])))
    assert np.max(abs(maps[0].get_array())) == pytest.approx(
        unit(np.max(abs(fields["sensor_data_x"]))), rel=1e-10
    )
    assert not np.any(maps[1].get_array())


def test_silent_archive_produces_explicit_unavailable_measurements(tmp_path, captured_figures):
    archive = tmp_path / "wave.npz"
    fields = _archive(archive)
    fields["sensor_data_x"][:] = fields["sensor_data_y"][:] = 0
    np.savez(archive, **fields)
    result = _run(archive)
    assert result["summary"]["accepted_velocity_samples"] == 0
    assert result["summary"]["coherent_polarization_samples"] == 0
    assert all(r["apparent_decay_np_m"] == "" for r in _rows(tmp_path, "spectral_decay"))
    assert all(
        r["opposite_direction_power_fraction"] == "" for r in _rows(tmp_path, "directionality")
    )
    assert len(captured_figures) == 2
    json.dumps(result, allow_nan=False)


def test_narrow_band_without_welch_bin_preserves_dispersion(tmp_path, captured_figures):
    archive = tmp_path / "wave.npz"
    _archive(archive)
    result = _run(archive, f_min=390_000, f_max=391_000)
    assert result["polarization_method"]["status"] == "unavailable"
    assert "Welch" in result["polarization_method"]["reason"]
    assert result["summary"]["accepted_velocity_samples"] == 1
    assert len(captured_figures) == 2
    rows = _rows(tmp_path, "phase_velocity")
    assert len(rows) == 1
    assert rows[0]["accepted"] == "True"


@pytest.mark.parametrize("leftward", [False, True])
def test_packet_spectral_decay_measures_known_attenuation_without_taper_bias(
    tmp_path, captured_figures, leftward
):
    archive = tmp_path / "wave.npz"
    fields = _archive(archive, leftward=leftward)
    times = np.arange(2048) * 2e-7
    x = fields["sensor_x"]
    distance = x[-1] - x if leftward else x - x[0]
    tau = times[:, None] - 60e-6 - distance[None, :] / 2500
    alpha, frequency = 12.0, 250_000.0
    packet = (
        np.exp(-alpha * distance[None, :])
        * np.exp(-((tau / 8e-6) ** 2))
        * np.cos(2 * np.pi * frequency * tau)
    )
    fields.update(
        times=times, sensor_data_x=np.zeros_like(packet), sensor_data_y=packet, f_center=frequency
    )
    np.savez(archive, **fields)
    _run(archive)
    rows = _rows(tmp_path, "spectral_decay")
    dominant = min(rows, key=lambda r: abs(float(r["frequency_hz"]) - frequency))
    assert dominant["status"] == "accepted"
    assert float(dominant["apparent_decay_np_m"]) == pytest.approx(alpha, rel=0.005)
    assert float(dominant["apparent_decay_db_m"]) == pytest.approx(
        alpha * 20 / np.log(10), rel=0.005
    )


@pytest.mark.parametrize(
    "problem", ["missing", "nan", "shape", "nonuniform", "reversed", "short", "source"]
)
def test_invalid_archive_fails_before_any_report_is_saved(tmp_path, captured_figures, problem):
    archive = tmp_path / "wave.npz"
    fields = _archive(archive)
    if problem == "missing":
        fields.pop("sensor_data_y")
    elif problem == "nan":
        fields["sensor_data_y"][0, 0] = np.nan
    elif problem == "shape":
        fields["sensor_data_y"] = fields["sensor_data_y"].T
    elif problem == "nonuniform":
        fields["times"][10] += 1e-9
    elif problem == "reversed":
        fields["sensor_x"] = fields["sensor_x"][::-1]
    elif problem == "short":
        fields["times"] = fields["times"][:16]
    elif problem == "source":
        fields["source_x"] = np.nan
    np.savez(archive, **fields)
    with pytest.raises(ValueError):
        _run(archive)
    assert captured_figures == []
    assert not (tmp_path / "wave_physics" / "report.json").exists()


def test_buried_receiver_does_not_receive_surface_rayleigh_ellipticity_reference(
    tmp_path, captured_figures
):
    config = yaml.safe_load(next(CONFIGS.glob("03_*.yaml")).read_text())
    archive = tmp_path / "wave.npz"
    _archive(archive, config=config, sensor_y=-0.001)
    _run(archive)
    axis = next(ax for ax in captured_figures[0].axes if ax.get_title().startswith("(d)"))
    assert not any("Surface Rayleigh" in line.get_label() for line in axis.lines)


def test_structure_report_readme_tracks_usetex_mode(tmp_path, captured_figures, monkeypatch):
    monkeypatch.setattr("wavesimfem.analysis.latex_available", lambda: True)
    archive = tmp_path / "wave.npz"
    _archive(archive)
    wave_reports.write_structure_report(archive, formats=("png",), usetex=True)
    readme = (tmp_path / "wave_physics" / "README.md").read_text()
    assert "190 mm export width and 10 pt external LaTeX" in readme


def test_scalar_pressure_propagation_colorbar_uses_pressure_only(tmp_path, captured_figures):
    t = np.arange(512) * 1e-7
    x = 0.02 + np.arange(24) * 1e-3
    # Build a zero-mean sparse high-amplitude burst (8/512 = 1.56% of samples):
    # pressure-only 99th percentile is 100 Pa, whereas stacking a zero placeholder
    # drops the tail fraction to 0.78% (< 1%) and collapses the limit to ~1 Pa.
    pressure = np.zeros((len(t), len(x)))
    pressure[:4, :] = 100.0
    pressure[4:8, :] = -100.0
    pressure[8::2, :] = 1.0
    pressure[9::2, :] = -1.0
    archive = tmp_path / "pressure.npz"
    np.savez(
        archive,
        schema_version=2,
        times=t,
        sensor_x=x,
        sensor_y=0.005,
        source_x=0.0,
        f_center=1e5,
        disp_f_max=1.5e5,
        c_fluid=1500.0,
        c_L=1500.0,
        sensor_observable="pressure",
        sensor_units="Pa",
        sensor_data_pressure=pressure,
        simulation_config_yaml=yaml.safe_dump(
            {
                "structure_mode": "1_layer",
                "single_layer": {
                    "name": "Water",
                    "thickness": 0.01,
                    "rho": 1000.0,
                    "c": 1500.0,
                    "material_model": "fluid",
                },
            }
        ),
    )
    wave_reports.write_structure_report(archive, formats=("png",), usetex=False)
    pressure_ax = next(ax for ax in captured_figures[1].axes if "Pressure" in ax.get_title())
    vmax = pressure_ax.collections[0].get_clim()[1]
    assert vmax == pytest.approx(100.0, rel=1e-6)
