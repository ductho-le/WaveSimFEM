"""Scientific display regressions using resolved, analytically known waves.

These tests need no FEM runtime. They verify that a polished figure still shows
the measured wave, its true receiver geometry, and the available sampling band.
"""

from __future__ import annotations

import numpy as np
import pytest

from guidedwavefem import analysis


def _travelling_archive(path, *, leftward=False, **metadata):
    t = np.arange(256) * 1e-7
    x = np.arange(64) * 0.0005
    frequency = 625_000.0
    wave = 2 * np.pi * 8 / (64 * 0.0005)
    phase = 2 * np.pi * frequency * t[:, None]
    phase = phase + (1 if leftward else -1) * wave * x[None, :]
    signal = 1e-9 * np.cos(phase)
    fields = dict(
        times=t,
        sensor_x=x,
        sensor_data_y=signal,
        sensor_data_x=signal,
        f_center=frequency,
        c_L=6000.0,
        c_T=3000.0,
        source_x=0.1 if leftward else -0.01,
    )
    fields.update(metadata)
    np.savez(path, **fields)
    return str(path), fields


def _capture(monkeypatch):
    figures = []

    def save(fig, *args, **kwargs):
        figures.append(fig)

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", save)
    return figures


def _axes(fig, title):
    return [ax for ax in fig.axes if title in ax.get_title()]


def test_automatic_wavenumber_range_keeps_slow_guided_modes():
    """A shear-speed bound is not a bound on flexural-mode wavenumber."""
    limits = analysis._resolve_limits({}, {}, f_center=100_000, c_ref=3000, pitch=0.0005)
    assert limits["k_max"] == pytest.approx(np.pi / 0.0005)
    # A resolvable 150 m/s packet at f0 lies outside the old c_T-derived limit.
    slow_mode_wave = 2 * np.pi * 100_000 / 150
    assert slow_mode_wave < limits["k_max"]


def test_left_of_source_spectrum_recovers_outgoing_wave(tmp_path, monkeypatch):
    path, _ = _travelling_archive(tmp_path / "left.npz", leftward=True)
    figures = _capture(monkeypatch)
    analysis.plot_full_analysis(path, str(tmp_path), formats=["png"], usetex=False)
    for ax in _axes(figures[0], "Dispersion"):
        spectrum = np.asarray(ax.images[0].get_array())
        frequency_index, wave_index = np.unravel_index(spectrum.argmax(), spectrum.shape)
        # 2x zero-padded FFT: 32 temporal cycles and 16 spatial cycles.
        assert frequency_index == 32
        assert wave_index == 16
    assert not figures[0].texts
    assert "Propagation: -x" in (tmp_path / "full_analysis_README.md").read_text()


def test_right_of_source_spectrum_preserves_forward_wave(tmp_path, monkeypatch):
    path, _ = _travelling_archive(tmp_path / "right.npz")
    figures = _capture(monkeypatch)
    analysis.plot_full_analysis(path, str(tmp_path), formats=["png"], usetex=False)
    for ax in _axes(figures[0], "Dispersion"):
        spectrum = np.asarray(ax.images[0].get_array())
        assert np.unravel_index(spectrum.argmax(), spectrum.shape) == (32, 16)
    assert not figures[0].texts
    assert "Propagation: +x" in (tmp_path / "full_analysis_README.md").read_text()


def test_recorded_geometry_overrides_stale_layout_metadata(tmp_path, monkeypatch):
    path, fields = _travelling_archive(
        tmp_path / "geometry.npz", sensor_offset=-12.0, sensor_pitch=2.0
    )
    figures = _capture(monkeypatch)
    analysis.plot_full_analysis(path, str(tmp_path), formats=["png"], usetex=False)
    scan = _axes(figures[0], "B-scan")[0]
    # Actual aperture is 10 to 41.5 mm from source. Layout metadata is stale.
    unit = analysis.length_unit(float(np.max(abs(fields["sensor_x"] - fields["source_x"]))))
    expected = unit(fields["sensor_x"][[0, -1]] - fields["source_x"])
    np.testing.assert_allclose(scan.get_ylim(), expected)
    expected_nyquist = np.pi / (fields["sensor_x"][1] - fields["sensor_x"][0])
    wave_unit = analysis.wavenumber_unit(analysis.length_unit(2 * np.pi / expected_nyquist))
    for ax in _axes(figures[0], "Dispersion"):
        assert ax.get_xlim()[1] == pytest.approx(wave_unit(expected_nyquist))


def test_display_range_cannot_claim_unrecorded_frequencies(tmp_path, monkeypatch):
    path, fields = _travelling_archive(tmp_path / "nyquist.npz")
    figures = _capture(monkeypatch)
    warnings = []
    monkeypatch.setattr(analysis, "warn", warnings.append)
    analysis.plot_full_analysis(
        path, str(tmp_path), f_max="20 MHz", k_max="100 rad/mm", formats=["png"], usetex=False
    )
    f_nyquist = 0.5 / np.diff(fields["times"])[0]
    k_nyquist = np.pi / np.diff(fields["sensor_x"])[0]
    f_unit = analysis.frequency_unit(f_nyquist)
    k_unit = analysis.wavenumber_unit(analysis.length_unit(2 * np.pi / k_nyquist))
    for ax in _axes(figures[0], "Dispersion"):
        assert ax.get_ylim()[1] <= f_unit(f_nyquist)
        assert ax.get_xlim()[1] <= k_unit(k_nyquist)
    assert warnings, "Clipping an explicit display request must be explained to the caller"


def test_failed_export_keeps_existing_figure(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    target = tmp_path / "figure.png"
    target.write_bytes(b"previous completed figure")

    def fail(fig, stream, **kwargs):
        stream.write(b"incomplete replacement")
        raise RuntimeError("render failed")

    monkeypatch.setattr(Figure, "savefig", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        analysis._save(Figure(), str(target.with_suffix("")), ["png"], 100)
    assert target.read_bytes() == b"previous completed figure"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["figure.png"]
