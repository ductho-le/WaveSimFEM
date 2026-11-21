"""Analytical packet checks for receiver characterisation."""

import json

import numpy as np
import pytest

from wavesimfem.characterisation import measure_packets, validate_options, write_characterisation


def packet():
    t = np.linspace(0, 0.012, 1201)
    x = np.linspace(0.1, 0.9, 9)
    phase = t[:, None] - 0.003 - x / 200
    envelope = np.exp(-((phase / 0.0004) ** 2)) * np.exp(-2 * x)
    return (
        t,
        x,
        envelope * np.cos(2 * np.pi * 4000 * phase),
        envelope * np.sin(2 * np.pi * 4000 * phase),
    )


def test_known_velocity_and_attenuation():
    report, _ = measure_packets(*packet())
    assert report["summary"]["velocity"]["signed_m_per_s"] == pytest.approx(200, rel=0.01)
    assert report["summary"]["apparent_attenuation"]["np_per_m"] == pytest.approx(2, rel=0.01)
    assert report["summary"]["valid_receivers"] == 9


def test_zero_signal_has_no_fits():
    t, x, ux, uy = packet()
    report, _ = measure_packets(t, x, ux * 0, uy * 0)
    assert report["summary"]["velocity"] is None
    assert report["summary"]["apparent_attenuation"] is None
    json.dumps(report, allow_nan=False)


def test_zero_lower_edge_uses_lowpass():
    from scipy.signal import butter, sosfiltfilt

    t, x, ux, uy = packet()
    options = {"frequency_band": ["0 MHz", "0.006 MHz"]}
    report, arrays = measure_packets(t, x, ux, uy, options)
    expected = sosfiltfilt(
        butter(4, 6000, btype="lowpass", fs=1 / (t[1] - t[0]), output="sos"),
        uy - uy.mean(axis=0),
        axis=0,
    )
    np.testing.assert_allclose(arrays[3], expected)
    assert report["summary"]["velocity"]["signed_m_per_s"] == pytest.approx(200, rel=0.01)


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": "false"},
        {"component": "z"},
        {"frequency_band": [2, 1]},
        {"time_window": [0, float("inf")]},
        {"min_relative_amplitude": 0},
        {"typo": 1},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        validate_options(options)


def test_nyquist_and_gate_validation():
    with pytest.raises(ValueError, match="Nyquist"):
        measure_packets(*packet(), {"frequency_band": [100, 100000]})
    with pytest.raises(ValueError, match="recorded"):
        measure_packets(*packet(), {"time_window": [0, 1]})


def test_report_outputs(tmp_path):
    t, x, ux, uy = packet()
    archive = tmp_path / "sensors.npz"
    np.savez(archive, times=t, sensor_x=x, sensor_data_x=ux, sensor_data_y=uy)
    report = write_characterisation(archive, options={"frequency_band": ["2 kHz", "6 kHz"]})
    assert len(report["figures"]) == 4
    for name in ("arrival_velocity", "pulse_evolution", "particle_motion", "time_frequency"):
        assert (tmp_path / "characterisation" / f"{name}.png").stat().st_size > 1000
    saved = json.loads((tmp_path / "characterisation/metrics.json").read_text())
    assert saved["summary"]["velocity"]["signed_m_per_s"] == pytest.approx(200, rel=0.02)
    assert (tmp_path / "characterisation/receivers.csv").exists()
    readme = (tmp_path / "characterisation/README.md").read_text()
    assert "190 mm width with 10 pt built-in serif/mathtext" in readme
    assert "Apparent decay includes spreading" in readme
    assert "Unpadded STFT frequency spacing" in readme
    assert saved["figure_parameters"]["export_width_mm"] == 190
    assert len(saved["figure_notes"]) == 4


def test_gaussian_packet_width_and_subsample_arrivals():
    t = np.linspace(0, 0.012, 301)
    x = np.linspace(0.1, 0.9, 9)
    arrivals = 0.003017 + x / 213.7
    phase = t[:, None] - arrivals
    signal = np.exp(-((phase / 0.0004) ** 2)) * np.sin(2 * np.pi * 4000 * phase)
    report, _ = measure_packets(t, x, signal, signal)
    observed = np.array([row["arrival_s"] for row in report["receivers"]])
    # Peak interpolation should resolve arrival times much better than the
    # nearest stored sample, including peaks deliberately between samples.
    assert np.max(np.abs(observed - arrivals)) < 0.02 * (t[1] - t[0])
    assert report["summary"]["velocity"]["signed_m_per_s"] == pytest.approx(213.7, rel=1e-4)
    assert report["summary"]["fit_status"] == "accepted"
    for row in report["receivers"]:
        assert row["envelope_rms_width_s"] == pytest.approx(0.0002, rel=0.005)
        assert row["envelope_fwhm_s"] == pytest.approx(2 * np.sqrt(np.log(2)) * 0.0004, rel=0.005)


def test_static_baselines_do_not_change_packet_measurements():
    t, x, ux, uy = packet()
    original_ux, original_uy = ux.copy(), uy.copy()
    reference, _ = measure_packets(t, x, ux, uy)
    offsets = np.arange(len(x)) + 2
    shifted, _ = measure_packets(t, x, ux + offsets, uy - 3 * offsets)
    for name, key in (("velocity", "signed_m_per_s"), ("apparent_attenuation", "np_per_m")):
        assert shifted["summary"][name][key] == pytest.approx(
            reference["summary"][name][key], rel=1e-10
        )
    np.testing.assert_array_equal(ux, original_ux)
    np.testing.assert_array_equal(uy, original_uy)
    constant, _ = measure_packets(t, x, np.ones_like(ux), np.ones_like(uy) * 1.23)
    assert constant["summary"]["valid_receivers"] == 0
    assert all(row["exclusion_reason"] == "no_signal" for row in constant["receivers"])
    json.dumps(constant, allow_nan=False)


def test_incoherent_arrivals_do_not_produce_velocity_or_attenuation():
    t, x, _, _ = packet()
    arrivals = np.array([0.003, 0.007, 0.004, 0.005, 0.006, 0.004, 0.007, 0.005, 0.006])
    phase = t[:, None] - arrivals
    signal = np.exp(-((phase / 0.0004) ** 2)) * np.sin(2 * np.pi * 4000 * phase)
    report, _ = measure_packets(t, x, signal, signal)
    assert report["summary"]["fit_status"] == "inconsistent_arrivals"
    assert report["summary"]["fit_diagnostics"]["r_squared"] < 0.2
    assert report["summary"]["velocity"] is None
    assert report["summary"]["apparent_attenuation"] is None


def test_unresolved_transit_time_does_not_produce_arbitrary_velocity():
    t, x, _, _ = packet()
    phase = t[:, None] - 0.005 - x / 100000
    signal = np.exp(-((phase / 0.0004) ** 2)) * np.sin(2 * np.pi * 4000 * phase)
    report, _ = measure_packets(t, x, signal, signal)
    assert report["summary"]["fit_status"] == "unresolved_travel_time"
    assert report["summary"]["velocity"] is None
    assert report["summary"]["fit_diagnostics"]["fitted_time_span_s"] < 2 * (t[1] - t[0])


def test_truncated_packet_is_excluded_even_with_interior_peak():
    report, _ = measure_packets(*packet(), {"time_window": [0.0034, 0.0076]})
    first, last = report["receivers"][0], report["receivers"][-1]
    assert first["arrival_s"] > 0.0034
    assert last["arrival_s"] < 0.0076
    assert first["exclusion_reason"] == last["exclusion_reason"] == "truncated_at_gate"
    assert not first["included_in_fit"] and not last["included_in_fit"]
    assert report["summary"]["velocity"]["signed_m_per_s"] == pytest.approx(200, rel=0.01)
    assert report["preprocessing"]["effective_time_window_s"] == pytest.approx([0.0034, 0.0076])


def test_multiple_strong_packets_require_a_narrower_gate():
    t, x, _, _ = packet()
    phase = t[:, None] - 0.003 - x / 200
    signal = np.exp(-((phase / 0.0004) ** 2)) * np.sin(2 * np.pi * 4000 * phase)
    reflection = np.exp(-(((phase - 0.002) / 0.0004) ** 2)) * np.sin(
        2 * np.pi * 4000 * (phase - 0.002)
    )
    report, _ = measure_packets(t, x, signal + 0.8 * reflection, signal + 0.8 * reflection)
    assert report["summary"]["valid_receivers"] == 0
    assert report["summary"]["velocity"] is None
    assert all("multiple_packets" in row["exclusion_reason"] for row in report["receivers"])
    assert all(
        row["secondary_peak_ratio"] == pytest.approx(0.8, rel=0.01) for row in report["receivers"]
    )


def test_left_going_packet_retains_signed_speed_and_positive_decay():
    t, x, _, _ = packet()
    phase = t[:, None] - 0.008 + x / 200
    signal = np.exp(-((phase / 0.0004) ** 2)) * np.exp(2 * x) * np.sin(2 * np.pi * 4000 * phase)
    report, _ = measure_packets(t, x, signal, signal)
    assert report["summary"]["velocity"]["signed_m_per_s"] == pytest.approx(-200, rel=0.01)
    assert report["summary"]["apparent_attenuation"]["np_per_m"] == pytest.approx(2, rel=0.01)
    assert report["summary"]["apparent_attenuation"]["propagation_direction"] == -1


def test_invalid_empty_options_are_not_silently_accepted():
    with pytest.raises(ValueError, match="mapping"):
        measure_packets(*packet(), options=[])


def test_nonexponential_amplitude_does_not_report_decay():
    t, x, ux, uy = packet()
    # Undo the known decay and replace it with incoherent receiver gains.
    gains = np.exp(2 * x) * np.array([1.0, 0.3, 0.9, 0.4, 0.8, 0.3, 0.7, 0.5, 1.0])
    report, _ = measure_packets(t, x, ux * gains, uy * gains)
    assert report["summary"]["velocity"]["signed_m_per_s"] == pytest.approx(200, rel=0.01)
    assert report["summary"]["apparent_attenuation"] is None
    assert report["summary"]["amplitude_fit_diagnostics"]["r_squared"] < 0.8


def test_report_zero_and_seismic_records_render_finite_metrics(tmp_path):
    t, x, ux, uy = packet()
    for name, scale in (("seismic", 1000), ("zero", 1)):
        archive = tmp_path / (name + ".npz")
        factor = 0 if name == "zero" else 1e-6
        np.savez(
            archive,
            times=t * scale,
            sensor_x=x * scale,
            sensor_data_x=ux * factor,
            sensor_data_y=uy * factor,
        )
        report = write_characterisation(archive, tmp_path / name, formats=("png",))
        json.dumps(report, allow_nan=False)
        assert len(report["figures"]) == 4
        if name == "zero":
            assert report["summary"]["velocity"] is None
        else:
            assert report["summary"]["velocity"]["signed_m_per_s"] == pytest.approx(200, rel=0.01)


def test_packet_figures_have_only_panel_content_and_fixed_width(tmp_path, monkeypatch):
    import matplotlib

    from wavesimfem.analysis import _rc
    from wavesimfem.characterisation_plots import render_characterisation

    captured = []
    monkeypatch.setattr(
        "wavesimfem.characterisation_plots._save",
        lambda fig, *_args: captured.append(fig) or [],
    )
    report, arrays = measure_packets(*packet())
    with matplotlib.rc_context(_rc(False)):
        render_characterisation(report, arrays, tmp_path, ("png",))
    assert len(captured) == 4
    for figure in captured:
        assert figure.get_figwidth() * 25.4 == pytest.approx(190)
        assert not figure.texts
        assert not any(axis.texts for axis in figure.axes)
        assert all(axis.title.get_fontsize() == 10 for axis in figure.axes)
    # Four receiver panels use two columns at the publication width.
    assert len(captured[2].axes) == len(captured[3].axes) == 5
    assert report["figure_parameters"]["font_size_pt"] == 10


def test_pressure_characterisation_defaults_component_and_tracks_usetex(tmp_path, monkeypatch):
    t, x, _, sig = packet()
    archive = tmp_path / "pressure.npz"
    np.savez(
        archive,
        times=t,
        sensor_x=x,
        sensor_data_pressure=sig,
        sensor_observable="pressure",
        sensor_units="Pa",
    )
    opts = validate_options({"enabled": True})
    assert opts is not None and opts["component"] is None
    report = write_characterisation(archive, tmp_path / "auto_out", options=opts, formats=("png",))
    assert report["options"]["component"] == "pressure"
    assert report["observable"] == "pressure"
    assert report["summary"]["velocity"]["signed_m_per_s"] == pytest.approx(200, rel=0.02)

    monkeypatch.setattr("wavesimfem.analysis.latex_available", lambda: True)
    monkeypatch.setattr("wavesimfem.characterisation_plots._save", lambda *args: [])
    write_characterisation(archive, tmp_path / "tex_out", formats=("png",), usetex=True)
    readme_tex = (tmp_path / "tex_out" / "characterisation" / "README.md").read_text()
    assert "190 mm width with 10 pt external LaTeX" in readme_tex


def test_measure_packets_tolerates_ulp_rounding_on_time_window():
    t, x, ux, uy = packet()
    t_shifted = t + 1e-4
    eps = np.finfo(float).eps * abs(t_shifted[-1])
    result, _ = measure_packets(
        t_shifted,
        x,
        ux,
        uy,
        {
            "time_window": [t_shifted[0] - 2 * eps, t_shifted[-1] + 2 * eps],
            "frequency_band": [1000.0, 7000.0],
        },
    )
    assert result["summary"]["velocity"]["signed_m_per_s"] == pytest.approx(200, rel=0.02)
    with pytest.raises(ValueError, match="time_window"):
        measure_packets(t_shifted, x, ux, uy, {"time_window": [t_shifted[0], t_shifted[-1] + 1e-3]})
