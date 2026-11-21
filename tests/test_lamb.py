"""Independent limits and synthetic travelling-wave reconstruction checks."""

import json

import numpy as np
import pytest
import yaml
from matplotlib.text import Text

from wavesimfem.lamb import s0_wavenumber, separate_fundamentals, write_lamb_report
from wavesimfem.materials import a0_wavenumber, compute_rayleigh_speed


def test_s0_thin_plate_and_rayleigh_limits():
    cl, ct, h = 6200.0, 3120.0, 0.003
    extensional = 2 * ct * np.sqrt(1 - (ct / cl) ** 2)
    assert 2 * np.pi * 100 / s0_wavenumber(100, h, cl, ct) == pytest.approx(extensional, rel=1e-5)
    cr = compute_rayleigh_speed(ct, c_L=cl)
    assert 2 * np.pi * 1e7 / s0_wavenumber(1e7, h, cl, ct) == pytest.approx(cr, rel=0.002)


def test_s0_satisfies_independent_complex_tangent_equation():
    for f in (1e4, 1e5, 5e5, 1e6):
        cl, ct, h = 6200.0, 3120.0, 0.003
        k = s0_wavenumber(f, h, cl, ct)
        w = 2 * np.pi * f
        p, q = np.sqrt((w / cl) ** 2 - k * k + 0j), np.sqrt((w / ct) ** 2 - k * k + 0j)
        lhs = np.tan(q * h / 2) / np.tan(p * h / 2)
        rhs = -4 * k * k * p * q / (q * q - k * k) ** 2
        assert abs(lhs - rhs) / max(abs(lhs), abs(rhs)) < 1e-6


def test_modal_reconstruction_and_selectivity():
    t = np.arange(512) * 1e-7
    x = np.arange(401) * 0.0005
    f = 195312.5
    ka = a0_wavenumber(f, 0.003, 6200, 3120)
    ks = s0_wavenumber(f, 0.003, 6200, 3120)
    a = np.cos(2 * np.pi * f * t[:, None] - ka * x)
    s = np.cos(2 * np.pi * f * t[:, None] - ks * x)
    r = separate_fundamentals(t, x, a + s, 0.003, 6200, 3120, [1e5, 3e5])
    np.testing.assert_allclose(r["A0"] + r["S0"] + r["residual"], r["tapered"], atol=1e-14)
    assert np.corrcoef(r["A0"].ravel(), a.ravel())[0, 1] > 0.9
    assert np.corrcoef(r["S0"].ravel(), s.ravel())[0, 1] > 0.9


def test_nearly_coincident_branches_are_unassigned():
    t = np.arange(256) * 1e-8
    x = np.arange(64) * 1e-4
    r = separate_fundamentals(t, x, np.ones((256, 64)), 0.003, 6200, 3120, [8e6, 12e6])
    assert not r["curves"][:, 3].any()
    assert not r["A0"].any()
    assert not r["S0"].any()


def test_counterpropagating_modes_are_both_retained():
    t = np.arange(512) * 1e-7
    x = np.arange(401) * 0.0005
    f = 195312.5
    k = a0_wavenumber(f, 0.003, 6200, 3120)
    forward = np.cos(2 * np.pi * f * t[:, None] - k * x)
    backward = np.cos(2 * np.pi * f * t[:, None] + k * x)
    for wave in (forward, backward):
        result = separate_fundamentals(t, x, wave, 0.003, 6200, 3120, [1e5, 3e5])
        assert np.corrcoef(result["A0"].ravel(), wave.ravel())[0, 1] > 0.9


def _report_archive(tmp_path, *, scale=1, zero=False, unresolved=False):
    """The same dimensional plate experiment at ultrasonic or metre scales."""
    t = np.arange(256) * 2e-7 * scale
    x = np.arange(129) * 1e-3 * scale
    if unresolved:
        x *= 0.03
    f = 195312.5 / scale
    k = a0_wavenumber(f, 0.003 * scale, 6200, 3120)
    signal = 1e-9 * np.cos(2 * np.pi * f * t[:, None] - k * x)
    if zero:
        signal *= 0
    cfg = {
        "structure_mode": "1_layer",
        "single_layer": {"thickness": 0.003 * scale, "rho": 2700, "E": 70e9, "nu": 0.33},
        "f_center": 2e5 / scale,
        "min_frequency": 1e5 / scale,
        "max_frequency": 3e5 / scale,
    }
    archive = tmp_path / "sensor_data.npz"
    np.savez(
        archive,
        times=t,
        sensor_x=x,
        sensor_data_x=0.3 * signal,
        sensor_data_y=signal,
        c_L=6200.0,
        c_T=3120.0,
        source_x=0.0,
        simulation_config_yaml=yaml.safe_dump(cfg),
        provenance="Synthetic analytical travelling wave; not a FEM solve",
    )
    return archive, t, x


def test_report_contains_sampling_and_reconstruction_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("wavesimfem.analysis._save", lambda *args: [])
    archive, t, x = _report_archive(tmp_path)
    report = write_lamb_report(archive, formats=())
    assert report["schema_version"] == 2
    assert report["usable_frequency_bins"] > 0
    assert report["usable_frequency_intervals_hz"]
    assert report["frequency_bin_spacing_hz"] == pytest.approx(1 / (len(t) * (t[1] - t[0])))
    assert report["wavenumber_bin_spacing_rad_m"] == pytest.approx(
        2 * np.pi / (len(x) * (x[1] - x[0]))
    )
    for component in ("x", "y"):
        assert report["components"][component]["reconstruction_relative_to_input_peak"] < 1e-14
    with np.load(tmp_path / "lamb_modes" / "modal_signals.npz") as saved:
        for suffix in ("", "_x"):
            np.testing.assert_allclose(
                saved["A0" + suffix] + saved["S0" + suffix] + saved["residual" + suffix],
                saved["tapered" + suffix],
                atol=1e-23,
            )
    persisted = json.loads((tmp_path / "lamb_modes" / "diagnostics.json").read_text())
    assert persisted == report
    readme = (tmp_path / "lamb_modes" / "README.md").read_text()
    assert "Synthetic analytical travelling wave; not a FEM solve" in readme
    assert "190 mm" in readme
    assert "10 pt built-in serif/mathtext" in readme
    assert "not mechanical energy" in readme
    assert "99th absolute percentile" in readme
    assert "edge ringing" in readme
    assert "Maximum reconstruction error" in readme


@pytest.mark.parametrize("zero,unresolved", [(True, False), (False, True)])
def test_report_distinguishes_missing_signal_and_unresolved_modes(
    tmp_path, monkeypatch, zero, unresolved
):
    monkeypatch.setattr("wavesimfem.analysis._save", lambda *args: [])
    archive, _, _ = _report_archive(tmp_path, zero=zero, unresolved=unresolved)
    report = write_lamb_report(archive, formats=())
    assert report["warnings"]
    assert report["status"] == ("no_vertical_signal" if zero else "unresolved")
    if zero:
        assert report["components"]["y"]["rms_relative_to_input"]["residual"] is None
    else:
        assert report["usable_frequency_intervals_hz"] == []
        assert report["components"]["y"]["rms_relative_to_input"]["residual"] == pytest.approx(1)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("scale,unit", [(1, "rad/mm"), (1000, "rad/m")])
def test_report_displays_both_directions_in_scale_appropriate_units(
    tmp_path, monkeypatch, scale, unit
):
    figures = {}

    def capture(fig, base, formats, dpi):
        figures[base.rsplit("/", 1)[-1]] = fig
        return []

    monkeypatch.setattr("wavesimfem.analysis._save", capture)
    archive, _, _ = _report_archive(tmp_path, scale=scale)
    write_lamb_report(archive, formats=())
    spectrum = figures["modal_resolution"].axes[0]
    lower, upper = spectrum.get_xlim()
    assert lower < 0 < upper
    assert unit in spectrum.get_xlabel()
    limits = [ax.collections[0].get_clim() for ax in figures["modal_separation"].axes[:4]]
    assert limits == [limits[0]] * 4


def test_modal_figures_use_only_publication_panels(tmp_path, monkeypatch):
    figures = []

    def capture(fig, *args):
        figures.append(fig)
        return []

    monkeypatch.setattr("wavesimfem.analysis._save", capture)
    archive, _, _ = _report_archive(tmp_path)
    write_lamb_report(archive, formats=())
    assert len(figures) == 3
    for fig in figures:
        assert fig.get_figwidth() * 25.4 == pytest.approx(190)
        assert not fig.texts
        for ax in fig.axes:
            assert not ax.texts
            title = ax.get_title(loc="left") or ax.get_title()
            if title:
                assert title.startswith(("(a)", "(b)", "(c)", "(d)"))
        for artist in fig.findobj(match=Text):
            if artist.get_text():
                assert not artist.get_usetex()
                assert artist.get_fontsize() == 10


@pytest.mark.parametrize("flat_override", [False, True])
def test_report_accepts_sectioned_config_and_unit_strings(tmp_path, monkeypatch, flat_override):
    monkeypatch.setattr("wavesimfem.analysis._save", lambda *args: [])
    archive, _, _ = _report_archive(tmp_path)
    sectioned = {
        "structure": {
            "mode": "1_layer",
            "single_layer": {"thickness": 0.003, "rho": 2700, "E": 70e9, "nu": 0.33},
        },
        "excitation": {"f_center": "200 kHz"},
        "mesh": {"min_frequency": "100 kHz", "max_frequency": "300 kHz"},
    }
    if flat_override:
        sectioned.update(min_frequency="120 kHz", max_frequency="280 kHz")
    with np.load(archive) as orig:
        payload = {k: orig[k] for k in orig.files}
    payload["simulation_config_yaml"] = yaml.safe_dump(sectioned)
    sectioned_archive = tmp_path / "sectioned_sensor_data.npz"
    np.savez(sectioned_archive, **payload)

    report = write_lamb_report(sectioned_archive, output_dir=tmp_path / "sec_out", formats=())
    assert report["band_hz"] == ([1.2e5, 2.8e5] if flat_override else [1e5, 3e5])
    assert report["usable_frequency_bins"] > 0


def test_s0_wavenumber_caches_and_readme_records_usetex(tmp_path, monkeypatch):
    from wavesimfem.lamb import _cached_s0_wavenumber

    _cached_s0_wavenumber.cache_clear()
    first = s0_wavenumber(2e5, 0.003, 6200.0, 3120.0)
    second = s0_wavenumber(2e5, 0.003, 6200.0, 3120.0)
    assert first == second
    assert _cached_s0_wavenumber.cache_info().hits >= 1

    monkeypatch.setattr("wavesimfem.analysis.latex_available", lambda: True)
    monkeypatch.setattr("wavesimfem.analysis._save", lambda *args: [])
    archive, _, _ = _report_archive(tmp_path)
    write_lamb_report(archive, output_dir=tmp_path / "tex_out", formats=(), usetex=True)
    readme = (tmp_path / "tex_out" / "lamb_modes" / "README.md").read_text()
    assert "10 pt external LaTeX" in readme


def test_lamb_report_handles_low_max_frequency_and_omitted_archive_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("wavesimfem.analysis._save", lambda *args: [])
    archive, _, _ = _report_archive(tmp_path)
    with np.load(archive) as orig:
        payload = {k: orig[k] for k in orig.files if k not in ("c_L", "c_T", "source_x")}
    cfg = {
        "structure": {
            "mode": "1_layer",
            "single_layer": {"thickness": 0.003, "rho": 2700, "E": 70e9, "nu": 0.33},
        },
        "excitation": {"f_center": "500 kHz"},
        # Intentionally below f_center / 5 = 100 kHz with min_frequency unset.
        "mesh": {"max_frequency": "80 kHz"},
    }
    payload["simulation_config_yaml"] = yaml.safe_dump(cfg)
    low_band_archive = tmp_path / "low_band.npz"
    np.savez(low_band_archive, **payload)

    report = write_lamb_report(low_band_archive, output_dir=tmp_path / "low_out", formats=())
    assert report["band_hz"] == [4e4, 8e4]
    assert report["frequency_bins"] > 0
    with np.load(tmp_path / "low_out" / "lamb_modes" / "modal_signals.npz") as signals:
        assert np.isfinite(signals["curves"]).all()
        assert (signals["curves"][:, 1:3] > 0).all()
