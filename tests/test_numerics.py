"""Independent numerical checks; run inside the fenicsx-waves environment."""
import numpy as np
import pytest
from scipy.linalg import expm
from dolfinx import fem, mesh
from mpi4py import MPI

from wave_numerics import PointSensors, dispersion_spectrum, rk4_step, validate_tensor
from guided_waves import parse_and_resolve_config, dump_config, load_config


def test_damped_rk4_convergence():
    exact = expm(np.array([[0., 1.], [-4., -1.]])) @ [1., 0.]
    errors = []
    for n in (20, 40, 80):
        u, v = np.array([1.]), np.array([0.])
        for i in range(n):
            u, v = rk4_step(u, v, i/n, 1/n, lambda u, v, t: -4*u-v)
        errors.append(np.linalg.norm(np.array([u[0], v[0]])-exact))
    assert np.all(np.log2(np.array(errors[:-1])/errors[1:]) > 3.8)


def test_exact_point_interpolation():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    space = fem.functionspace(domain, ('Lagrange', 2, (2,)))
    field = fem.Function(space)
    field.interpolate(lambda x: np.vstack([x[0]**2+2*x[1], x[0]-x[1]**2]))
    field.x.scatter_forward()
    points = np.array([[.137, .291, 0.], [.61, .43, 0.], [1., .17, 0.]])
    sensors = PointSensors(domain, points)
    values = sensors.sample(field)
    p = points[sensors.indices]
    np.testing.assert_allclose(values[:, 0], p[:, 0]**2+2*p[:, 1], atol=1e-12)
    np.testing.assert_allclose(values[:, 1], p[:, 0]-p[:, 1]**2, atol=1e-12)
    with pytest.raises(ValueError, match='outside'):
        PointSensors(domain, np.array([[2., 0., 0.]]))


def test_forward_spectrum_physical_coordinates():
    t = np.arange(256)*1e-7
    x = np.arange(64)*.0005
    frequency, wave = 625000., 2*np.pi*8/(64*.0005)
    signal = np.cos(2*np.pi*frequency*t[:, None]-wave*x[None, :])
    f, k, power = dispersion_spectrum(signal, t, x)
    fi, ki = np.unravel_index(power.argmax(), power.shape)
    assert abs(f[fi]-frequency/1e6) < f[1]-f[0]
    assert abs(k[ki]-wave/1000) < k[1]-k[0]
    assert k[-1] == pytest.approx(np.pi/.0005/1000)


@pytest.mark.parametrize('key,value', [('CFL', -1), ('f_center', 0),
                                      ('sensor_pitch', 0), ('n_sensors', 1)])
def test_invalid_config(key, value):
    cfg = {'sensing_length': .3, 'absorb_length': .1, 'f_center': 1e6, key: value}
    with pytest.raises(ValueError):
        parse_and_resolve_config(custom_cfg=cfg)


def test_material_energy():
    with pytest.raises(ValueError, match='positive'):
        validate_tensor([1, 2, 0, 1, 0, 1], 'C')
    np.testing.assert_array_equal(validate_tensor([0]*6, 'Cv', positive=False), np.zeros((3,3)))


def test_boolean_provenance(tmp_path):
    path = str(tmp_path/'config.yaml')
    dump_config({'enabled': True, 'disabled': False}, path)
    result = load_config(path)
    assert result['enabled'] is True and result['disabled'] is False


def test_plot_limits_do_not_stretch_fft(tmp_path, monkeypatch):
    import guided_waves as gw
    t, x = np.arange(256)*1e-7, np.arange(16)*.001
    signal = np.sin(2*np.pi*5e5*t[:, None]-1000*x[None, :])
    path = tmp_path/'signal.npz'
    np.savez(path, times=t, sensor_x=x, sensor_data_y=signal,
             sensor_data_x=signal, f_center=5e5, c_L=6000., c_T=3000.)
    extents = []
    def inspect_save(fig, *args, **kwargs):
        for ax in fig.axes:
            extents.extend(image.get_extent() for image in ax.images)
    monkeypatch.setattr('matplotlib.figure.Figure.savefig', inspect_save)
    gw.plot_full_analysis(str(path), str(tmp_path), k_max_disp=10.)
    assert extents[1][1] < 3.3  # Nyquist ~pi rad/mm, NOT the requested 10


def test_all_shipped_configurations():
    from pathlib import Path
    import guided_waves as gw
    for path in (Path(__file__).resolve().parents[1]/'configs').glob('*'):
        if path.suffix not in ('.yaml', '.json', '.toml'):
            continue
        cfg = gw.parse_and_resolve_config(str(path))
        gw._normalize_structure(
            mode=cfg['structure_mode'], single_layer_input=cfg.get('single_layer'),
            layers_input=cfg.get('layers'), halfspace_input=cfg.get('halfspace'),
            bottom_hs_input=cfg.get('bottom_halfspace'), top_hs_input=cfg.get('top_halfspace'))


def test_assembled_operator_bound():
    import ufl
    from scipy.linalg import eigvalsh
    from wave_numerics import operator_bound
    domain = mesh.create_unit_square(MPI.COMM_SELF, 3, 3)
    space = fem.functionspace(domain, ('Lagrange', 1))
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    stiffness = fem.petsc.assemble_matrix(fem.form(ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx))
    mass = fem.petsc.assemble_matrix(fem.form(u*v*ufl.dx))
    stiffness.assemble()
    mass.assemble()
    kd = stiffness.convert('dense')
    md = mass.convert('dense')
    lumped = md.getDenseArray().sum(axis=1)
    exact = eigvalsh(kd.getDenseArray(), np.diag(lumped))[-1]
    assert operator_bound(stiffness, lumped) >= exact*(1-1e-12)
    for matrix in (kd, md, stiffness, mass):
        matrix.destroy()


def test_package_namespace_exports():
    import guidedwavefem as gw
    assert gw.__version__ == "0.2.0"
    for name in (
        "run_simulation",
        "plot_full_analysis",
        "load_config",
        "dump_config",
        "parse_and_resolve_config",
        "preflight_estimate",
        "rk4_step",
        "operator_bound",
        "time_grid",
        "PointSensors",
        "validate_tensor",
        "dispersion_spectrum",
    ):
        assert hasattr(gw, name) and callable(getattr(gw, name))
