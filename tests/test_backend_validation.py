"""Independent, hardware-selectable checks for the shared array integrator.

Set GWFEM_REQUIRE_CUDA_BACKENDS=cupy on a CUDA validation machine.
Requested backends must actually run; missing libraries/devices then fail.
"""

from __future__ import annotations

import importlib
import os
from functools import lru_cache

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp

from guidedwavefem.device import DeviceInfo
from guidedwavefem.solvers_gpu import integrate_explicit_rk4, make_backend

BACKENDS = [
    "scipy",
    pytest.param("cupy", marks=pytest.mark.gpu),
]


def _backend(name, precision):
    required = set(filter(None, os.environ.get("GWFEM_REQUIRE_CUDA_BACKENDS", "").split(",")))
    if required - {"cupy"}:
        pytest.fail("GWFEM_REQUIRE_CUDA_BACKENDS must contain only cupy")
    library = name
    cuda = name == "cupy"
    if cuda and required and library not in required:
        pytest.skip(f"{library} was not selected for this hardware run")
    if library != "scipy":
        try:
            module = importlib.import_module(library)
            if cuda:
                available = module.cuda.runtime.getDeviceCount() > 0
                if not available:
                    raise RuntimeError("no usable CUDA device")
        except (ImportError, RuntimeError) as exc:
            if cuda and library in required:
                pytest.fail(f"Required CUDA backend {library} unavailable: {exc}")
            pytest.skip(f"Optional backend {name} unavailable: {exc}")
    return make_backend(DeviceInfo(library, "cuda:0" if cuda else "cpu", name), precision=precision)


def _system():
    # Non-diagonal damping, non-uniform mass, and mixed sensor interpolation
    # exercise sparse products and both components beyond a diagonal oscillator.
    n = 8
    stiffness = sp.diags([-8.0, 30.0, -8.0], [-1, 0, 1], shape=(n, n), format="csr")
    damping = sp.diags([-0.1, 0.6, -0.1], [-1, 0, 1], shape=(n, n), format="csr")
    inv_mass = np.linspace(0.8, 1.4, n)
    load_shape = np.linspace(-0.8, 1.1, n)
    sensors = sp.csr_matrix(
        np.array(
            [
                [0.3, 0, 0.7, 0, 0, 0, 0, 0],
                [0, 0.3, 0, 0.7, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.8, 0, 0.2, 0],
                [0, 0, 0, 0, 0, 0.8, 0, 0.2],
            ]
        )
    )
    return stiffness, damping, inv_mass, load_shape, sensors


@lru_cache(maxsize=3)
def _exact_samples(sample_every):
    """Matrix exponential of the affine first-order ODE; independent of RK4."""
    stiffness, damping, inv_mass, load_shape, sensors = _system()
    n = len(inv_mass)
    generator = np.zeros((2 * n + 1, 2 * n + 1))
    generator[:n, n : 2 * n] = np.eye(n)
    generator[n : 2 * n, :n] = -inv_mass[:, None] * stiffness.toarray()
    generator[n : 2 * n, n : 2 * n] = -inv_mass[:, None] * damping.toarray()
    generator[n : 2 * n, -1] = inv_mass * load_shape
    times = np.arange(37 // sample_every + 1) * (sample_every * 0.002)
    history = np.stack([sensors @ scipy.linalg.expm(t * generator)[:n, -1] for t in times])
    return times, history


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("sample_every", [1, 4, 50])
@pytest.mark.parametrize("retain_history", [False, True])
def test_backends_match_exact_solution_with_chunked_streaming(
    name, precision, sample_every, retain_history
):
    backend = _backend(name, precision)
    chunks = []
    # A two-row buffer forces complete and partial flushes, including a budget
    # not divisible by a sample's size. The initial sample is a separate chunk.
    chunk_bytes = 2 * 4 * backend.dtype.itemsize + 1
    result = integrate_explicit_rk4(
        *_system(),
        dt=0.002,
        n_steps=37,
        source=lambda t: 1.0,
        backend=backend,
        sample_every=sample_every,
        history_chunk_bytes=chunk_bytes,
        history_callback=lambda times, values: chunks.append((times, values)),
        retain_history=retain_history,
    )
    expected_times, expected = _exact_samples(sample_every)
    values = np.concatenate([values for _, values in chunks])
    np.testing.assert_array_equal(result.times, expected_times)
    np.testing.assert_array_equal(np.concatenate([times for times, _ in chunks]), expected_times)
    assert values.dtype == backend.dtype
    assert all(1 <= len(times) <= 2 and values.nbytes <= chunk_bytes for times, values in chunks)
    np.testing.assert_allclose(
        values,
        expected,
        rtol=3e-6 if precision == "single" else 2e-8,
        atol=2e-10 if precision == "single" else 2e-12,
    )
    if retain_history:
        np.testing.assert_array_equal(result.sensor_data_x, values[:, 0::2])
        np.testing.assert_array_equal(result.sensor_data_y, values[:, 1::2])
    else:
        assert result.sensor_data_x.shape == result.sensor_data_y.shape == (0, 2)
    # Chunks remain valid after subsequent flushes; callbacks may retain their arrays.
    assert chunks[0][1].shape == (1, 4)
    assert not chunks[0][1].any()


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("failure_call", [1, 3])
def test_backends_propagate_history_sink_failure(name, failure_call):
    backend = _backend(name, "double")
    calls = 0

    def sink(times, values):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("history sink full")

    with pytest.raises(OSError, match="history sink full"):
        integrate_explicit_rk4(
            *_system(),
            dt=0.002,
            n_steps=9,
            source=lambda t: 1.0,
            backend=backend,
            history_chunk_bytes=4 * 8,
            history_callback=sink,
            retain_history=False,
        )
    assert calls == failure_call


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("precision", ["single", "double"])
def test_backends_reject_nonfinite_state(name, precision):
    with pytest.raises(FloatingPointError, match="Non-finite state"):
        integrate_explicit_rk4(
            *_system(),
            dt=0.002,
            n_steps=2,
            source=lambda t: float("nan"),
            backend=_backend(name, precision),
            report_interval=1,
        )
