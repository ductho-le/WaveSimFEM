"""Sensor-free and ordinary streaming use the same sampling clock."""

import numpy as np
import pytest
import scipy.sparse as sp

from guidedwavefem.solvers_gpu import _NumpyLikeBackend, integrate_explicit_rk4


def _system(dtype="float64", sensors=False):
    return dict(
        stiffness=sp.csr_matrix((2, 2)),
        damping=sp.csr_matrix((2, 2)),
        inv_mass=np.array([1.0, 2.0]),
        load_shape=np.array([1.0, -3.0]),
        sensor_matrix=sp.eye(2, format="csr") if sensors else sp.csr_matrix((0, 2)),
        dt=0.01,
        n_steps=11,
        source=lambda t: 1.0,
        backend=_NumpyLikeBackend(np.dtype(dtype)),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("retain_history", [False, True])
@pytest.mark.parametrize("sample_every,chunk_bytes", [(1, 24), (2, 24), (3, 1), (20, 24)])
def test_empty_sensor_stream_preserves_sample_clock(
    dtype, retain_history, sample_every, chunk_bytes
):
    chunks = []
    result = integrate_explicit_rk4(
        **_system(dtype),
        sample_every=sample_every,
        history_chunk_bytes=chunk_bytes,
        history_callback=lambda times, values: chunks.append((times, values)),
        retain_history=retain_history,
    )

    expected_times = np.arange(11 // sample_every + 1) * (sample_every * 0.01)
    np.testing.assert_array_equal(result.times, expected_times)
    np.testing.assert_array_equal(np.concatenate([times for times, _ in chunks]), result.times)
    assert chunks[0][0].tolist() == [0.0]
    max_chunk_rows = max(1, chunk_bytes // result.times.dtype.itemsize)
    for times, values in chunks:
        assert 1 <= len(times) <= max_chunk_rows
        assert values.shape == (len(times), 0)
        assert values.dtype == np.dtype(dtype)
    rows = len(expected_times) if retain_history else 0
    assert result.sensor_data_x.shape == result.sensor_data_y.shape == (rows, 0)
    assert result.sensor_data_x.dtype == result.sensor_data_y.dtype == np.dtype(dtype)


def test_empty_sensor_stream_flushes_final_partial_chunk():
    chunk_lengths = []
    integrate_explicit_rk4(
        **_system(),
        sample_every=2,
        history_chunk_bytes=24,
        history_callback=lambda times, values: chunk_lengths.append(len(times)),
    )
    assert chunk_lengths == [1, 3, 2]


def test_empty_sensor_stream_avoids_empty_device_operator_and_transfers():
    class NoEmptySensorBackend(_NumpyLikeBackend):
        def csr(self, matrix):
            assert matrix.shape[0] > 0
            return super().csr(matrix)

        def to_host(self, array):
            raise AssertionError("Empty sensor streaming needs no device transfer")

    kwargs = _system()
    kwargs["backend"] = NoEmptySensorBackend(np.dtype("float64"))
    chunks = []
    result = integrate_explicit_rk4(
        **kwargs,
        history_callback=lambda times, values: chunks.append(times),
        retain_history=False,
    )
    np.testing.assert_array_equal(np.concatenate(chunks), result.times)


@pytest.mark.parametrize("retain_history", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_nonempty_sensor_stream_keeps_values_and_buffer_budget(dtype, retain_history):
    chunks = []
    chunk_bytes = 4 * np.dtype(dtype).itemsize  # Two samples of one x/y sensor.
    result = integrate_explicit_rk4(
        **_system(dtype, sensors=True),
        sample_every=2,
        history_chunk_bytes=chunk_bytes,
        history_callback=lambda times, values: chunks.append((times, values)),
        retain_history=retain_history,
    )
    times = np.concatenate([t for t, _ in chunks])
    values = np.concatenate([v for _, v in chunks])
    np.testing.assert_array_equal(times, result.times)
    expected = times[:, None] ** 2 * np.array([0.5, -3.0])
    np.testing.assert_allclose(values, expected, rtol=1e-6, atol=1e-9)
    assert [len(t) for t, _ in chunks] == [1, 2, 2, 1]
    assert all(v.nbytes <= chunk_bytes for _, v in chunks)
    if retain_history:
        np.testing.assert_array_equal(result.sensor_data_x, values[:, 0::2])
        np.testing.assert_array_equal(result.sensor_data_y, values[:, 1::2])
    else:
        assert result.sensor_data_x.shape == result.sensor_data_y.shape == (0, 1)


@pytest.mark.parametrize("failure_call", [1, 2])
def test_empty_sensor_stream_propagates_callback_failure(failure_call):
    calls = 0

    def fail(times, values):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError("History sink unavailable")

    with pytest.raises(RuntimeError, match="History sink unavailable"):
        integrate_explicit_rk4(**_system(), history_chunk_bytes=8, history_callback=fail)
    assert calls == failure_call


def test_empty_sensor_integration_requires_positive_step_count():
    kwargs = _system()
    kwargs["n_steps"] = 0
    with pytest.raises(ValueError, match="n_steps must be an integer >= 1"):
        integrate_explicit_rk4(**kwargs)


def test_empty_sensor_history_without_callback():
    result = integrate_explicit_rk4(**_system(), sample_every=2, history_chunk_bytes=8)
    np.testing.assert_array_equal(result.times, np.arange(6) * 0.02)
    assert result.sensor_data_x.shape == result.sensor_data_y.shape == (6, 0)
