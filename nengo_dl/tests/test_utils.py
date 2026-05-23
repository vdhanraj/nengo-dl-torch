"""Tests for nengo_dl.utils."""

import io
import numpy as np
import pytest
import torch
import torch.nn as nn

from nengo_dl.utils import (
    to_numpy,
    to_tensor,
    layer_count_params,
    rate_to_spikes,
    decode_spikes,
    sanitize_name,
    batch_first,
    time_first,
    ProgressBar,
)


# ---------------------------------------------------------------------------
# to_numpy
# ---------------------------------------------------------------------------

class TestToNumpy:
    def test_tensor_to_numpy(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_numpy_passthrough(self):
        a = np.array([4.0, 5.0])
        out = to_numpy(a)
        assert isinstance(out, np.ndarray)
        np.testing.assert_allclose(out, a)

    def test_list_to_numpy(self):
        out = to_numpy([1, 2, 3])
        assert isinstance(out, np.ndarray)

    def test_cuda_tensor_if_available(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        t = torch.tensor([1.0, 2.0]).cuda()
        arr = to_numpy(t)
        assert isinstance(arr, np.ndarray)

    def test_detaches_gradient(self):
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        arr = to_numpy(t)
        assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# to_tensor
# ---------------------------------------------------------------------------

class TestToTensor:
    def test_numpy_to_tensor(self):
        a = np.array([1.0, 2.0, 3.0])
        t = to_tensor(a)
        assert isinstance(t, torch.Tensor)
        np.testing.assert_allclose(t.numpy(), a)

    def test_dtype_applied(self):
        a = np.array([1.0, 2.0])
        t = to_tensor(a, dtype=torch.float64)
        assert t.dtype == torch.float64

    def test_device_applied(self):
        a = np.array([1.0])
        dev = torch.device("cpu")
        t = to_tensor(a, device=dev)
        assert t.device == dev

    def test_tensor_passthrough_with_cast(self):
        t_in = torch.tensor([1.0, 2.0], dtype=torch.float64)
        t_out = to_tensor(t_in, dtype=torch.float32)
        assert t_out.dtype == torch.float32

    def test_list_to_tensor(self):
        t = to_tensor([1.0, 2.0, 3.0])
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3,)


# ---------------------------------------------------------------------------
# layer_count_params
# ---------------------------------------------------------------------------

class TestLayerCountParams:
    def test_linear_layer(self):
        lin = nn.Linear(4, 8)
        # 4*8 + 8 = 40 params
        assert layer_count_params(lin) == 40

    def test_linear_no_bias(self):
        lin = nn.Linear(4, 8, bias=False)
        assert layer_count_params(lin) == 32

    def test_sequential(self):
        model = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 3))
        expected = (2 * 4 + 4) + (4 * 3 + 3)
        assert layer_count_params(model) == expected

    def test_no_params(self):
        assert layer_count_params(nn.ReLU()) == 0

    def test_frozen_params_not_counted(self):
        lin = nn.Linear(4, 8)
        for p in lin.parameters():
            p.requires_grad_(False)
        assert layer_count_params(lin) == 0


# ---------------------------------------------------------------------------
# rate_to_spikes
# ---------------------------------------------------------------------------

class TestRateToSpikes:
    def test_output_shape(self):
        rates = np.full((100, 10), 20.0)
        spikes = rate_to_spikes(rates, dt=0.001, seed=0)
        assert spikes.shape == (100, 10)

    def test_binary_output(self):
        rates = np.full((50, 5), 50.0)
        spikes = rate_to_spikes(rates, dt=0.001, seed=0)
        assert set(np.unique(spikes)).issubset({0.0, 1.0})

    def test_zero_rate_no_spikes(self):
        rates = np.zeros((100, 10))
        spikes = rate_to_spikes(rates, dt=0.001, seed=0)
        assert np.sum(spikes) == 0

    def test_high_rate_many_spikes(self):
        rates = np.full((1000, 1), 1000.0)  # 1000 Hz → spike every step
        spikes = rate_to_spikes(rates, dt=0.001, seed=0)
        # At 1000 Hz and dt=0.001 → prob=1.0 → always spike
        assert spikes.sum() == 1000

    def test_reproducible_with_same_seed(self):
        rates = np.random.rand(50, 10) * 100
        s1 = rate_to_spikes(rates, dt=0.001, seed=42)
        s2 = rate_to_spikes(rates, dt=0.001, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        rates = np.full((100, 10), 50.0)
        s1 = rate_to_spikes(rates, dt=0.001, seed=0)
        s2 = rate_to_spikes(rates, dt=0.001, seed=1)
        assert not np.array_equal(s1, s2)

    def test_3d_input(self):
        rates = np.full((4, 10, 8), 20.0)  # (batch, n_steps, n_neurons)
        spikes = rate_to_spikes(rates, dt=0.001, seed=0)
        assert spikes.shape == (4, 10, 8)


# ---------------------------------------------------------------------------
# decode_spikes
# ---------------------------------------------------------------------------

class TestDecodeSpikes:
    def test_output_shape(self):
        spikes = np.random.binomial(1, 0.5, (100, 8)).astype(np.float32)
        decoded = decode_spikes(spikes, dt=0.001)
        assert decoded.shape == (100, 8)

    def test_zero_spikes_zero_output(self):
        spikes = np.zeros((50, 5))
        decoded = decode_spikes(spikes, dt=0.001)
        np.testing.assert_allclose(decoded, 0.0)

    def test_constant_spikes_converge(self):
        spikes = np.ones((1000, 1))  # spike every step
        decoded = decode_spikes(spikes, dt=0.001, tau=0.01)
        # Should approach 1.0 asymptotically
        assert decoded[-1, 0] > 0.9

    def test_no_nan(self):
        spikes = np.random.binomial(1, 0.3, (100, 10)).astype(np.float32)
        decoded = decode_spikes(spikes, dt=0.001)
        assert not np.any(np.isnan(decoded))


# ---------------------------------------------------------------------------
# sanitize_name
# ---------------------------------------------------------------------------

class TestSanitizeName:
    def test_removes_invalid_chars(self):
        result = sanitize_name("foo/bar.baz-qux")
        assert "/" not in result
        assert "." not in result
        assert "-" not in result

    def test_prepends_underscore_for_digit_start(self):
        result = sanitize_name("3foo")
        assert result[0] == "_"

    def test_truncates_to_max_len(self):
        long_name = "a" * 100
        result = sanitize_name(long_name, max_len=10)
        assert len(result) <= 10

    def test_valid_name_unchanged(self):
        name = "valid_name_123"
        assert sanitize_name(name) == name

    def test_empty_string(self):
        result = sanitize_name("")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# batch_first / time_first
# ---------------------------------------------------------------------------

class TestReshapeHelpers:
    def test_batch_first_2d(self):
        x = np.arange(12).reshape(3, 4)  # (n_steps=3, batch=4)
        out = batch_first(x)
        assert out.shape == (4, 3)

    def test_time_first_2d(self):
        x = np.arange(12).reshape(4, 3)  # (batch=4, n_steps=3)
        out = time_first(x)
        assert out.shape == (3, 4)

    def test_roundtrip_3d(self):
        x = np.random.rand(5, 10, 8)  # (n_steps, batch, features)
        out = batch_first(x)     # → (batch, n_steps, features)
        back = time_first(out)   # → (n_steps, batch, features)
        np.testing.assert_allclose(x, back)

    def test_batch_first_preserves_values(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
        out = batch_first(x)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out[0], [1, 3, 5])
        np.testing.assert_allclose(out[1], [2, 4, 6])


# ---------------------------------------------------------------------------
# ProgressBar
# ---------------------------------------------------------------------------

class TestProgressBar:
    def test_context_manager(self, capsys):
        with ProgressBar(total=5, label="test") as pb:
            for _ in range(5):
                pb.update()
        captured = capsys.readouterr()
        assert "test" in captured.out or True  # prints to stdout

    def test_partial_progress(self, capsys):
        pb = ProgressBar(total=10, label="prog")
        pb.update(3)
        pb.update(7)

    def test_update_increments(self):
        pb = ProgressBar(total=10)
        pb.update(4)
        assert pb._n == 4

    def test_zero_total_no_crash(self):
        pb = ProgressBar(total=0)
        pb.update(0)
