"""Tests for nengo_dl.signals.SignalDict."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import nengo
from nengo.builder.signal import Signal

import nengo_dl
from nengo_dl.signals import SignalDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(shape, name="sig", initial_value=None):
    if initial_value is None:
        initial_value = np.zeros(shape)
    return Signal(initial_value, name=name, shape=shape)


def _make_signal_dict(minibatch_size=2):
    return SignalDict(minibatch_size=minibatch_size,
                      device=torch.device("cpu"),
                      dtype=torch.float32)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestAddSignal:
    def test_base_signal_registered(self):
        sd = _make_signal_dict()
        sig = _make_signal((3,))
        sd.add_signal(sig)
        assert sig in sd

    def test_float_dtype(self):
        sd = _make_signal_dict()
        sig = _make_signal((4,), initial_value=np.ones(4, dtype=np.float32))
        sd.add_signal(sig)
        assert sd._data[id(sig)].dtype == torch.float32

    def test_int_dtype(self):
        sd = _make_signal_dict()
        sig = _make_signal((3,), initial_value=np.array([1, 2, 3], dtype=np.int32))
        sd.add_signal(sig)
        assert sd._data[id(sig)].dtype == torch.int64

    def test_trainable_stores_parameter(self):
        sd = _make_signal_dict()
        sig = _make_signal((5,), initial_value=np.ones(5))
        sd.add_signal(sig, trainable=True)
        assert isinstance(sd._data[id(sig)], nn.Parameter)
        assert sd._data[id(sig)].requires_grad

    def test_state_has_batch_dim(self):
        sd = _make_signal_dict(minibatch_size=4)
        sig = _make_signal((3,))
        sd.add_signal(sig, trainable=False)
        assert sd._data[id(sig)].shape == (4, 3)

    def test_param_has_no_batch_dim(self):
        sd = _make_signal_dict(minibatch_size=4)
        sig = _make_signal((3,), initial_value=np.ones(3))
        sd.add_signal(sig, trainable=True)
        assert sd._data[id(sig)].shape == (3,)

    def test_duplicate_add_ignored(self):
        sd = _make_signal_dict()
        sig = _make_signal((3,))
        sd.add_signal(sig)
        sd.add_signal(sig)  # second call must not raise
        assert len(sd._data) == 1

    def test_view_signal_rejected(self):
        base = Signal(np.zeros(6), name="base", shape=(6,))
        view = base[2:5]
        sd = _make_signal_dict()
        with pytest.raises(AssertionError):
            sd.add_signal(view)


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------

class TestGather:
    def test_gather_base_state(self):
        sd = _make_signal_dict(minibatch_size=3)
        data = np.arange(4, dtype=np.float32)
        sig = _make_signal((4,), initial_value=data)
        sd.add_signal(sig)
        out = sd.gather(sig)
        assert out.shape == (3, 4)
        np.testing.assert_allclose(out[0].numpy(), data)

    def test_gather_base_param(self):
        sd = _make_signal_dict()
        data = np.array([1.0, 2.0, 3.0])
        sig = _make_signal((3,), initial_value=data)
        sd.add_signal(sig, trainable=True)
        out = sd.gather(sig)
        assert out.shape == (3,)
        np.testing.assert_allclose(out.detach().numpy(), data)

    def test_gather_view_state(self):
        sd = _make_signal_dict(minibatch_size=2)
        data = np.arange(6, dtype=np.float32)
        base = Signal(data.copy(), name="base", shape=(6,))
        sd.add_signal(base)
        view = base[2:5]  # elemoffset=2, size=3
        out = sd.gather(view)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out[0].numpy(), data[2:5])

    def test_gather_view_param(self):
        sd = _make_signal_dict()
        data = np.arange(6, dtype=np.float32)
        base = Signal(data.copy(), name="base", shape=(6,))
        sd.add_signal(base, trainable=True)
        view = base[1:4]
        out = sd.gather(view)
        assert out.shape == (3,)
        np.testing.assert_allclose(out.detach().numpy(), data[1:4])


# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------

class TestScatter:
    def test_scatter_set_state(self):
        sd = _make_signal_dict(minibatch_size=2)
        sig = _make_signal((3,))
        sd.add_signal(sig)
        val = torch.ones(2, 3)
        sd.scatter(sig, val, mode="set")
        out = sd.gather(sig)
        np.testing.assert_allclose(out.numpy(), val.numpy())

    def test_scatter_inc_state(self):
        sd = _make_signal_dict(minibatch_size=2)
        sig = _make_signal((3,), initial_value=np.ones(3))
        sd.add_signal(sig)
        val = torch.ones(2, 3)
        sd.scatter(sig, val, mode="inc")
        out = sd.gather(sig)
        np.testing.assert_allclose(out.numpy(), np.full((2, 3), 2.0))

    def test_scatter_set_param(self):
        sd = _make_signal_dict()
        sig = _make_signal((3,), initial_value=np.zeros(3))
        sd.add_signal(sig, trainable=True)
        val = torch.tensor([1.0, 2.0, 3.0])
        sd.scatter(sig, val, mode="set")
        out = sd.gather(sig)
        np.testing.assert_allclose(out.detach().numpy(), [1.0, 2.0, 3.0])

    def test_scatter_view_set_state(self):
        sd = _make_signal_dict(minibatch_size=2)
        data = np.zeros(6, dtype=np.float32)
        base = Signal(data.copy(), name="base", shape=(6,))
        sd.add_signal(base)
        view = base[2:5]
        val = torch.ones(2, 3)
        sd.scatter(view, val, mode="set")
        out = sd.gather(base)
        np.testing.assert_allclose(out[0, 2:5].numpy(), [1.0, 1.0, 1.0])
        np.testing.assert_allclose(out[0, :2].numpy(), [0.0, 0.0])

    def test_scatter_view_inc_state(self):
        sd = _make_signal_dict(minibatch_size=2)
        data = np.ones(6, dtype=np.float32)
        base = Signal(data.copy(), name="base", shape=(6,))
        sd.add_signal(base)
        view = base[0:3]
        val = torch.ones(2, 3) * 2.0
        sd.scatter(view, val, mode="inc")
        out = sd.gather(base)
        np.testing.assert_allclose(out[0, :3].numpy(), [3.0, 3.0, 3.0])
        np.testing.assert_allclose(out[0, 3:].numpy(), [1.0, 1.0, 1.0])

    def test_scatter_broadcasts_no_batch_dim(self):
        sd = _make_signal_dict(minibatch_size=4)
        sig = _make_signal((3,))
        sd.add_signal(sig)
        val = torch.tensor([7.0, 8.0, 9.0])  # no batch dim
        sd.scatter(sig, val, mode="set")
        out = sd.gather(sig)
        assert out.shape == (4, 3)
        np.testing.assert_allclose(out[2].numpy(), [7.0, 8.0, 9.0])


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_restores_state(self):
        sd = _make_signal_dict(minibatch_size=2)
        init = np.array([5.0, 6.0, 7.0])
        sig = _make_signal((3,), initial_value=init)
        sd.add_signal(sig)
        sd.scatter(sig, torch.zeros(2, 3), mode="set")
        sd.reset()
        out = sd.gather(sig)
        np.testing.assert_allclose(out[0].numpy(), init)

    def test_reset_does_not_change_params(self):
        sd = _make_signal_dict()
        init = np.array([1.0, 2.0, 3.0])
        sig = _make_signal((3,), initial_value=init)
        sd.add_signal(sig, trainable=True)
        trained = torch.tensor([9.0, 9.0, 9.0])
        sd.scatter(sig, trained, mode="set")
        sd.reset()
        out = sd.gather(sig)
        np.testing.assert_allclose(out.detach().numpy(), trained.numpy())


# ---------------------------------------------------------------------------
# Parameter access
# ---------------------------------------------------------------------------

class TestParameters:
    def test_get_parameter_returns_param(self):
        sd = _make_signal_dict()
        sig = _make_signal((4,), initial_value=np.ones(4))
        sd.add_signal(sig, trainable=True)
        param = sd.get_parameter(sig)
        assert isinstance(param, nn.Parameter)

    def test_get_parameter_returns_none_for_state(self):
        sd = _make_signal_dict()
        sig = _make_signal((4,))
        sd.add_signal(sig, trainable=False)
        assert sd.get_parameter(sig) is None

    def test_get_all_parameters_stable_keys(self):
        """Key order must match insertion order, not memory addresses."""
        with nengo.Network(seed=0) as net:
            ens = nengo.Ensemble(10, 1, seed=0)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim1:
            weights1 = sim1.tensor_graph.signals.get_all_parameters()
        with nengo_dl.Simulator(net, seed=0) as sim2:
            weights2 = sim2.tensor_graph.signals.get_all_parameters()

        assert set(weights1.keys()) == set(weights2.keys())
        for k in weights1:
            assert k.startswith("param_")

    def test_get_all_parameters_count(self):
        sd = _make_signal_dict()
        for i in range(3):
            sig = _make_signal((2,), name=f"p{i}", initial_value=np.ones(2) * i)
            sd.add_signal(sig, trainable=True)
        for i in range(2):
            sig = _make_signal((2,), name=f"s{i}")
            sd.add_signal(sig, trainable=False)
        params = sd.get_all_parameters()
        assert len(params) == 3
        assert list(params.keys()) == ["param_0000", "param_0001", "param_0002"]


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

class TestMisc:
    def test_contains_true(self):
        sd = _make_signal_dict()
        sig = _make_signal((3,))
        sd.add_signal(sig)
        assert sig in sd

    def test_contains_false(self):
        sd = _make_signal_dict()
        sig = _make_signal((3,))
        assert sig not in sd

    def test_repr(self):
        sd = _make_signal_dict(minibatch_size=5)
        for i in range(2):
            s = _make_signal((1,), name=f"p{i}", initial_value=np.ones(1))
            sd.add_signal(s, trainable=True)
        for i in range(3):
            s = _make_signal((1,), name=f"s{i}")
            sd.add_signal(s, trainable=False)
        r = repr(sd)
        assert "params=2" in r
        assert "state=3" in r
        assert "batch=5" in r
