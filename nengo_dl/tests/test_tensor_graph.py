"""Tests for nengo_dl.tensor_graph.TensorGraph."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import nengo
import nengo_dl
from nengo_dl.tensor_graph import TensorGraph, _is_trainable_signal, topo_sort
from nengo.builder import Builder as NengoBuilder, Model as NengoModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(net, dt=0.001, minibatch_size=1, device="cpu"):
    model = NengoModel(dt=dt)
    NengoBuilder.build(model, net)
    return TensorGraph(model, dt=dt, minibatch_size=minibatch_size, device=device)


def _simple_net(seed=0):
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(np.zeros(1))
        ens = nengo.Ensemble(10, 1, seed=seed)
        nengo.Connection(inp, ens, synapse=None)
        p = nengo.Probe(ens, synapse=None)
    return net, inp, p


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestTensorGraphConstruction:
    def test_is_nn_module(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert isinstance(tg, nn.Module)

    def test_device_cpu(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert tg.device == torch.device("cpu")

    def test_dtype_float32(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert tg.dtype == torch.float32

    def test_minibatch_size_stored(self):
        net, inp, p = _simple_net()
        tg = _build(net, minibatch_size=8)
        assert tg.minibatch_size == 8

    def test_signals_populated(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert len(tg.signals._data) > 0

    def test_param_dict_is_nn_parameter_dict(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert isinstance(tg._param_dict, nn.ParameterDict)

    def test_probe_sig_map_populated(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert p in tg._probe_sig_map

    def test_input_node_sigs_populated(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert inp in tg._input_node_sigs

    def test_extra_repr(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        r = tg.extra_repr()
        assert "dt=" in r
        assert "minibatch_size=" in r


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestTensorGraphForward:
    def test_forward_returns_probe_dict(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        result = tg.forward(3)
        assert isinstance(result, dict)
        assert p in result

    def test_output_shape_batch1(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        result = tg.forward(5)
        # (batch=1, n_steps=5, probe_size=1)
        assert result[p].shape == (1, 5, 1)

    def test_output_shape_batch_n(self):
        net, inp, p = _simple_net()
        tg = _build(net, minibatch_size=4)
        result = tg.forward(5)
        assert result[p].shape == (4, 5, 1)

    def test_input_injection(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            out = nengo.Node(size_in=2)
            nengo.Connection(inp, out, synapse=None)
            p = nengo.Probe(out, synapse=None)
        tg = _build(net)
        data = np.array([[[3.0, 4.0]]])  # (batch=1, steps=1, size=2)
        result = tg.forward(1, input_data={inp: data})
        np.testing.assert_allclose(
            result[p][0, 0].cpu().numpy(), [3.0, 4.0], atol=1e-5
        )

    def test_multiple_forward_calls_independent(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        r1 = tg.forward(3)
        tg.reset_state()
        r2 = tg.forward(3)
        np.testing.assert_allclose(
            r1[p].detach().cpu().numpy(), r2[p].detach().cpu().numpy(), rtol=1e-4
        )

    def test_forward_training_flag(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        # Should not raise in either mode
        tg.forward(1, training=False)
        tg.forward(1, training=True)


# ---------------------------------------------------------------------------
# Signal collection ordering
# ---------------------------------------------------------------------------

class TestSignalOrdering:
    def test_deterministic_signal_order(self):
        """Building the same network twice gives the same param key order."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        keys1 = list(_build(net).signals.get_all_parameters().keys())
        keys2 = list(_build(net).signals.get_all_parameters().keys())
        assert keys1 == keys2

    def test_stable_param_keys(self):
        """Param keys must use stable indices, not memory addresses."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        tg = _build(net)
        for key in tg.signals.get_all_parameters():
            assert key.startswith("param_"), f"Unstable key: {key}"
            assert key[6:].isdigit(), f"Unexpected key format: {key}"


# ---------------------------------------------------------------------------
# Trainable detection
# ---------------------------------------------------------------------------

class TestTrainableDetection:
    def test_readonly_signal_can_be_trainable(self):
        from nengo.builder.signal import Signal
        sig = Signal(np.ones(5), name="w", shape=(5,))
        object.__setattr__(sig, '_readonly', True)

        class FakeModel:
            toplevel = None
            sig = {}
        assert _is_trainable_signal(sig, FakeModel()) is True

    def test_nonreadonly_is_not_trainable(self):
        from nengo.builder.signal import Signal
        sig = Signal(np.ones(5), name="v", shape=(5,))

        class FakeModel:
            toplevel = None
            sig = {}
        assert _is_trainable_signal(sig, FakeModel()) is False

    def test_scalar_constant_not_trainable(self):
        from nengo.builder.signal import Signal
        sig = Signal(np.array([0.0]), name="ZERO", shape=(1,))
        object.__setattr__(sig, '_readonly', True)

        class FakeModel:
            toplevel = None
            sig = {}
        assert _is_trainable_signal(sig, FakeModel()) is False


# ---------------------------------------------------------------------------
# Weight management
# ---------------------------------------------------------------------------

class TestWeightManagement:
    def test_get_weights_returns_dict(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        w = tg.get_weights()
        assert isinstance(w, dict)

    def test_get_weights_numpy_values(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        for k, v in tg.get_weights().items():
            assert isinstance(v, np.ndarray)

    def test_set_weights_roundtrip(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        tg1 = _build(net)
        tg2 = _build(net)
        w1 = tg1.get_weights()
        tg2.set_weights(w1)
        w2 = tg2.get_weights()
        for k in w1:
            np.testing.assert_allclose(w1[k], w2[k], rtol=1e-5)

    def test_set_weights_ignores_unknown_keys(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        tg.set_weights({"nonexistent_key_xyz": np.zeros(5)})  # must not raise


# ---------------------------------------------------------------------------
# reset_state
# ---------------------------------------------------------------------------

class TestResetState:
    def test_reset_state_gives_reproducible_output(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        r1 = tg.forward(5)
        tg.reset_state()
        r2 = tg.forward(5)
        np.testing.assert_allclose(
            r1[p].detach().cpu().numpy(), r2[p].detach().cpu().numpy(), rtol=1e-4
        )


# ---------------------------------------------------------------------------
# TorchNode module discovery
# ---------------------------------------------------------------------------

class TestTorchModuleDiscovery:
    def test_torch_modules_list_populated(self):
        lin = nn.Linear(4, 2)
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(4))
            node = nengo_dl.TorchNode(lin, size_in=4, size_out=2)
            nengo.Connection(inp, node, synapse=None)
            p = nengo.Probe(node, synapse=None)

        tg = _build(net)
        assert len(tg._torch_modules) > 0

    def test_torch_module_on_correct_device(self):
        lin = nn.Linear(4, 2)
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(4))
            node = nengo_dl.TorchNode(lin, size_in=4, size_out=2)
            nengo.Connection(inp, node, synapse=None)
            p = nengo.Probe(node, synapse=None)

        tg = _build(net)
        for module in tg._torch_modules:
            for param in module.parameters():
                assert param.device == tg.device
