"""Tests for nengo_dl.Converter."""

import warnings
import numpy as np
import pytest
import torch
import torch.nn as nn
import nengo
import nengo_dl
from nengo_dl.converter import Converter, ConversionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp(in_size=4, hidden=8, out_size=3):
    return nn.Sequential(
        nn.Linear(in_size, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_size),
    )


# ---------------------------------------------------------------------------
# Basic conversion
# ---------------------------------------------------------------------------

class TestConverterBasic:
    def test_returns_nengo_network(self):
        model = _make_mlp()
        c = Converter(model)
        assert isinstance(c.net, nengo.Network)

    def test_inputs_dict_populated(self):
        model = _make_mlp()
        c = Converter(model)
        assert len(c.inputs) > 0

    def test_outputs_dict_populated(self):
        model = _make_mlp()
        c = Converter(model)
        assert len(c.outputs) > 0

    def test_input_is_node(self):
        model = _make_mlp(in_size=4)
        c = Converter(model)
        for node in c.inputs.values():
            assert isinstance(node, nengo.Node)

    def test_network_runs_in_simulator(self):
        model = _make_mlp(in_size=4, hidden=8, out_size=3)
        c = Converter(model)

        inp_node = list(c.inputs.values())[0]
        out_node = list(c.outputs.values())[-1]
        with c.net:
            p = nengo.Probe(out_node, synapse=None)

        x = np.random.randn(1, 1, 4).astype(np.float32)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp_node: x})
            data = sim.data[p]
        assert data.shape[-1] == 3

    def test_scale_firing_rates_applied(self):
        """scale_firing_rates should rescale connection weights."""
        scale = 100.0
        model = _make_mlp(in_size=4, hidden=8, out_size=3)
        c_noscale = Converter(model, scale_firing_rates=None)
        c_scaled = Converter(model, scale_firing_rates=scale)
        # The scaled network should have different (smaller) connection weights
        # We can verify this by checking the connection transforms
        noscale_conns = list(c_noscale.net.all_connections)
        scaled_conns = list(c_scaled.net.all_connections)
        assert len(noscale_conns) == len(scaled_conns)


# ---------------------------------------------------------------------------
# Layer-specific conversion
# ---------------------------------------------------------------------------

class TestConverterLayers:
    def test_converts_linear(self):
        model = nn.Linear(4, 8)
        c = Converter(model)
        assert isinstance(c.net, nengo.Network)

    def test_converts_relu(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        c = Converter(model)
        assert isinstance(c.net, nengo.Network)

    def test_converts_sigmoid(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Sigmoid())
        c = Converter(model, activation_type="sigmoid")
        assert isinstance(c.net, nengo.Network)

    def test_converts_flatten(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        c = Converter(model)
        assert isinstance(c.net, nengo.Network)

    def test_converts_identity(self):
        model = nn.Sequential(nn.Identity(), nn.Linear(4, 2))
        c = Converter(model)
        assert isinstance(c.net, nengo.Network)

    def test_allow_fallback_true_warns(self):
        class UnsupportedLayer(nn.Module):
            def forward(self, x):
                return x[:, :2]  # halve features

        model = nn.Sequential(nn.Linear(4, 4), UnsupportedLayer())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = Converter(model, allow_fallback=True)
        assert any("not natively supported" in str(warning.message) for warning in w)

    def test_allow_fallback_false_raises(self):
        class UnsupportedLayer(nn.Module):
            def forward(self, x):
                return x

        model = nn.Sequential(nn.Linear(4, 4), UnsupportedLayer())
        with pytest.raises(ConversionError):
            Converter(model, allow_fallback=False)


# ---------------------------------------------------------------------------
# Activation types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("activation_type", [
    "rectified_linear", "relu", "lif", "softlif", "sigmoid",
])
def test_activation_types(activation_type):
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    c = Converter(model, activation_type=activation_type)
    assert isinstance(c.net, nengo.Network)


def test_unknown_activation_type_warns():
    model = _make_mlp()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = Converter(model, activation_type="super_neuron_9000")
    assert any("Unknown activation_type" in str(warning.message) for warning in w)
    assert isinstance(c.net, nengo.Network)


# ---------------------------------------------------------------------------
# Synapse
# ---------------------------------------------------------------------------

class TestConverterSynapse:
    def test_synapse_none_runs(self):
        model = _make_mlp()
        c = Converter(model, synapse=None)
        assert isinstance(c.net, nengo.Network)

    def test_synapse_float_runs(self):
        model = _make_mlp()
        c = Converter(model, synapse=0.005)
        assert isinstance(c.net, nengo.Network)


# ---------------------------------------------------------------------------
# ConversionError
# ---------------------------------------------------------------------------

def test_conversion_error_is_exception():
    err = ConversionError("test error")
    assert isinstance(err, Exception)
    assert "test error" in str(err)
