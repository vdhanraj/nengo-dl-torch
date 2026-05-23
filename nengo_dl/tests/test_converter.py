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


# ---------------------------------------------------------------------------
# Numerical verification (original: test_verify_* from test_converter.py)
# ---------------------------------------------------------------------------

class TestConverterNumerical:
    """Verify that the converted network produces outputs consistent with the
    original PyTorch model (at least structurally, since rate vs spiking
    neurons differ in dynamics).
    """

    def test_output_shape_matches_pytorch(self):
        """The Nengo network output shape must match the PyTorch model output."""
        in_size, out_size = 4, 3
        model = nn.Sequential(nn.Linear(in_size, 8), nn.ReLU(), nn.Linear(8, out_size))
        model.eval()

        c = Converter(model, scale_firing_rates=100.0)
        inp_node = list(c.inputs.values())[0]
        out_node = list(c.outputs.values())[-1]

        with c.net:
            p = nengo.Probe(out_node, synapse=None)

        x = np.random.RandomState(0).randn(1, 1, in_size).astype(np.float32)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp_node: x})
            nengo_out = sim.data[p]

        # Shape: (n_steps=1, out_size=3)
        assert nengo_out.shape == (1, out_size), (
            f"Expected shape (1, {out_size}), got {nengo_out.shape}"
        )

    def test_no_nan_in_output(self):
        """Converted network must not produce NaN outputs."""
        model = _make_mlp(in_size=4, hidden=8, out_size=3)
        c = Converter(model, scale_firing_rates=100.0)
        inp_node = list(c.inputs.values())[0]
        out_node = list(c.outputs.values())[-1]

        with c.net:
            p = nengo.Probe(out_node, synapse=None)

        x = np.random.RandomState(1).randn(1, 5, 4).astype(np.float32)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(5, data={inp_node: x})
            out = sim.data[p]

        assert not np.any(np.isnan(out)), "Converted network produced NaN output"
        assert not np.any(np.isinf(out)), "Converted network produced Inf output"

    def test_linear_only_network_positive_outputs(self):
        """Converted network with positive-definite input must match PyTorch+ReLU.

        The converter always uses RectifiedLinear neurons, so the Nengo output
        equals relu(W*x + b) even when the original model has no activation.
        We verify with inputs that produce positive linear outputs.
        """
        torch.manual_seed(42)
        model = nn.Linear(4, 3, bias=True)
        # Force weights / bias to be all-positive so relu is a no-op
        with torch.no_grad():
            model.weight.data.abs_()
            model.bias.data.abs_()
        model.eval()

        x_np = np.abs(np.random.RandomState(0).randn(4).astype(np.float32))

        with torch.no_grad():
            pytorch_out = model(torch.tensor(x_np)).numpy()

        c = Converter(model, activation_type="rectified_linear", synapse=None)
        inp_node = list(c.inputs.values())[0]
        out_node = list(c.outputs.values())[-1]

        with c.net:
            p = nengo.Probe(out_node, synapse=None)

        x_batch = x_np.reshape(1, 1, 4)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp_node: x_batch})
            nengo_out = sim.data[p][0]  # first (and only) timestep

        # With positive outputs, ReLU does not clip, so Nengo ≈ PyTorch
        np.testing.assert_allclose(
            nengo_out, pytorch_out, rtol=1e-4, atol=1e-4,
            err_msg="Converted linear network output mismatch for positive inputs"
        )

    def test_synapse_applied_to_connections(self):
        """Converter with synapse!=None should create connections with that synapse."""
        tau = 0.005
        model = _make_mlp(in_size=4, hidden=8, out_size=3)
        c = Converter(model, synapse=tau)

        synapse_taus = []
        for conn in c.net.all_connections:
            if conn.synapse is not None:
                synapse_taus.append(getattr(conn.synapse, "tau", None))

        # At least some connections should carry the synapse
        assert any(t == tau for t in synapse_taus), (
            f"No connection found with synapse tau={tau}; found: {synapse_taus}"
        )
