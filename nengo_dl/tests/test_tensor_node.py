"""Tests for nengo_dl.TorchNode and nengo_dl.Layer."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import nengo
import nengo_dl
from nengo_dl.tensor_node import TorchNode, Layer


# ---------------------------------------------------------------------------
# TorchNode basics
# ---------------------------------------------------------------------------

class TestTorchNode:
    def test_size_out_required(self):
        with pytest.raises(ValueError, match="size_out"):
            TorchNode(nn.Linear(4, 3))

    def test_creates_nengo_node(self):
        with nengo.Network():
            node = TorchNode(nn.Linear(4, 3), size_in=4, size_out=3)
        assert isinstance(node, nengo.Node)
        assert node.size_out == 3

    def test_get_module_returns_module(self):
        lin = nn.Linear(4, 3)
        with nengo.Network():
            node = TorchNode(lin, size_in=4, size_out=3)
        assert node.get_module() is lin

    def test_get_module_returns_none_for_callable(self):
        fn = lambda x: x
        with nengo.Network():
            node = TorchNode(fn, size_in=2, size_out=2)
        assert node.get_module() is None

    def test_shape_in_shape_out_infer_sizes(self):
        with nengo.Network():
            node = TorchNode(nn.Linear(12, 5),
                             shape_in=(3, 4),
                             shape_out=(5,))
        assert node.size_in == 12
        assert node.size_out == 5

    def test_label(self):
        with nengo.Network():
            node = TorchNode(nn.Linear(2, 2), size_in=2, size_out=2,
                             label="MyNode")
        assert node.label == "MyNode"


# ---------------------------------------------------------------------------
# TorchNode in simulation
# ---------------------------------------------------------------------------

class TestTorchNodeSimulation:
    def test_identity_node(self):
        """TorchNode wrapping identity produces same output as input."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(3))
            identity = TorchNode(nn.Identity(), size_in=3, size_out=3)
            nengo.Connection(inp, identity, synapse=None)
            p = nengo.Probe(identity, synapse=None)

        x = np.array([[0.5, -0.3, 1.2]])
        x_nd = x.reshape(1, 1, 3)  # (batch=1, steps=1, features=3)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_nd})
            out = sim.data[p]  # (steps=1, features=3)
        np.testing.assert_allclose(out[0], x[0], atol=1e-5)

    def test_linear_transform(self):
        """TorchNode with fixed Linear weights applies correct transform."""
        weight = np.array([[2.0, 0.0], [0.0, 3.0]])
        linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(weight, dtype=torch.float32))

        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            node = TorchNode(linear, size_in=2, size_out=2)
            nengo.Connection(inp, node, synapse=None)
            p = nengo.Probe(node, synapse=None)

        x = np.array([[1.0, 1.0]])
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1, data={inp: x.reshape(1, 1, 2)})
            out = sim.data[p]
        np.testing.assert_allclose(out[0], [2.0, 3.0], atol=1e-5)

    def test_gradient_flows_through_node(self):
        """Training loss must decrease when a TorchNode is the only trainable part."""
        linear = nn.Linear(1, 1, bias=False)
        nn.init.ones_(linear.weight)

        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            node = TorchNode(linear, size_in=1, size_out=1)
            nengo.Connection(inp, node, synapse=None)
            p = nengo.Probe(node, synapse=None)

        n = 32
        x = np.random.RandomState(0).randn(n, 1, 1).astype(np.float32)
        y = x * 0.5  # target: scale by 0.5

        with nengo_dl.Simulator(net, minibatch_size=n, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            history = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=50)

        # Loss must decrease — gradient is flowing through TorchNode
        assert history["loss"][-1] < history["loss"][0], "Loss did not decrease"

    def test_relu_node(self):
        """TorchNode with ReLU gives zero for negative inputs."""
        relu = nn.ReLU()
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(3))
            node = TorchNode(relu, size_in=3, size_out=3)
            nengo.Connection(inp, node, synapse=None)
            p = nengo.Probe(node, synapse=None)

        x = np.array([[[-1.0, 0.0, 2.0]]])  # (batch=1, steps=1, features=3)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1, data={inp: x})
            out = sim.data[p]
        np.testing.assert_allclose(out[0], [0.0, 0.0, 2.0], atol=1e-5)

    def test_module_parameters_are_trainable(self):
        linear = nn.Linear(4, 2)
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(4))
            node = TorchNode(linear, size_in=4, size_out=2)
            nengo.Connection(inp, node, synapse=None)
            p = nengo.Probe(node, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            params = sim.trainable_params()
        assert len(params) > 0


# ---------------------------------------------------------------------------
# Layer API
# ---------------------------------------------------------------------------

class TestLayer:
    def test_layer_with_linear(self):
        """Layer(nn.Linear) creates a TorchNode and Connection."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(4))
            out_node = nengo_dl.Layer(nn.Linear(4, 8))(inp)
        assert isinstance(out_node, TorchNode)
        assert out_node.size_out == 8

    def test_layer_with_neuron_type(self):
        """Layer(RectifiedLinear) creates an Ensemble that preserves spiking."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(6))
            act = nengo_dl.Layer(nengo.RectifiedLinear())(inp)
        assert not isinstance(act, TorchNode)
        assert act.size_out == 6

    def test_layer_with_neuron_type_use_rate(self):
        """Layer(RectifiedLinear, use_rate=True) creates a rate TorchNode."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(6))
            act = nengo_dl.Layer(nengo.RectifiedLinear(), use_rate=True)(inp)
        assert isinstance(act, TorchNode)
        assert act.size_out == 6

    def test_layer_with_lif(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(4))
            act = nengo_dl.Layer(nengo.LIF())(inp)
        assert act.size_out == 4

    def test_layer_with_softlif(self):
        from nengo_dl.neurons import SoftLIFRate
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(4))
            act = nengo_dl.Layer(SoftLIFRate())(inp)
        assert act.size_out == 4

    def test_layer_chaining(self):
        """Multiple Layers can be chained."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(8))
            h1 = nengo_dl.Layer(nn.Linear(8, 16))(inp)
            h2 = nengo_dl.Layer(nengo.RectifiedLinear())(h1)
            out = nengo_dl.Layer(nn.Linear(16, 4))(h2)
        assert out.size_out == 4

    def test_layer_with_synapse(self):
        """Synapse parameter is forwarded to the Connection."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(3))
            node = nengo_dl.Layer(nn.Identity())(inp, synapse=0.01)
        assert isinstance(node, TorchNode)

    def test_layer_shape_in(self):
        """shape_in parameter controls input reshaping."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(12))
            # Flatten layer: pass through
            node = nengo_dl.Layer(nn.Linear(12, 5))(inp, shape_in=(12,))
        assert node.size_out == 5

    def test_layer_runs_in_simulator(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(4))
            h = nengo_dl.Layer(nn.Linear(4, 8))(inp)
            out = nengo_dl.Layer(nn.Linear(8, 2))(h)
            p = nengo.Probe(out, synapse=None)

        x = np.random.randn(1, 3, 4).astype(np.float32)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(3, data={inp: x})
            data = sim.data[p]
        assert data.shape == (3, 2)

    def test_layer_invalid_type_raises(self):
        with nengo.Network():
            inp = nengo.Node(np.zeros(4))
            with pytest.raises(TypeError):
                nengo_dl.Layer("not_a_layer")(inp)


# ---------------------------------------------------------------------------
# neuron_type_to_module
# ---------------------------------------------------------------------------

class TestNeuronTypeToModule:
    @pytest.mark.parametrize("neuron_type", [
        nengo.LIF(),
        nengo.LIFRate(),
        nengo.RectifiedLinear(),
        nengo.SpikingRectifiedLinear(),
    ])
    def test_known_types_return_module(self, neuron_type):
        from nengo_dl.tensor_node import neuron_type_to_module
        module = neuron_type_to_module(neuron_type)
        assert isinstance(module, nn.Module)

    def test_unknown_type_returns_identity(self):
        from nengo_dl.tensor_node import neuron_type_to_module
        import warnings

        class FakeNeuron(nengo.neurons.NeuronType):
            probeable = ()
            def gain_bias(self, *a): pass
            def max_rates_intercepts(self, *a): pass
            def step(self, *a): pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            module = neuron_type_to_module(FakeNeuron())
        assert isinstance(module, nn.Identity)
        assert len(w) == 1

    def test_lif_module_forward(self):
        from nengo_dl.tensor_node import _LIFRateModule
        mod = _LIFRateModule(tau_rc=0.02, tau_ref=0.002, amplitude=1.0)
        x = torch.tensor([[2.0, 0.0, -1.0]])
        out = mod(x)
        assert out.shape == (1, 3)
        assert out[0, 1].item() == pytest.approx(0.0, abs=1e-5)
        assert out[0, 0].item() > 0
