"""Dummy/mock objects used across nengo-dl tests."""

import numpy as np
import torch
import torch.nn as nn
import nengo
import nengo.dists
from nengo.builder.operator import Operator
from nengo.builder.signal import Signal as NengoSignal


# ---------------------------------------------------------------------------
# Dummy Nengo operators
# ---------------------------------------------------------------------------

class DummyOperator(Operator):
    """Minimal Nengo operator that does nothing — used to populate op lists."""

    def __init__(self, tag=None):
        super().__init__(tag=tag)
        self.sets = []
        self.incs = []
        self.reads = []
        self.updates = []

    def make_step(self, signals, dt, rng):
        def step():
            pass
        return step


class WriteOperator(Operator):
    """Operator that writes to a single signal."""

    def __init__(self, sig, tag=None):
        super().__init__(tag=tag)
        self.sig = sig
        self.sets = [sig]
        self.incs = []
        self.reads = []
        self.updates = []

    def make_step(self, signals, dt, rng):
        def step():
            signals[self.sig][...] = 1.0
        return step


class ReadOperator(Operator):
    """Operator that reads from a single signal."""

    def __init__(self, sig, tag=None):
        super().__init__(tag=tag)
        self.sig = sig
        self.sets = []
        self.incs = []
        self.reads = [sig]
        self.updates = []

    def make_step(self, signals, dt, rng):
        def step():
            pass
        return step


# ---------------------------------------------------------------------------
# Dummy PyTorch modules
# ---------------------------------------------------------------------------

class IdentityModule(nn.Module):
    """nn.Module that passes input through unchanged."""

    def forward(self, x):
        return x


class ScaleModule(nn.Module):
    """nn.Module that scales input by a fixed factor (no learned params)."""

    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class LinearModule(nn.Module):
    """Single linear layer with controllable initialization."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.eye_(self.linear.weight[:min(out_features, in_features),
                                        :min(out_features, in_features)])
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


# ---------------------------------------------------------------------------
# Dummy Nengo networks
# ---------------------------------------------------------------------------

def make_node_to_node_net(seed=0, input_val=1.0):
    """Minimal Node→Node network: no ensembles, single connection."""
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(np.array([input_val]))
        out = nengo.Node(size_in=1)
        nengo.Connection(inp, out, synapse=None)
        p = nengo.Probe(out, synapse=None)
    return net, inp, out, p


def make_ensemble_net(n_neurons=20, dimensions=1, seed=0):
    """Simple input→ensemble→probe network."""
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(np.zeros(dimensions))
        ens = nengo.Ensemble(n_neurons, dimensions, seed=seed)
        nengo.Connection(inp, ens, synapse=None)
        p = nengo.Probe(ens, synapse=None)
    return net, inp, ens, p


def make_two_layer_net(seed=0):
    """Two-ensemble feed-forward network."""
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(np.zeros(2))
        ens1 = nengo.Ensemble(20, 2, neuron_type=nengo.RectifiedLinear(), seed=seed)
        ens2 = nengo.Ensemble(10, 2, neuron_type=nengo.RectifiedLinear(), seed=seed + 1)
        nengo.Connection(inp, ens1, synapse=None)
        nengo.Connection(ens1, ens2, synapse=None)
        out = nengo.Node(size_in=2)
        nengo.Connection(ens2, out, function=lambda x: x, synapse=None)
        p = nengo.Probe(out, synapse=None)
    return net, inp, ens1, ens2, out, p


# ---------------------------------------------------------------------------
# Dummy signals
# ---------------------------------------------------------------------------

def make_signal(shape, name="sig", trainable=False):
    """Create a Nengo Signal with the given shape."""
    return NengoSignal(np.zeros(shape), name=name, shape=shape)


def make_signal_pair(shape, name_a="sig_a", name_b="sig_b"):
    """Create two independent signals with the same shape."""
    return make_signal(shape, name_a), make_signal(shape, name_b)


# ---------------------------------------------------------------------------
# Compatibility helpers used by test_nengo_tests / test_keras
# ---------------------------------------------------------------------------

class Probe(nengo.Probe):
    """Minimal Probe that bypasses target validation (for use with raw Signals)."""

    def __init__(self, target=None, add_to_container=True):
        # pylint: disable=super-init-not-called
        if target is not None:
            nengo.Probe.target.data[self] = target

    @property
    def size_in(self):
        return (
            self.target.size
            if isinstance(self.target, NengoSignal)
            else self.target.size_out
        )


def linear_net():
    """Simple node→node network with a 1× connection and a probe."""
    with nengo.Network() as net:
        a = nengo.Node([1])
        b = nengo.Node(size_in=1)
        nengo.Connection(a, b, synapse=None, transform=1)
        p = nengo.Probe(b)
    return net, a, p


def DeterministicLIF(**kwargs):
    """LIF with deterministic initial voltage (voltage fixed at 0)."""
    return nengo.LIF(
        initial_state={"voltage": nengo.dists.Choice([0])}, **kwargs
    )
