"""Shared pytest fixtures for nengo-dl tests."""

import numpy as np
import pytest
import torch
import nengo
import nengo_dl
from nengo_dl.config import _global_settings


@pytest.fixture(autouse=True)
def _reset_global_settings():
    """Clear nengo-dl global settings before and after every test."""
    _global_settings.clear()
    yield
    _global_settings.clear()


@pytest.fixture
def rng():
    return np.random.RandomState(0)


@pytest.fixture
def seed():
    return 0


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def simple_net():
    """A minimal Nengo network: Node → Ensemble → Probe."""
    with nengo.Network(seed=0) as net:
        node = nengo.Node(np.zeros(1))
        ens = nengo.Ensemble(10, dimensions=1, seed=0)
        nengo.Connection(node, ens, synapse=None)
        p = nengo.Probe(ens, synapse=None)
    return net, node, ens, p


@pytest.fixture
def ff_net():
    """Feedforward network suitable for training: Node → Ensemble → Node(probe)."""
    with nengo.Network(seed=0) as net:
        inp = nengo.Node(np.zeros(1))
        ens = nengo.Ensemble(50, dimensions=1,
                             neuron_type=nengo.RectifiedLinear(), seed=0)
        nengo.Connection(inp, ens, synapse=None)
        out = nengo.Node(size_in=1)
        nengo.Connection(ens, out, function=lambda x: x, synapse=None)
        p = nengo.Probe(out, synapse=None)
    return net, inp, out, p
