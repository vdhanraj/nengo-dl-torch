"""Benchmark network constructors for nengo-dl.

Each function returns a nengo.Network with standardised attributes so that
timing/accuracy benchmarks can target them uniformly:

  net.inp   – input Node (or net.inp_a / net.inp_b for cconv)
  net.p     – Probe on the network output
"""

import numpy as np
import nengo
import nengo.networks


# ---------------------------------------------------------------------------
# Core benchmark networks
# ---------------------------------------------------------------------------

def integrator(dimensions, neurons_per_d, neuron_type):
    """Recurrent integrator network."""
    with nengo.Network() as net:
        net.inp = nengo.Node(np.zeros(dimensions))
        ens = nengo.Ensemble(
            dimensions * neurons_per_d, dimensions, neuron_type=neuron_type
        )
        nengo.Connection(net.inp, ens, transform=0.1, synapse=0.01)
        nengo.Connection(ens, ens, synapse=0.01)
        net.p = nengo.Probe(ens, synapse=0.01)
    return net


def cconv(dimensions, neurons_per_d, neuron_type):
    """Circular convolution via two ensemble arrays (one per input)."""
    with nengo.Network() as net:
        net.inp_a = nengo.Node(np.zeros(dimensions))
        net.inp_b = nengo.Node(np.zeros(dimensions))

        # Two EnsembleArrays — each sub-ensemble has dimension=1,
        # n_neurons = neurons_per_d // 2, satisfying:
        #   ens.n_neurons == ens.dimensions * (neurons_per_d // 2)
        ea_a = nengo.networks.EnsembleArray(
            neurons_per_d // 2, dimensions, neuron_type=neuron_type
        )
        ea_b = nengo.networks.EnsembleArray(
            neurons_per_d // 2, dimensions, neuron_type=neuron_type
        )

        nengo.Connection(net.inp_a, ea_a.input, synapse=None)
        nengo.Connection(net.inp_b, ea_b.input, synapse=None)

        output = nengo.Node(size_in=dimensions)
        nengo.Connection(ea_a.output, output, synapse=0.01)
        nengo.Connection(ea_b.output, output, synapse=0.01)
        net.p = nengo.Probe(output, synapse=0.01)
    return net


def pes(dimensions, neurons_per_d, neuron_type):
    """Network with a PES learning connection."""
    with nengo.Network() as net:
        net.inp = nengo.Node(np.zeros(dimensions))
        pre = nengo.Ensemble(
            dimensions * neurons_per_d, dimensions, neuron_type=neuron_type
        )
        post = nengo.Ensemble(
            dimensions * neurons_per_d, dimensions, neuron_type=neuron_type
        )
        err = nengo.Node(np.zeros(dimensions))
        conn = nengo.Connection(
            pre, post, learning_rule_type=nengo.PES(), synapse=0.01
        )
        nengo.Connection(err, conn.learning_rule)
        nengo.Connection(net.inp, pre, synapse=None)
        net.p = nengo.Probe(post, synapse=0.01)
    return net


def basal_ganglia(dimensions, neurons_per_d, neuron_type):
    """Simplified basal ganglia proxy (inhibitory pathway)."""
    with nengo.Network() as net:
        net.inp = nengo.Node(np.zeros(dimensions))
        ens = nengo.Ensemble(
            dimensions * neurons_per_d, dimensions, neuron_type=neuron_type
        )
        output = nengo.Node(size_in=dimensions)
        nengo.Connection(net.inp, ens, transform=-1, synapse=None)
        nengo.Connection(ens, output, synapse=0.01)
        net.p = nengo.Probe(output, synapse=0.01)
    return net


def random_network(
    dimensions,
    neurons_per_d,
    neuron_type,
    n_ensembles=5,
    connections_per_ensemble=3,
    seed=0,
):
    """Random feed-forward + lateral connectivity network.

    Structure
    ---------
    * ``net.inp``  — input Node (size_out=dimensions)
    * ``net.out``  — output Node (size_in=dimensions)
    * ``n_ensembles`` Ensembles, each with ``dimensions * neurons_per_d`` neurons
    * Every ensemble receives from ``net.inp`` and sends to ``net.out``
    * Every ensemble also targets ``connections_per_ensemble`` random other ensembles
    """
    rng = np.random.RandomState(seed)
    with nengo.Network(seed=seed) as net:
        net.inp = nengo.Node(np.zeros(dimensions))
        net.out = nengo.Node(size_in=dimensions)

        ensembles = [
            nengo.Ensemble(
                dimensions * neurons_per_d,
                dimensions,
                neuron_type=neuron_type,
                seed=seed + i,
            )
            for i in range(n_ensembles)
        ]

        for i, ens in enumerate(ensembles):
            nengo.Connection(net.inp, ens, synapse=None)
            nengo.Connection(ens, net.out, synapse=0.01)

            other_idxs = [j for j in range(n_ensembles) if j != i]
            n_targets = min(connections_per_ensemble, len(other_idxs))
            target_idxs = rng.choice(other_idxs, size=n_targets, replace=False)
            for ti in target_idxs:
                nengo.Connection(ens, ensembles[int(ti)], synapse=0.01)

    return net


def mnist(use_tensor_layer=True):
    """Minimal MNIST network (784 → 10).

    Parameters
    ----------
    use_tensor_layer : bool
        If True, uses ``nengo_dl.Layer`` with ensemble-based activations.
        If False, uses a single ``TorchNode`` wrapping the whole network.
    """
    import torch.nn as nn
    import nengo_dl

    n_out = 10

    with nengo.Network() as net:
        net.inp = nengo.Node(np.zeros(28 * 28))

        if use_tensor_layer:
            x = nengo_dl.Layer(nn.Linear(28 * 28, 128))(net.inp)
            x = nengo_dl.Layer(nengo.RectifiedLinear())(x, shape_in=(128,))
            x = nengo_dl.Layer(nn.Linear(128, 64))(x)
            x = nengo_dl.Layer(nengo.RectifiedLinear())(x, shape_in=(64,))
            x = nengo_dl.Layer(nn.Linear(64, n_out))(x)
            x = nengo_dl.Layer(nengo.RectifiedLinear())(x, shape_in=(n_out,))
            net.p = nengo.Probe(x, synapse=None)
        else:
            model = nn.Sequential(
                nn.Linear(28 * 28, 128),
                nn.ReLU(),
                nn.Linear(128, n_out),
            )
            node = nengo_dl.TorchNode(model, size_in=28 * 28, size_out=n_out)
            nengo.Connection(net.inp, node, synapse=None)
            net.p = nengo.Probe(node, synapse=None)

    return net
