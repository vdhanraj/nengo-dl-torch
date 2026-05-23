"""Tests for nengo_dl.benchmarks network constructors."""

from collections import defaultdict

import nengo
import numpy as np
import pytest

from nengo_dl import benchmarks


# ---------------------------------------------------------------------------
# Standard network structure tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "benchmark",
    (benchmarks.cconv, benchmarks.integrator, benchmarks.pes, benchmarks.basal_ganglia),
)
def test_networks(benchmark):
    dimensions = 16
    neurons_per_d = 10
    neuron_type = nengo.RectifiedLinear()

    net = benchmark(dimensions, neurons_per_d, neuron_type)

    # input node(s) size
    if benchmark == benchmarks.cconv:
        assert net.inp_a.size_out == dimensions
        assert net.inp_b.size_out == dimensions
    else:
        assert net.inp.size_out == dimensions

    # output probe size
    assert net.p.size_in == dimensions

    # ensemble neuron type and size
    for ens in net.all_ensembles:
        assert ens.neuron_type == neuron_type
        if benchmark == benchmarks.cconv:
            assert ens.n_neurons == ens.dimensions * (neurons_per_d // 2)
        else:
            assert ens.n_neurons == ens.dimensions * neurons_per_d


# ---------------------------------------------------------------------------
# mnist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tensor_layer", (True, False))
def test_mnist(tensor_layer):
    net = benchmarks.mnist(use_tensor_layer=tensor_layer)
    assert net.inp.size_out == 28 * 28
    assert net.p.size_in == 10


def test_mnist_runs():
    """MNIST network should run without errors in nengo-dl."""
    import nengo_dl

    net = benchmarks.mnist(use_tensor_layer=False)
    x = np.zeros((1, 1, 28 * 28), dtype=np.float32)
    with nengo_dl.Simulator(net, minibatch_size=1, seed=0) as sim:
        sim.run_steps(1, data={net.inp: x})
    assert not np.any(np.isnan(sim.data[net.p]))


# ---------------------------------------------------------------------------
# random_network
# ---------------------------------------------------------------------------

def _check_random(net, dimensions, neurons_per_d, neuron_type, n_ensembles, n_connections):
    assert net.inp.size_out == dimensions
    assert net.out.size_in == dimensions
    assert len(net.all_ensembles) == n_ensembles
    assert all(ens.neuron_type == neuron_type for ens in net.all_ensembles)
    assert all(
        ens.n_neurons == dimensions * neurons_per_d for ens in net.all_ensembles
    )

    pre_conns = defaultdict(list)
    post_conns = defaultdict(list)
    for conn in net.all_connections:
        if isinstance(conn.pre_obj, nengo.Ensemble):
            pre_conns[conn.pre_obj].append(conn.post_obj)
        if isinstance(conn.post_obj, nengo.Ensemble):
            post_conns[conn.post_obj].append(conn.pre_obj)

    assert len(pre_conns) == n_ensembles
    assert all(len(x) == n_connections + 1 for x in pre_conns.values())
    assert all(net.out in x for x in pre_conns.values())
    assert all(net.inp in x for x in post_conns.values())


@pytest.mark.parametrize(
    "dimensions, neurons_per_d, neuron_type, n_ensembles, n_connections",
    (
        (1, 10, nengo.RectifiedLinear(), 5, 3),
        (2, 4, nengo.LIF(), 10, 2),
    ),
)
def test_random_network(dimensions, neurons_per_d, neuron_type, n_ensembles, n_connections):
    net = benchmarks.random_network(
        dimensions, neurons_per_d, neuron_type, n_ensembles, n_connections
    )
    _check_random(net, dimensions, neurons_per_d, neuron_type, n_ensembles, n_connections)


def test_random_network_reproducible():
    """Same seed should produce the same connectivity."""
    kw = dict(dimensions=2, neurons_per_d=4, neuron_type=nengo.RectifiedLinear(),
              n_ensembles=5, connections_per_ensemble=2, seed=42)
    net1 = benchmarks.random_network(**kw)
    net2 = benchmarks.random_network(**kw)
    assert len(net1.all_connections) == len(net2.all_connections)


# ---------------------------------------------------------------------------
# integrator runs in nengo-dl
# ---------------------------------------------------------------------------

def test_integrator_runs():
    """Integrator should run without NaN in nengo-dl."""
    import nengo_dl

    net = benchmarks.integrator(4, 5, nengo.RectifiedLinear())
    with nengo_dl.Simulator(net, seed=0) as sim:
        sim.run_steps(5)
    assert not np.any(np.isnan(sim.data[net.p]))
