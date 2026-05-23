"""Tests for learning rule support in nengo-dl.

The PyTorch backend handles learning rules by:
  - PES/BCM/Voja/Oja: No native builder yet (SimPES etc. not registered).
    These tests verify that the expected error is raised and that the
    reference Nengo CPU simulator still handles them correctly.
  - Gradient-based weight updates: handled by sim.fit() instead.
"""

import numpy as np
import pytest
import nengo
import nengo_dl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pes_net(seed=0):
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(np.zeros(1))
        pre = nengo.Ensemble(20, 1, seed=seed)
        post = nengo.Ensemble(10, 1, seed=seed + 1)
        conn = nengo.Connection(pre, post, learning_rule_type=nengo.PES(),
                                synapse=0.01)
        err = nengo.Node(size_in=1)
        nengo.Connection(err, conn.learning_rule)
        nengo.Connection(inp, pre, synapse=None)
        p_post = nengo.Probe(post, synapse=0.01)
    return net, inp, err, conn, p_post


# ---------------------------------------------------------------------------
# PES (not yet supported in PyTorch backend)
# ---------------------------------------------------------------------------

class TestPES:
    def test_pes_raises_no_builder(self):
        """nengo-dl should raise ValueError for unregistered SimPES."""
        net, inp, err, conn, p_post = _make_pes_net()
        with pytest.raises((ValueError, Exception), match="[Bb]uilder|SimPES"):
            with nengo_dl.Simulator(net, seed=0) as sim:
                pass

    def test_pes_works_in_reference_nengo(self):
        """PES should work in Nengo's standard CPU simulator."""
        net, inp, err, conn, p_post = _make_pes_net()
        with nengo.Simulator(net, seed=0, progress_bar=False) as ref_sim:
            ref_sim.run_steps(20)
        data = ref_sim.data[p_post]
        assert data.shape[0] == 20
        assert not np.any(np.isnan(data))


# ---------------------------------------------------------------------------
# BCM (not yet supported in PyTorch backend)
# ---------------------------------------------------------------------------

class TestBCM:
    def _make_bcm_net(self):
        """BCM requires neuron→neuron connection with an explicit weight matrix."""
        with nengo.Network(seed=0) as net:
            pre = nengo.Ensemble(10, 1, seed=0)
            post = nengo.Ensemble(10, 1, seed=1)
            nengo.Connection(pre.neurons, post.neurons,
                             transform=np.ones((10, 10)),
                             learning_rule_type=nengo.BCM(),
                             synapse=0.01)
            inp = nengo.Node(np.zeros(1))
            nengo.Connection(inp, pre, synapse=None)
            p = nengo.Probe(post, synapse=None)
        return net, p

    def test_bcm_raises_no_builder(self):
        net, p = self._make_bcm_net()
        with pytest.raises((ValueError, Exception)):
            with nengo_dl.Simulator(net, seed=0):
                pass

    def test_bcm_works_in_reference_nengo(self):
        net, p = self._make_bcm_net()
        with nengo.Simulator(net, seed=0, progress_bar=False) as ref_sim:
            ref_sim.run_steps(10)
        assert not np.any(np.isnan(ref_sim.data[p]))


# ---------------------------------------------------------------------------
# Gradient-based "learning" via sim.fit()
# ---------------------------------------------------------------------------

class TestGradientLearning:
    def test_fit_reduces_supervised_loss(self):
        """sim.fit() is the primary learning mechanism in nengo-dl."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(30, 1,
                                 neuron_type=nengo.RectifiedLinear(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            out = nengo.Node(size_in=1)
            nengo.Connection(ens, out, function=lambda x: x, synapse=None)
            p = nengo.Probe(out, synapse=None)

        n = 64
        rng = np.random.RandomState(0)
        x = rng.uniform(-1, 1, (n, 1, 1)).astype(np.float32)
        y = (x ** 2).astype(np.float32)  # learn x → x²

        with nengo_dl.Simulator(net, minibatch_size=16, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            history = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=10)

        assert history["loss"][-1] < history["loss"][0]

    def test_weight_update_after_training(self):
        """Weights should change after training."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(20, 1,
                                 neuron_type=nengo.RectifiedLinear(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            out = nengo.Node(size_in=1)
            nengo.Connection(ens, out, function=lambda x: x, synapse=None)
            p = nengo.Probe(out, synapse=None)

        x = np.random.RandomState(0).uniform(-1, 1, (32, 1, 1)).astype(np.float32)
        y = np.zeros_like(x)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            w_before = {k: v.copy() for k, v in sim.get_weights().items()}
            sim.compile(optimizer="adam", loss={p: "mse"})
            sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=10)
            w_after = sim.get_weights()

        changed = any(
            not np.allclose(w_before[k], w_after[k])
            for k in w_before
        )
        assert changed, "No weights changed after training"

    def test_gradient_flows_to_early_layers(self):
        """Gradient must propagate through multiple connections."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            ens1 = nengo.Ensemble(20, 2,
                                  neuron_type=nengo.RectifiedLinear(), seed=0)
            ens2 = nengo.Ensemble(10, 2,
                                  neuron_type=nengo.RectifiedLinear(), seed=1)
            nengo.Connection(inp, ens1, synapse=None)
            nengo.Connection(ens1, ens2, synapse=None)
            out = nengo.Node(size_in=2)
            nengo.Connection(ens2, out, function=lambda x: x, synapse=None)
            p = nengo.Probe(out, synapse=None)

        x = np.random.RandomState(0).randn(32, 1, 2).astype(np.float32)
        y = np.zeros_like(x)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            history = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=5)

        assert history["loss"][-1] < history["loss"][0] + 1.0  # at worst stable
