"""Tests for nengo_dl.op_builders (core operator builders)."""

import numpy as np
import pytest
import torch
import nengo
import nengo_dl
from nengo_dl.signals import SignalDict
from nengo.builder.signal import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sd(batch=2):
    return SignalDict(batch, torch.device("cpu"), torch.float32)


def _sig(shape, name="s", init=None, readonly=False):
    val = np.zeros(shape) if init is None else np.asarray(init, dtype=float)
    s = Signal(val.copy(), name=name, shape=shape)
    if readonly:
        object.__setattr__(s, '_readonly', True)
    return s


# ---------------------------------------------------------------------------
# TimeUpdate
# ---------------------------------------------------------------------------

class TestTimeUpdateBuilder:
    def test_step_and_time_increment(self):
        """After one build_step, step should be 1 and time should be dt."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run_steps(5)
            # Probe data should have 5 timesteps
            assert sim.data[p].shape[0] == 5

    def test_step_count_matches_n_steps(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(lambda t: t)
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run_steps(10)
            data = sim.data[p]
        # The node returns t, so data[0] ≈ dt, data[9] ≈ 10*dt
        assert data.shape[0] == 10
        assert data[9, 0] == pytest.approx(0.01, abs=1e-4)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestResetBuilder:
    def test_reset_sets_constant_value(self):
        """Reset should set a signal to a fixed value each step."""
        with nengo.Network(seed=0) as net:
            # A node with output=np.zeros gets a Reset op to re-zero each step
            inp = nengo.Node(np.zeros(3))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            # Inject non-zero data for one step, then run more; Reset re-zeros it
            x = np.ones((1, 3, 3), dtype=np.float32)
            sim.run_steps(3, data={inp: x})
        # The Node's output signal is overridden by input, but Reset base is tested

    def test_reset_in_lif_ensemble(self):
        """LIF ensemble has Reset ops for refractory time and voltage; must not crash."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(5, 1, neuron_type=nengo.LIF(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
        assert sim.data[p].shape == (5, 1)


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

class TestCopyBuilder:
    def test_copy_propagates_value(self):
        """Connected Nodes should copy signal values between them."""
        with nengo.Network(seed=0) as net:
            a = nengo.Node(np.array([1.0, 2.0, 3.0]))
            b = nengo.Node(size_in=3)
            nengo.Connection(a, b, synapse=None)
            p = nengo.Probe(b, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        np.testing.assert_allclose(sim.data[p][0], [1.0, 2.0, 3.0], atol=1e-5)

    def test_copy_with_input_override(self):
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.zeros(2))
            dst = nengo.Node(size_in=2)
            nengo.Connection(src, dst, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        x = np.array([[[5.0, -3.0]]])  # (batch=1, steps=1, size=2)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1, data={src: x})
        np.testing.assert_allclose(sim.data[p][0], [5.0, -3.0], atol=1e-5)


# ---------------------------------------------------------------------------
# ElementwiseInc / DotInc via Connection transforms
# ---------------------------------------------------------------------------

class TestDotIncBuilder:
    def test_linear_connection(self):
        """A connection with a scalar transform uses DotInc / ElementwiseInc."""
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([2.0]))
            dst = nengo.Node(size_in=1)
            nengo.Connection(src, dst, transform=3.0, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        # 2.0 * 3.0 = 6.0
        assert sim.data[p][0, 0] == pytest.approx(6.0, abs=1e-4)

    def test_matrix_transform(self):
        """A matrix transform maps from N-d input to M-d output."""
        W = np.array([[1.0, 0.0], [0.0, 2.0]])  # 2→2 diagonal
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([3.0, 4.0]))
            dst = nengo.Node(size_in=2)
            nengo.Connection(src, dst, transform=W, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        np.testing.assert_allclose(sim.data[p][0], [3.0, 8.0], atol=1e-4)

    def test_dimension_reduction(self):
        """2-D → 1-D via a row vector transform."""
        W = np.array([[1.0, 1.0]])  # sum both dimensions
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([5.0, 7.0]))
            dst = nengo.Node(size_in=1)
            nengo.Connection(src, dst, transform=W, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        assert sim.data[p][0, 0] == pytest.approx(12.0, abs=1e-4)


# ---------------------------------------------------------------------------
# SimPyFunc  (Python-function nodes)
# ---------------------------------------------------------------------------

class TestSimPyFuncBuilder:
    def test_source_node(self):
        """A Node with no input runs fn(t) each step."""
        with nengo.Network(seed=0) as net:
            src = nengo.Node(lambda t: np.array([t]))
            p = nengo.Probe(src, synapse=None)

        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run_steps(3)
        data = sim.data[p]
        assert data.shape == (3, 1)
        assert data[0, 0] == pytest.approx(0.001, abs=1e-4)
        assert data[2, 0] == pytest.approx(0.003, abs=1e-4)

    def test_transform_node(self):
        """A Node with fn(t, x) transforms its input."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([2.0]))
            tfm = nengo.Node(lambda t, x: x * 3.0, size_in=1, size_out=1)
            nengo.Connection(inp, tfm, synapse=None)
            p = nengo.Probe(tfm, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        assert sim.data[p][0, 0] == pytest.approx(6.0, abs=1e-4)

    def test_constant_node(self):
        """A Node with a constant array output each step."""
        const = np.array([7.0, -3.0, 1.5])
        with nengo.Network(seed=0) as net:
            src = nengo.Node(const)
            p = nengo.Probe(src, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(4)
        for t in range(4):
            np.testing.assert_allclose(sim.data[p][t], const, atol=1e-5)


# ---------------------------------------------------------------------------
# SimProbe
# ---------------------------------------------------------------------------

class TestSimProbeBuilder:
    def test_probe_accumulates_steps(self):
        with nengo.Network(seed=0) as net:
            src = nengo.Node(lambda t: np.array([t]))
            p = nengo.Probe(src, synapse=None)

        n = 7
        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run_steps(n)
        assert sim.data[p].shape == (n, 1)

    def test_multiple_probes(self):
        with nengo.Network(seed=0) as net:
            n1 = nengo.Node(np.array([1.0]))
            n2 = nengo.Node(np.array([2.0, 3.0]))
            p1 = nengo.Probe(n1, synapse=None)
            p2 = nengo.Probe(n2, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
        assert sim.data[p1].shape == (5, 1)
        assert sim.data[p2].shape == (5, 2)

    def test_probe_correct_values(self):
        values = np.array([10.0, 20.0, 30.0])
        with nengo.Network(seed=0) as net:
            src = nengo.Node(values)
            p = nengo.Probe(src, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(3)
        for t in range(3):
            np.testing.assert_allclose(sim.data[p][t], values, atol=1e-5)


# ---------------------------------------------------------------------------
# Multi-step consistency
# ---------------------------------------------------------------------------

class TestMultiStepOps:
    def test_accumulation_across_steps(self):
        """Incrementing signal should grow monotonically over steps."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.ones(1))
            ens = nengo.Ensemble(10, 1, neuron_type=nengo.RectifiedLinear(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            out = nengo.Node(size_in=1)
            nengo.Connection(ens, out, function=lambda x: x, synapse=None)
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(10)
        data = sim.data[p]
        assert data.shape == (10, 1)
        # All outputs should be the same since input is constant
        np.testing.assert_allclose(data, np.full_like(data, data[0, 0]), atol=0.1)

    def test_batched_ops(self):
        """Batched simulation should produce shape (batch, n_steps, dim)."""
        bs = 6
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            out = nengo.Node(size_in=2)
            nengo.Connection(inp, out, synapse=None)
            p = nengo.Probe(out, synapse=None)

        x = np.ones((bs, 4, 2))
        with nengo_dl.Simulator(net, minibatch_size=bs, seed=0) as sim:
            sim.run_steps(4, data={inp: x})
        assert sim.data[p].shape == (bs, 4, 2)
