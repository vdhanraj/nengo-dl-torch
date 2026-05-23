"""Nengo compatibility tests for the nengo-dl PyTorch backend.

Adapted from the original nengo-dl test_nengo_tests.py; TF/Keras-specific
tests have been removed or replaced with PyTorch-compatible equivalents.
"""

import numpy as np
import pytest
import nengo
import nengo_dl


# ---------------------------------------------------------------------------
# Node function argument types
# ---------------------------------------------------------------------------

class TestNodeArgs:
    def test_output_is_numpy(self):
        """Node output callable should receive a numpy scalar t and numpy x."""
        received = {}

        def fn(t, x):
            received["t_type"] = type(t)
            received["x_type"] = type(x)
            received["t_shape"] = np.shape(t)
            received["x_shape"] = x.shape
            return x

        with nengo.Network() as net:
            u = nengo.Node(lambda t: np.array([t]))
            v = nengo.Node(fn, size_in=1, size_out=1)
            nengo.Connection(u, v, synapse=None)
            p = nengo.Probe(v, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(3)

        assert received["x_type"] == np.ndarray
        assert sim.data[p].shape == (3, 1)

    def test_output_not_reused(self):
        """The x array passed to a node fn should be a fresh copy each step."""
        last_x = [None]

        def fn(t, x):
            if last_x[0] is not None:
                assert last_x[0] is not x
            last_x[0] = x
            return x

        with nengo.Network() as net:
            u = nengo.Node(np.zeros(1))
            v = nengo.Node(fn, size_in=1, size_out=1)
            nengo.Connection(u, v, synapse=None)
            p = nengo.Probe(v, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)


# ---------------------------------------------------------------------------
# Timing: n_steps and time
# ---------------------------------------------------------------------------

class TestTiming:
    def test_n_steps_and_time(self):
        """n_steps and time should advance correctly with each step."""
        dt = 0.001
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, dt=dt, seed=0) as sim:
            assert sim.n_steps == 0
            sim.run_steps(1)
            assert sim.n_steps == 1
            assert np.isscalar(sim.n_steps)
            assert np.allclose(sim.time, dt)
            sim.run_steps(4)
            assert sim.n_steps == 5
            assert np.allclose(sim.time, 5 * dt)

    def test_trange_matches_n_steps(self):
        """trange() should produce exactly n_steps time-points."""
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        n = 7
        dt = 0.001
        with nengo_dl.Simulator(net, dt=dt, seed=0) as sim:
            sim.run_steps(n)
            t = sim.trange()

        assert len(t) == n
        np.testing.assert_allclose(t[-1], n * dt, rtol=1e-4)

    def test_time_absolute(self):
        """trange() values should equal dt * [1, 2, …, n]."""
        dt = 0.001
        n = 10
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, dt=dt, seed=0) as sim:
            sim.run_steps(n)
            t = sim.trange()

        expected = dt * np.arange(1, n + 1)
        np.testing.assert_allclose(t, expected, rtol=1e-4)

    def test_run_by_time(self):
        """sim.run(t) should run for the correct number of steps."""
        dt = 0.001
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, dt=dt, seed=0) as sim:
            sim.run(0.01)

        assert sim.data[p].shape[0] == 10


# ---------------------------------------------------------------------------
# Multiple sequential runs
# ---------------------------------------------------------------------------

class TestMultiRun:
    def test_probe_per_run(self):
        """Probe data shape reflects the most recent run_steps() call."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.ones(1))
            out = nengo.Node(size_in=1)
            nengo.Connection(inp, out, synapse=None)
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(3)
            assert sim.data[p].shape[0] == 3
            sim.run_steps(2)
            # Each run_steps() replaces probe data
            assert sim.data[p].shape[0] == 2

    def test_time_continues_after_multi_run(self):
        """sim.time should continue from where the last run left off."""
        dt = 0.001
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, dt=dt, seed=0) as sim:
            sim.run_steps(4)
            t1 = sim.time
            sim.run_steps(3)
            t2 = sim.time

        assert np.allclose(t2 - t1, 3 * dt, rtol=1e-4)


# ---------------------------------------------------------------------------
# Gain and bias retrieval
# ---------------------------------------------------------------------------

class TestGainBias:
    def test_gain_bias_stored(self):
        """User-specified gain and bias should be accessible via sim.data[ens]."""
        N = 10
        D = 2
        gain = np.random.uniform(0.2, 5.0, size=N)
        bias = np.random.uniform(0.2, 1.0, size=N)

        with nengo.Network() as net:
            ens = nengo.Ensemble(N, D)
            ens.gain = gain
            ens.bias = bias

        with nengo_dl.Simulator(net, seed=0) as sim:
            ens_data = sim.data[ens]
            # sim.data[ens] is a dict with 'gain', 'bias', etc.
            assert np.allclose(gain, ens_data["gain"])
            assert np.allclose(bias, ens_data["bias"])

    def test_default_gain_bias_not_nan(self):
        """Default gain/bias should be computed without NaN."""
        with nengo.Network(seed=0) as net:
            ens = nengo.Ensemble(20, 2, seed=0)

        with nengo_dl.Simulator(net, seed=0) as sim:
            ens_data = sim.data[ens]
            assert "gain" in ens_data
            assert not np.any(np.isnan(ens_data["gain"]))
            assert not np.any(np.isnan(ens_data["bias"]))


# ---------------------------------------------------------------------------
# reset_state
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_state_clears_probe_data(self):
        """After reset_state(), probe data should start fresh."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.ones(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
            assert sim.data[p].shape[0] == 5
            sim.reset_state()
            sim.run_steps(2)
            assert sim.data[p].shape[0] == 2

    def test_reset_state_resets_time(self):
        """After reset_state(), n_steps and time should be 0."""
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run_steps(5)
            assert sim.n_steps == 5
            sim.reset_state()
            assert sim.n_steps == 0
            assert sim.time == 0.0


# ---------------------------------------------------------------------------
# Probe data shapes
# ---------------------------------------------------------------------------

class TestProbeShapes:
    def test_node_probe_shape(self):
        """Node probe: (n_steps, size_out)."""
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(3))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(7)

        assert sim.data[p].shape == (7, 3)

    def test_ensemble_probe_shape(self):
        """Ensemble probe: (n_steps, dimensions)."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            ens = nengo.Ensemble(20, 2, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(4)

        assert sim.data[p].shape == (4, 2)

    def test_batched_probe_shape(self):
        """Batched run: (batch, n_steps, size)."""
        bs = 3
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(2))
            p = nengo.Probe(inp, synapse=None)

        x = np.ones((bs, 5, 2), dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=bs, seed=0) as sim:
            sim.run_steps(5, data={inp: x})

        assert sim.data[p].shape == (bs, 5, 2)
