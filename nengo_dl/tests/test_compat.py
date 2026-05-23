"""Compatibility tests: nengo-dl vs reference Nengo CPU simulator.

These tests verify that nengo-dl produces outputs consistent with the
reference Nengo simulator for operations that should be numerically identical
(e.g., Node→Node signal passing, deterministic probes).
"""

import numpy as np
import pytest
import nengo
import nengo_dl


# ---------------------------------------------------------------------------
# Node-only networks (no stochastic elements)
# ---------------------------------------------------------------------------

class TestNodeNetworkCompat:
    def test_passthrough_node(self):
        """Node→Node signal should pass through unmodified in both simulators."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([3.14]))
            out = nengo.Node(size_in=1)
            nengo.Connection(inp, out, synapse=None)
            p = nengo.Probe(out, synapse=None)

        with nengo.Simulator(net, seed=0, progress_bar=False) as ref:
            ref.run_steps(5)
            ref_data = ref.data[p].copy()

        with nengo_dl.Simulator(net, seed=0) as dl:
            dl.run_steps(5)
            dl_data = dl.data[p].copy()

        np.testing.assert_allclose(dl_data, ref_data, atol=1e-5)

    def test_scalar_transform(self):
        """Scalar connection transform should match reference."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([2.0]))
            out = nengo.Node(size_in=1)
            nengo.Connection(inp, out, transform=3.0, synapse=None)
            p = nengo.Probe(out, synapse=None)

        with nengo.Simulator(net, seed=0, progress_bar=False) as ref:
            ref.run_steps(3)
            ref_data = ref.data[p].copy()

        with nengo_dl.Simulator(net, seed=0) as dl:
            dl.run_steps(3)
            dl_data = dl.data[p].copy()

        np.testing.assert_allclose(dl_data, ref_data, atol=1e-5)

    def test_matrix_transform(self):
        """Matrix transform should produce the same result in both simulators."""
        W = np.array([[1.0, 0.0], [0.0, 2.0]])
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([3.0, 4.0]))
            out = nengo.Node(size_in=2)
            nengo.Connection(inp, out, transform=W, synapse=None)
            p = nengo.Probe(out, synapse=None)

        with nengo.Simulator(net, seed=0, progress_bar=False) as ref:
            ref.run_steps(1)
            ref_data = ref.data[p].copy()

        with nengo_dl.Simulator(net, seed=0) as dl:
            dl.run_steps(1)
            dl_data = dl.data[p].copy()

        np.testing.assert_allclose(dl_data, ref_data, atol=1e-5)

    def test_probe_shape_matches(self):
        """Probe data shape from nengo-dl should match reference Nengo."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(3))
            out = nengo.Node(size_in=3)
            nengo.Connection(inp, out, synapse=None)
            p = nengo.Probe(out, synapse=None)

        with nengo.Simulator(net, seed=0, progress_bar=False) as ref:
            ref.run_steps(7)

        with nengo_dl.Simulator(net, seed=0) as dl:
            dl.run_steps(7)

        assert dl.data[p].shape == ref.data[p].shape


# ---------------------------------------------------------------------------
# Lowpass synapse (deterministic)
# ---------------------------------------------------------------------------

class TestSynapseCompat:
    def test_lowpass_converges_to_same_value(self):
        """After many steps, both sims should converge to the same steady state."""
        tau = 0.01
        input_val = 2.5
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([input_val]))
            out = nengo.Node(size_in=1)
            nengo.Connection(inp, out, synapse=nengo.synapses.Lowpass(tau))
            p = nengo.Probe(out, synapse=None)

        with nengo.Simulator(net, dt=0.001, seed=0, progress_bar=False) as ref:
            ref.run_steps(200)
            ref_final = ref.data[p][-1, 0]

        with nengo_dl.Simulator(net, dt=0.001, seed=0) as dl:
            dl.run_steps(200)
            dl_final = dl.data[p][-1, 0]

        # Both should approach input_val; check they agree
        assert abs(dl_final - ref_final) < 0.01
        assert abs(dl_final - input_val) < 0.01

    def test_lowpass_no_nan(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.ones(1))
            out = nengo.Node(size_in=1)
            nengo.Connection(inp, out, synapse=0.005)
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as dl:
            dl.run_steps(50)
        assert not np.any(np.isnan(dl.data[p]))


# ---------------------------------------------------------------------------
# Ensemble compatibility
# ---------------------------------------------------------------------------

class TestEnsembleCompat:
    def test_ensemble_probe_shape(self):
        """Ensemble probe shape should agree between simulators."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(20, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=0.005)

        with nengo.Simulator(net, seed=0, progress_bar=False) as ref:
            ref.run_steps(10)

        with nengo_dl.Simulator(net, seed=0) as dl:
            dl.run_steps(10)

        assert dl.data[p].shape == ref.data[p].shape

    def test_ensemble_no_nan(self):
        """Ensemble output should contain no NaN."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            ens = nengo.Ensemble(30, 2, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as dl:
            dl.run_steps(5)
        assert not np.any(np.isnan(dl.data[p]))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_output(self):
        """Two runs with the same seed should give identical output."""
        with nengo.Network(seed=42) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=42)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=42) as sim1:
            sim1.run_steps(5)
            data1 = sim1.data[p].copy()

        with nengo_dl.Simulator(net, seed=42) as sim2:
            sim2.run_steps(5)
            data2 = sim2.data[p].copy()

        np.testing.assert_array_equal(data1, data2)

    def test_different_seeds_differ(self):
        """Different seeds should generally give different outputs."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(20, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=1) as sim1:
            sim1.run_steps(3)
            data1 = sim1.data[p].copy()

        with nengo_dl.Simulator(net, seed=99) as sim2:
            sim2.run_steps(3)
            data2 = sim2.data[p].copy()

        assert not np.array_equal(data1, data2)

    def test_reset_gives_same_output(self):
        """After reset_state(), re-running should give identical probe data."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
            data1 = sim.data[p].copy()
            sim.reset_state()
            sim.run_steps(5)
            data2 = sim.data[p].copy()

        np.testing.assert_array_equal(data1, data2)


# ---------------------------------------------------------------------------
# dt compatibility
# ---------------------------------------------------------------------------

class TestDtCompat:
    def test_probe_length_matches_n_steps(self):
        """Probe data should have exactly n_steps rows."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        n = 13
        with nengo_dl.Simulator(net, dt=0.001, seed=0) as dl:
            dl.run_steps(n)
        assert dl.data[p].shape[0] == n

    def test_custom_dt(self):
        """A non-standard dt should not crash the simulator."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, dt=0.005, seed=0) as dl:
            dl.run_steps(4)
        assert dl.data[p].shape[0] == 4
