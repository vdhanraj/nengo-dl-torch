"""Tests for nengo_dl.process_builders (SimProcess / synaptic filters)."""

import numpy as np
import pytest
import nengo
import nengo_dl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_filtered(process, n_steps=50, dt=0.001, seed=0, bs=1):
    """Run a network with a single synapse applied to a constant input."""
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(np.ones(1))
        out = nengo.Node(size_in=1)
        nengo.Connection(inp, out, synapse=process)
        p = nengo.Probe(out, synapse=None)

    if bs == 1:
        with nengo_dl.Simulator(net, dt=dt, seed=seed) as sim:
            sim.run_steps(n_steps)
            return sim.data[p].copy()  # (n_steps, 1)
    else:
        x = np.ones((bs, n_steps, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, dt=dt, minibatch_size=bs, seed=seed) as sim:
            sim.run_steps(n_steps, data={inp: x})
            return sim.data[p].copy()  # (bs, n_steps, 1)


# ---------------------------------------------------------------------------
# Lowpass synapse
# ---------------------------------------------------------------------------

class TestLowpassSynapse:
    def test_output_shape(self):
        out = _run_filtered(nengo.synapses.Lowpass(tau=0.01))
        assert out.shape == (50, 1)

    def test_output_approaches_one(self):
        """With constant input=1 and Lowpass, output should approach 1."""
        out = _run_filtered(nengo.synapses.Lowpass(tau=0.005), n_steps=100)
        # After 100ms (100 steps at dt=0.001), tau=5ms → nearly converged
        assert out[-1, 0] > 0.9

    def test_output_starts_near_zero(self):
        """Fresh start: first output should be close to 0."""
        out = _run_filtered(nengo.synapses.Lowpass(tau=0.01))
        assert abs(out[0, 0]) < 0.15

    def test_smaller_tau_converges_faster(self):
        out_fast = _run_filtered(nengo.synapses.Lowpass(tau=0.001), n_steps=50)
        out_slow = _run_filtered(nengo.synapses.Lowpass(tau=0.05),  n_steps=50)
        assert out_fast[-1, 0] > out_slow[-1, 0]

    def test_no_nan(self):
        out = _run_filtered(nengo.synapses.Lowpass(tau=0.01), n_steps=100)
        assert not np.any(np.isnan(out))

    def test_monotone_increase_with_constant_input(self):
        out = _run_filtered(nengo.synapses.Lowpass(tau=0.01), n_steps=50)
        diffs = np.diff(out[:, 0])
        assert np.all(diffs >= -1e-6)  # non-decreasing (allow small numerical error)

    def test_batched_lowpass(self):
        out = _run_filtered(nengo.synapses.Lowpass(tau=0.01), n_steps=20, bs=4)
        assert out.shape == (4, 20, 1)
        # All batch items with same input should produce same output
        np.testing.assert_allclose(out[0], out[1], rtol=1e-4)


# ---------------------------------------------------------------------------
# Alpha synapse
# ---------------------------------------------------------------------------

class TestAlphaSynapse:
    def test_output_shape(self):
        out = _run_filtered(nengo.synapses.Alpha(tau=0.01))
        assert out.shape == (50, 1)

    def test_no_nan(self):
        out = _run_filtered(nengo.synapses.Alpha(tau=0.01), n_steps=100)
        assert not np.any(np.isnan(out))

    def test_output_approaches_one(self):
        out = _run_filtered(nengo.synapses.Alpha(tau=0.005), n_steps=200)
        assert out[-1, 0] > 0.8


# ---------------------------------------------------------------------------
# No synapse (None)
# ---------------------------------------------------------------------------

class TestNoSynapse:
    def test_passthrough(self):
        """synapse=None: output should equal input directly."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([7.0]))
            out = nengo.Node(size_in=1)
            nengo.Connection(inp, out, synapse=None)
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(3)
        np.testing.assert_allclose(sim.data[p][:, 0], 7.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Synapse on ensemble probe
# ---------------------------------------------------------------------------

class TestSynapseOnProbe:
    def test_probe_synapse_smooths_output(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(20, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p_raw = nengo.Probe(ens, synapse=None)
            p_filt = nengo.Probe(ens, synapse=0.01)

        x = np.ones((1, 30, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(30, data={inp: x})
            raw = sim.data[p_raw]
            filt = sim.data[p_filt]

        # Filtered output has lower variance than raw
        assert np.std(filt) <= np.std(raw) + 1e-6


# ---------------------------------------------------------------------------
# Process via full simulation comparison with Nengo
# ---------------------------------------------------------------------------

class TestLowpassMatchesNengo:
    def test_lowpass_matches_reference(self):
        """nengo-dl Lowpass output should match the analytical step response."""
        tau = 0.01
        dt = 0.001
        n_steps = 50
        input_val = 1.5

        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([input_val]))
            out = nengo.Node(size_in=1)
            nengo.Connection(inp, out, synapse=nengo.synapses.Lowpass(tau))
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(net, dt=dt, seed=0) as dl_sim:
            dl_sim.run_steps(n_steps)
            dl_out = dl_sim.data[p]

        # Analytical: y[k] = input_val * (1 - alpha^k), alpha = exp(-dt/tau)
        alpha = np.exp(-dt / tau)
        ks = np.arange(1, n_steps + 1)
        expected = input_val * (1.0 - alpha ** ks)

        np.testing.assert_allclose(dl_out[:, 0], expected, rtol=1e-3, atol=1e-4)
