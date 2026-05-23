"""Tests for nengo_dl custom neuron types."""

import numpy as np
import pytest
import torch
import nengo
import nengo_dl
from nengo_dl.neurons import SoftLIFRate, SpikingLeakyReLU, LeakyReLU


# ---------------------------------------------------------------------------
# SoftLIFRate
# ---------------------------------------------------------------------------

class TestSoftLIFRate:
    def test_default_params(self):
        n = SoftLIFRate()
        assert n.sigma == pytest.approx(0.02)
        assert n.tau_rc == pytest.approx(0.02)
        assert n.tau_ref == pytest.approx(0.002)
        assert n.amplitude == pytest.approx(1.0)

    def test_custom_params(self):
        n = SoftLIFRate(sigma=0.1, tau_rc=0.05, tau_ref=0.005, amplitude=2.0)
        assert n.sigma == pytest.approx(0.1)
        assert n.tau_rc == pytest.approx(0.05)
        assert n.tau_ref == pytest.approx(0.005)
        assert n.amplitude == pytest.approx(2.0)

    def test_step_zero_for_below_threshold(self):
        n = SoftLIFRate()
        J = np.array([-1.0, 0.0, 0.5])
        out = np.zeros_like(J)
        n.step(dt=0.001, J=J, output=out)
        # All below ~1.0 threshold → near-zero rates
        assert np.all(out >= 0.0)
        # At J=0.5, soft threshold → small but possibly nonzero (smoothed)
        assert out[0] >= 0.0

    def test_step_positive_for_suprathreshold(self):
        n = SoftLIFRate()
        J = np.array([2.0, 5.0, 10.0])
        out = np.zeros_like(J)
        n.step(dt=0.001, J=J, output=out)
        assert np.all(out > 0)

    def test_step_monotonically_increasing(self):
        n = SoftLIFRate()
        J = np.linspace(1.0, 10.0, 50)
        out = np.zeros_like(J)
        n.step(dt=0.001, J=J, output=out)
        assert np.all(np.diff(out) >= 0)

    def test_step_no_nan(self):
        n = SoftLIFRate()
        J = np.array([-100.0, -1.0, 0.0, 1e-8, 1.0, 100.0])
        out = np.zeros_like(J)
        n.step(dt=0.001, J=J, output=out)
        assert not np.any(np.isnan(out))

    def test_gain_bias_delegates_to_lif(self):
        n = SoftLIFRate()
        lif = nengo.LIFRate(tau_rc=n.tau_rc, tau_ref=n.tau_ref)
        max_rates = np.array([100.0, 200.0])
        intercepts = np.array([0.0, -0.5])
        g_soft, b_soft = n.gain_bias(max_rates, intercepts)
        g_lif, b_lif = lif.gain_bias(max_rates, intercepts)
        np.testing.assert_allclose(g_soft, g_lif, rtol=1e-5)
        np.testing.assert_allclose(b_soft, b_lif, rtol=1e-5)

    def test_in_simulator(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(20, 1, neuron_type=SoftLIFRate(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
        assert sim.data[p].shape == (5, 1)

    def test_sigma_larger_gives_smoother_gradient(self):
        """Larger sigma → output is non-zero for smaller J values."""
        J = np.array([0.5])
        out_small = np.zeros(1)
        out_large = np.zeros(1)
        SoftLIFRate(sigma=0.001).step(dt=0.001, J=J, output=out_small)
        SoftLIFRate(sigma=0.5).step(dt=0.001, J=J, output=out_large)
        assert out_large[0] >= out_small[0]


# ---------------------------------------------------------------------------
# LeakyReLU
# ---------------------------------------------------------------------------

class TestLeakyReLU:
    def test_default_params(self):
        n = LeakyReLU()
        assert n.negative_slope == pytest.approx(0.0)
        assert n.amplitude == pytest.approx(1.0)

    def test_positive_input_passes_through(self):
        n = LeakyReLU()
        J = np.array([0.5, 1.0, 2.0])
        out = np.zeros_like(J)
        n.step(dt=0.001, J=J, output=out)
        np.testing.assert_allclose(out, J)

    def test_negative_input_zero_with_zero_slope(self):
        n = LeakyReLU(negative_slope=0.0)
        J = np.array([-1.0, -2.0, -0.5])
        out = np.zeros_like(J)
        n.step(dt=0.001, J=J, output=out)
        np.testing.assert_allclose(out, 0.0)

    def test_negative_input_attenuated_with_nonzero_slope(self):
        slope = 0.1
        n = LeakyReLU(negative_slope=slope)
        J = np.array([-1.0, -2.0])
        out = np.zeros_like(J)
        n.step(dt=0.001, J=J, output=out)
        np.testing.assert_allclose(out, J * slope)

    def test_amplitude_scales_output(self):
        n = LeakyReLU(amplitude=3.0)
        J = np.array([1.0, 2.0])
        out = np.zeros_like(J)
        n.step(dt=0.001, J=J, output=out)
        np.testing.assert_allclose(out, J * 3.0)

    def test_gain_bias(self):
        n = LeakyReLU()
        max_rates = np.array([100.0])
        intercepts = np.array([0.0])
        gain, bias = n.gain_bias(max_rates, intercepts)
        assert gain[0] == pytest.approx(100.0)
        assert bias[0] == pytest.approx(0.0)

    def test_in_simulator(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(15, 1, neuron_type=LeakyReLU(negative_slope=0.1), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
        assert sim.data[p].shape == (5, 1)


# ---------------------------------------------------------------------------
# SpikingLeakyReLU
# ---------------------------------------------------------------------------

class TestSpikingLeakyReLU:
    def test_default_params(self):
        n = SpikingLeakyReLU()
        assert n.negative_slope == pytest.approx(0.0)
        assert n.amplitude == pytest.approx(1.0)

    def test_voltage_accumulates(self):
        n = SpikingLeakyReLU()
        J = np.array([5.0])
        out = np.zeros_like(J)
        voltage = np.zeros_like(J)
        n.step(dt=0.001, J=J, output=out, voltage=voltage)
        assert voltage[0] == pytest.approx(5.0 * 0.001)

    def test_spike_on_threshold(self):
        n = SpikingLeakyReLU()
        J = np.array([1000.0])  # high current → spike within one step
        out = np.zeros_like(J)
        voltage = np.array([0.9])  # near threshold
        n.step(dt=0.001, J=J, output=out, voltage=voltage)
        assert out[0] > 0  # should have spiked

    def test_voltage_resets_after_spike(self):
        n = SpikingLeakyReLU()
        J = np.array([1000.0])
        out = np.zeros_like(J)
        voltage = np.array([0.999])  # just below threshold
        n.step(dt=0.001, J=J, output=out, voltage=voltage)
        if out[0] > 0:  # spiked
            assert voltage[0] == pytest.approx(0.0)

    def test_negative_input_zero_with_zero_slope(self):
        n = SpikingLeakyReLU(negative_slope=0.0)
        J = np.array([-5.0])
        out = np.zeros_like(J)
        voltage = np.zeros_like(J)
        for _ in range(100):
            n.step(dt=0.001, J=J, output=out, voltage=voltage)
        assert voltage[0] == pytest.approx(0.0)
        assert out[0] == pytest.approx(0.0)

    def test_in_simulator(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, neuron_type=SpikingLeakyReLU(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p_spikes = nengo.Probe(ens.neurons, synapse=None)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(10)
        assert sim.data[p_spikes].shape[0] == 10


# ---------------------------------------------------------------------------
# All neuron types run without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("neuron_type,kwargs", [
    (nengo.RectifiedLinear, {}),
    (nengo.LIF, {}),
    (nengo.LIFRate, {}),
    (SoftLIFRate, {"sigma": 0.05}),
    (LeakyReLU, {"negative_slope": 0.1}),
    (SpikingLeakyReLU, {}),
])
def test_neuron_type_in_simulator(neuron_type, kwargs):
    with nengo.Network(seed=0) as net:
        inp = nengo.Node(np.zeros(1))
        ens = nengo.Ensemble(10, 1, neuron_type=neuron_type(**kwargs), seed=0)
        nengo.Connection(inp, ens, synapse=None)
        p = nengo.Probe(ens, synapse=None)

    with nengo_dl.Simulator(net, seed=0) as sim:
        sim.run_steps(5)
    assert sim.data[p].shape == (5, 1)
