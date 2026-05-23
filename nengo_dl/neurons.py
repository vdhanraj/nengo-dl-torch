"""Custom Nengo neuron types for nengo-dl (PyTorch backend).

These neuron types are designed to work well with gradient-based learning.
"""

import numpy as np
import nengo
import nengo.params
from nengo.neurons import NeuronType


class SoftLIFRate(NeuronType):
    """A smooth approximation of LIF rate neurons, suitable for training.

    This neuron type gives a differentiable output that approximates the
    LIF rate response. It is used as the "training" substitute for LIF
    spiking neurons.

    Parameters
    ----------
    sigma : float
        Smoothing parameter. Lower values approach true LIF rate; higher
        values give a smoother (more differentiable) response.
    tau_rc : float
        Membrane RC time constant (seconds).
    tau_ref : float
        Refractory period (seconds).
    amplitude : float
        Scaling factor for output firing rates.
    """

    probeable = ("rates",)

    sigma = nengo.params.NumberParam("sigma", low=0, low_open=True)
    tau_rc = nengo.params.NumberParam("tau_rc", low=0, low_open=True)
    tau_ref = nengo.params.NumberParam("tau_ref", low=0)
    amplitude = nengo.params.NumberParam("amplitude", low=0, low_open=True)

    def __init__(self, sigma=0.02, tau_rc=0.02, tau_ref=0.002, amplitude=1.0):
        super().__init__()
        self.sigma = sigma
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.amplitude = amplitude

    @property
    def _argreprs(self):
        args = []
        if self.sigma != 0.02:
            args.append(f"sigma={self.sigma}")
        if self.tau_rc != 0.02:
            args.append(f"tau_rc={self.tau_rc}")
        if self.tau_ref != 0.002:
            args.append(f"tau_ref={self.tau_ref}")
        if self.amplitude != 1.0:
            args.append(f"amplitude={self.amplitude}")
        return args

    def gain_bias(self, max_rates, intercepts):
        # Delegate to LIFRate for gain/bias computation
        lif = nengo.LIFRate(tau_rc=self.tau_rc, tau_ref=self.tau_ref)
        return lif.gain_bias(max_rates, intercepts)

    def max_rates_intercepts(self, gain, bias):
        lif = nengo.LIFRate(tau_rc=self.tau_rc, tau_ref=self.tau_ref)
        return lif.max_rates_intercepts(gain, bias)

    def step(self, dt, J, output):
        """Compute soft LIF rates (numpy version, numerically stable)."""
        sigma = self.sigma
        u = (J - 1) / sigma
        # Numerically stable softplus: avoids exp overflow for large u
        x = np.where(
            u > 20.0,
            u * sigma,                          # large u: softplus ≈ u
            np.log1p(np.exp(np.minimum(u, 20.0))) * sigma  # safe range
        )
        # Rate: 1 / (tau_ref + tau_rc * log(1 + 1 / x))
        safe_x = np.maximum(x, 1e-8)
        rates = self.amplitude / (self.tau_ref + self.tau_rc * np.log1p(1.0 / safe_x))
        rates = np.where(x > 1e-8, rates, 0.0)
        output[...] = rates


class SpikingLeakyReLU(NeuronType):
    """Spiking version of leaky ReLU.

    Generates spikes with a rate equal to the rectified input current.
    Uses a simple threshold-and-reset mechanism.

    Parameters
    ----------
    negative_slope : float
        Slope for negative inputs.
    amplitude : float
        Scaling factor for spike amplitude.
    """

    probeable = ("spikes", "voltage")
    state = {"voltage": nengo.dists.Uniform(0, 1)}

    negative_slope = nengo.params.NumberParam("negative_slope", low=0)
    amplitude = nengo.params.NumberParam("amplitude", low=0, low_open=True)

    def __init__(self, negative_slope=0.0, amplitude=1.0):
        super().__init__()
        self.negative_slope = negative_slope
        self.amplitude = amplitude

    @property
    def _argreprs(self):
        args = []
        if self.negative_slope != 0.0:
            args.append(f"negative_slope={self.negative_slope}")
        if self.amplitude != 1.0:
            args.append(f"amplitude={self.amplitude}")
        return args

    def gain_bias(self, max_rates, intercepts):
        gain = max_rates / (1.0 - intercepts)
        bias = -intercepts * gain
        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        intercepts = -bias / gain
        max_rates = gain * (1 - intercepts)
        return max_rates, intercepts

    def rates(self, x, gain, bias):
        """Use the rate-mode approximation for decoder computation."""
        J = self.current(x, gain, bias)
        out = np.zeros_like(J)
        LeakyReLU.step(self, dt=1.0, J=J, output=out)
        return out

    def step(self, dt, J, output, voltage):
        """Update voltages and generate spikes."""
        J_eff = np.where(J >= 0, J, self.negative_slope * J)
        voltage += J_eff * dt
        spiked = voltage >= 1.0
        output[...] = spiked / dt * self.amplitude
        voltage[spiked] = 0.0


class LeakyReLU(NeuronType):
    """Leaky ReLU rate neuron (non-spiking).

    Parameters
    ----------
    negative_slope : float
        Slope for negative inputs.
    amplitude : float
        Scaling factor for output.
    """

    probeable = ("rates",)

    negative_slope = nengo.params.NumberParam("negative_slope", low=0)
    amplitude = nengo.params.NumberParam("amplitude", low=0, low_open=True)

    def __init__(self, negative_slope=0.0, amplitude=1.0):
        super().__init__()
        self.negative_slope = negative_slope
        self.amplitude = amplitude

    @property
    def _argreprs(self):
        args = []
        if self.negative_slope != 0.0:
            args.append(f"negative_slope={self.negative_slope}")
        if self.amplitude != 1.0:
            args.append(f"amplitude={self.amplitude}")
        return args

    def gain_bias(self, max_rates, intercepts):
        gain = max_rates / (1.0 - intercepts)
        bias = -intercepts * gain
        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        intercepts = -bias / gain
        max_rates = gain * (1 - intercepts)
        return max_rates, intercepts

    def step(self, dt, J, output):
        output[...] = np.where(J >= 0, J, self.negative_slope * J) * self.amplitude
