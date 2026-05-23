"""Builders for Nengo neuron operators (PyTorch backend).

Implements LIF, LIFRate, RectifiedLinear, Sigmoid, Tanh, Direct, and
custom nengo-dl neuron types. Spiking neurons use surrogate gradients
during training to allow backpropagation.
"""

import numpy as np
import torch
import torch.nn.functional as F

import nengo
import nengo.neurons
from nengo.builder.neurons import SimNeurons

from .builder import Builder, BuildConfig, OpBuilder
from .neurons import SoftLIFRate, SpikingLeakyReLU, LeakyReLU


# ---------------------------------------------------------------------------
# Surrogate gradient helpers
# ---------------------------------------------------------------------------

class _SpikeFunction(torch.autograd.Function):
    """Heaviside spike with SuperSpike surrogate gradient.

    Forward: ``(v >= threshold)``.
    Backward: fast-sigmoid surrogate ``1 / (1 + |sharpness * (v - thresh)|)^2``.
    """

    @staticmethod
    def forward(ctx, v, threshold, sharpness):
        ctx.save_for_backward(v - threshold)
        ctx.sharpness = sharpness
        return (v >= threshold).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        delta, = ctx.saved_tensors
        sg = 1.0 / (1.0 + ctx.sharpness * delta.abs()) ** 2
        return grad_output * sg, None, None


def spike_fn(v, threshold=1.0, sharpness=10.0):
    """Apply spike function with surrogate gradient for training."""
    return _SpikeFunction.apply(v, threshold, sharpness)


# ---------------------------------------------------------------------------
# LIF  (spiking)
# ---------------------------------------------------------------------------

@Builder.register(SimNeurons)
class SimNeuronsBuilder(OpBuilder):
    """Dispatches to a neuron-type-specific implementation."""

    # Registry: neuron_type class → builder function
    _neuron_builders = {}

    @classmethod
    def register_neuron(cls, neuron_type_cls):
        """Decorator to register a per-neuron-type build_step function."""
        def decorator(fn):
            cls._neuron_builders[neuron_type_cls] = fn
            return fn
        return decorator

    def build_pre(self, ops, signals, config):
        self._ops = ops
        # Pre-build each op's neuron state
        self._neuron_state = []
        for op in ops:
            state = {}
            for key, sig in op.state.items():
                state[key] = sig
            self._neuron_state.append(state)

    def build_step(self, ops, signals, config):
        for op, state_sigs in zip(self._ops, self._neuron_state):
            neurons = op.neurons
            neuron_cls = type(neurons)

            builder_fn = None
            for cls in type(neurons).__mro__:
                if cls in SimNeuronsBuilder._neuron_builders:
                    builder_fn = SimNeuronsBuilder._neuron_builders[cls]
                    break

            if builder_fn is not None:
                builder_fn(op, state_sigs, signals, config)
            else:
                # Fallback: run the numpy step function
                _generic_neuron_step(op, state_sigs, signals, config)


# ---------------------------------------------------------------------------
# Generic (numpy fallback)
# ---------------------------------------------------------------------------

def _generic_neuron_step(op, state_sigs, signals, config):
    """Execute a neuron step using Nengo's numpy implementation."""
    J = signals.gather(op.J)  # (batch, n_neurons)
    batch = J.shape[0]

    # Build state dict with numpy arrays (use batch item 0)
    state_np = {}
    for key, sig in state_sigs.items():
        state_np[key] = signals.gather(sig)[0].detach().cpu().numpy().copy()

    output_list = []
    state_out_np = {k: [] for k in state_sigs}

    for b in range(batch):
        J_np = J[b].detach().cpu().numpy().copy()
        out_np = np.zeros(op.output.shape, dtype=np.float32)
        st_b = {k: v.copy() for k, v in state_np.items()}

        # Call the Nengo numpy step function
        step_fn = op.neurons.step
        import inspect
        sig_names = list(inspect.signature(step_fn).parameters.keys())
        # step(dt, J, output, **state)
        kwargs = {k: st_b[k] for k in state_sigs if k in sig_names}
        step_fn(config.dt, J_np, out_np, **kwargs)

        output_list.append(out_np)
        for k in state_sigs:
            state_out_np[k].append(st_b[k])

    output = torch.tensor(
        np.stack(output_list), dtype=config.dtype, device=config.device
    )
    signals.scatter(op.output, output, mode="set")

    for key, sig in state_sigs.items():
        state_val = torch.tensor(
            np.stack(state_out_np[key]), dtype=config.dtype, device=config.device
        )
        signals.scatter(sig, state_val, mode="set")


# ---------------------------------------------------------------------------
# LIF (spiking) – registered for nengo.LIF
# ---------------------------------------------------------------------------

@SimNeuronsBuilder.register_neuron(nengo.LIF)
def _lif_step(op, state_sigs, signals, config):
    """LIF spiking neuron with surrogate gradients for training."""
    neurons: nengo.LIF = op.neurons
    dt = config.dt
    tau_rc = neurons.tau_rc
    tau_ref = neurons.tau_ref
    amplitude = neurons.amplitude

    J = signals.gather(op.J)  # (batch, n)
    V = signals.gather(state_sigs["voltage"])   # (batch, n)
    R = signals.gather(state_sigs["refractory_time"])  # (batch, n)

    # Ensure float
    J = J.to(config.dtype)
    V = V.to(config.dtype)
    R = R.to(config.dtype)

    if config.training:
        # During training, swap spiking → SoftLIF rate neurons (matches original NengoDL
        # behavior: "spiking neurons are automatically being swapped for differentiable
        # rate neurons"). SoftLIF avoids the infinite gradient of hard rate LIF near
        # threshold. Default sigma=0.002 matches original NengoDL's SoftLIFRate default.
        sigma = config.lif_smoothing if config.lif_smoothing > 0 else 0.002
        output = _soft_lif_rate(J, tau_rc, tau_ref, amplitude, sigma)
        signals.scatter(op.output, output, mode="set")
        return

    # ------ Exact LIF spiking (with surrogate gradient) ------
    # Decay constant
    decay = float(np.exp(-dt / tau_rc))

    # Voltage update for non-refractory neurons
    refractory = (R > 0.5 * dt)  # boolean mask
    dV = J * (1.0 - decay) + V * (decay - 1.0)  # = (J - V) * (1 - decay)
    V_new = V + dV
    V_new = torch.where(refractory, V, V_new)
    V_new = torch.clamp(V_new, max=2.0)  # numerical stability

    # Spike detection
    if config.training:
        spiked = spike_fn(V_new, threshold=1.0, sharpness=10.0)
    else:
        spiked = (V_new >= 1.0).to(J.dtype)

    # Reset after spike
    V_new = V_new * (1.0 - spiked)

    # Update refractory time
    R_new = torch.clamp(R - dt, min=0.0)
    R_new = torch.where(spiked.bool(), torch.full_like(R_new, tau_ref), R_new)

    # Output: spike rate in Hz
    output = spiked * (amplitude / dt)

    signals.scatter(op.output, output, mode="set")
    signals.scatter(state_sigs["voltage"], V_new, mode="set")
    signals.scatter(state_sigs["refractory_time"], R_new, mode="set")


# ---------------------------------------------------------------------------
# LIFRate – registered for nengo.LIFRate
# ---------------------------------------------------------------------------

@SimNeuronsBuilder.register_neuron(nengo.LIFRate)
def _lif_rate_step(op, state_sigs, signals, config):
    """LIF rate neuron (fully differentiable)."""
    neurons: nengo.LIFRate = op.neurons
    J = signals.gather(op.J).to(config.dtype)
    output = _lif_rate(J, neurons.tau_rc, neurons.tau_ref, neurons.amplitude)
    signals.scatter(op.output, output, mode="set")


def _lif_rate(J, tau_rc, tau_ref, amplitude=1.0):
    """Exact LIF rate function (differentiable via autograd)."""
    # Rate = 1 / (tau_ref + tau_rc * log(1 + 1 / max(J - 1, eps)))
    x = J - 1.0
    # Use clamp to avoid log(0) or negative args
    x_safe = torch.clamp(x, min=1e-6)
    rate = amplitude / (tau_ref + tau_rc * torch.log1p(1.0 / x_safe))
    rate = torch.where(x > 0, rate, torch.zeros_like(rate))
    return rate


def _soft_lif_rate(J, tau_rc, tau_ref, amplitude, sigma):
    """Smoothed LIF rate (differentiable everywhere)."""
    # Use softplus for smooth threshold
    x = F.softplus((J - 1.0) / sigma) * sigma
    x_safe = torch.clamp(x, min=1e-8)
    rate = amplitude / (tau_ref + tau_rc * torch.log1p(1.0 / x_safe))
    rate = rate * (x > 1e-8).float()
    return rate


# ---------------------------------------------------------------------------
# RectifiedLinear – nengo.RectifiedLinear
# ---------------------------------------------------------------------------

@SimNeuronsBuilder.register_neuron(nengo.RectifiedLinear)
def _relu_step(op, state_sigs, signals, config):
    """ReLU rate neuron."""
    neurons: nengo.RectifiedLinear = op.neurons
    J = signals.gather(op.J).to(config.dtype)
    output = F.relu(J) * neurons.amplitude
    signals.scatter(op.output, output, mode="set")


# ---------------------------------------------------------------------------
# SpikingRectifiedLinear – nengo.SpikingRectifiedLinear
# ---------------------------------------------------------------------------

@SimNeuronsBuilder.register_neuron(nengo.SpikingRectifiedLinear)
def _spiking_relu_step(op, state_sigs, signals, config):
    """Spiking ReLU neuron."""
    neurons: nengo.SpikingRectifiedLinear = op.neurons
    dt = config.dt
    J = signals.gather(op.J).to(config.dtype)
    V = signals.gather(state_sigs["voltage"]).to(config.dtype)

    if config.training:
        # Use rate approximation for training
        output = F.relu(J) * neurons.amplitude
        signals.scatter(op.output, output, mode="set")
        return

    J_rect = F.relu(J)
    V_new = V + J_rect * dt
    spiked = (V_new >= 1.0).to(J.dtype)
    V_new = V_new - spiked  # reset by 1 after spike
    output = spiked * (neurons.amplitude / dt)
    signals.scatter(op.output, output, mode="set")
    signals.scatter(state_sigs["voltage"], V_new, mode="set")


# ---------------------------------------------------------------------------
# Sigmoid – nengo.Sigmoid
# ---------------------------------------------------------------------------

@SimNeuronsBuilder.register_neuron(nengo.Sigmoid)
def _sigmoid_step(op, state_sigs, signals, config):
    """Sigmoid rate neuron."""
    neurons: nengo.Sigmoid = op.neurons
    J = signals.gather(op.J).to(config.dtype)
    output = torch.sigmoid(J / neurons.tau_ref) * neurons.amplitude
    signals.scatter(op.output, output, mode="set")


# ---------------------------------------------------------------------------
# Tanh – nengo.Tanh
# ---------------------------------------------------------------------------

@SimNeuronsBuilder.register_neuron(nengo.Tanh)
def _tanh_step(op, state_sigs, signals, config):
    """Tanh rate neuron."""
    neurons: nengo.Tanh = op.neurons
    J = signals.gather(op.J).to(config.dtype)
    output = torch.tanh(J) * neurons.amplitude
    signals.scatter(op.output, output, mode="set")


# ---------------------------------------------------------------------------
# Direct – nengo.Direct (passthrough)
# ---------------------------------------------------------------------------

@SimNeuronsBuilder.register_neuron(nengo.Direct)
def _direct_step(op, state_sigs, signals, config):
    """Direct (passthrough) neuron."""
    J = signals.gather(op.J).to(config.dtype)
    signals.scatter(op.output, J, mode="set")


# ---------------------------------------------------------------------------
# AdaptiveLIF
# ---------------------------------------------------------------------------

@SimNeuronsBuilder.register_neuron(nengo.AdaptiveLIF)
def _adaptive_lif_step(op, state_sigs, signals, config):
    """Adaptive LIF neuron."""
    neurons: nengo.AdaptiveLIF = op.neurons
    dt = config.dt

    J = signals.gather(op.J).to(config.dtype)
    V = signals.gather(state_sigs["voltage"]).to(config.dtype)
    R = signals.gather(state_sigs["refractory_time"]).to(config.dtype)
    A = signals.gather(state_sigs["adaptation"]).to(config.dtype)

    tau_rc = neurons.tau_rc
    tau_ref = neurons.tau_ref
    tau_n = neurons.tau_n
    inc_n = neurons.inc_n
    amplitude = neurons.amplitude

    if config.training:
        J_eff = J - A
        sigma = config.lif_smoothing if config.lif_smoothing > 0 else 0.002
        output = _soft_lif_rate(J_eff, tau_rc, tau_ref, amplitude, sigma)
        signals.scatter(op.output, output, mode="set")
        return

    J_eff = J - A
    decay = float(np.exp(-dt / tau_rc))
    refractory = (R > 0.5 * dt)
    dV = J_eff * (1.0 - decay) + V * (decay - 1.0)
    V_new = V + dV
    V_new = torch.where(refractory, V, V_new)

    if config.training:
        spiked = spike_fn(V_new)
    else:
        spiked = (V_new >= 1.0).to(J.dtype)

    V_new = V_new * (1.0 - spiked)
    R_new = torch.clamp(R - dt, min=0.0)
    R_new = torch.where(spiked.bool(), torch.full_like(R_new, tau_ref), R_new)

    A_decay = float(np.exp(-dt / tau_n))
    A_new = A * A_decay + spiked * inc_n

    output = spiked * (amplitude / dt)
    signals.scatter(op.output, output, mode="set")
    signals.scatter(state_sigs["voltage"], V_new, mode="set")
    signals.scatter(state_sigs["refractory_time"], R_new, mode="set")
    signals.scatter(state_sigs["adaptation"], A_new, mode="set")


# ---------------------------------------------------------------------------
# SoftLIFRate (custom neuron)
# ---------------------------------------------------------------------------

@SimNeuronsBuilder.register_neuron(SoftLIFRate)
def _soft_lif_rate_step(op, state_sigs, signals, config):
    """Soft LIF rate neuron (always differentiable)."""
    neurons: SoftLIFRate = op.neurons
    J = signals.gather(op.J).to(config.dtype)
    output = _soft_lif_rate(
        J, neurons.tau_rc, neurons.tau_ref, neurons.amplitude, neurons.sigma
    )
    signals.scatter(op.output, output, mode="set")


# ---------------------------------------------------------------------------
# SpikingLeakyReLU / LeakyReLU (custom neurons)
# ---------------------------------------------------------------------------

@SimNeuronsBuilder.register_neuron(LeakyReLU)
def _leaky_relu_step(op, state_sigs, signals, config):
    """Leaky ReLU rate neuron."""
    neurons: LeakyReLU = op.neurons
    J = signals.gather(op.J).to(config.dtype)
    output = torch.where(J >= 0, J, J * neurons.negative_slope) * neurons.amplitude
    signals.scatter(op.output, output, mode="set")


@SimNeuronsBuilder.register_neuron(SpikingLeakyReLU)
def _spiking_leaky_relu_step(op, state_sigs, signals, config):
    """Spiking leaky ReLU neuron."""
    neurons: SpikingLeakyReLU = op.neurons
    dt = config.dt

    J = signals.gather(op.J).to(config.dtype)
    V = signals.gather(state_sigs["voltage"]).to(config.dtype)

    if config.training:
        J_eff = torch.where(J >= 0, J, J * neurons.negative_slope)
        output = J_eff * neurons.amplitude
        signals.scatter(op.output, output, mode="set")
        return

    J_eff = torch.where(J >= 0, J, J * neurons.negative_slope)
    V_new = V + J_eff * dt
    spiked = (V_new >= 1.0).to(J.dtype)
    V_new = V_new - spiked
    output = spiked * (neurons.amplitude / dt)
    signals.scatter(op.output, output, mode="set")
    signals.scatter(state_sigs["voltage"], V_new, mode="set")
