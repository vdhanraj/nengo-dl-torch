"""TorchNode: embed arbitrary PyTorch modules inside a Nengo network.

Analogous to NengoDL's ``TensorNode`` but using PyTorch ``nn.Module``
instead of Keras layers. Supports gradient flow through layers during training.
"""

import warnings
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nengo
import nengo.neurons
from nengo.builder import Builder as NengoBuilder
from nengo.builder.node import build_node
from nengo.builder.operator import Operator
from nengo.builder.signal import Signal

from .builder import Builder, BuildConfig, OpBuilder


# ---------------------------------------------------------------------------
# Neuron rate modules (differentiable activations for NeuronType layers)
# ---------------------------------------------------------------------------

class _LIFRateModule(nn.Module):
    """Differentiable LIF rate activation.

    Uses intercepts=0 convention (threshold at x=0), matching the standard
    Nengo practice of configure_settings(intercepts=[0]) for Layer-API networks.
    """
    def __init__(self, tau_rc, tau_ref, amplitude):
        super().__init__()
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.amplitude = amplitude

    def forward(self, x):
        # intercepts=0: threshold at x=0 (neurons fire for any positive input)
        j = x
        j_safe = torch.clamp(j, min=1e-6)
        rate = self.amplitude / (self.tau_ref + self.tau_rc * torch.log1p(1.0 / j_safe))
        return torch.where(j > 0, rate, torch.zeros_like(rate))


class _SoftLIFRateModule(nn.Module):
    """Smooth differentiable LIF rate activation.

    Uses intercepts=0 convention (threshold at x=0), giving non-zero gradients
    for all inputs and matching the standard practice of intercepts=[0].
    """
    def __init__(self, tau_rc, tau_ref, amplitude, sigma):
        super().__init__()
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.amplitude = amplitude
        self.sigma = sigma

    def forward(self, x):
        sigma = self.sigma
        # intercepts=0: threshold at x=0
        u = x / sigma
        # Numerically stable softplus
        safe_u = torch.clamp(u, max=20.0)
        x_smooth = torch.where(u > 20.0, u * sigma, F.softplus(safe_u) * sigma)
        x_safe = torch.clamp(x_smooth, min=1e-8)
        rate = self.amplitude / (self.tau_ref + self.tau_rc * torch.log1p(1.0 / x_safe))
        return rate * (x_smooth > 1e-8).float()


class _ReLURateModule(nn.Module):
    """RectifiedLinear rate activation."""
    def __init__(self, amplitude):
        super().__init__()
        self.amplitude = amplitude

    def forward(self, x):
        return F.relu(x) * self.amplitude


class _SpikingReLURateModule(nn.Module):
    """Rate approximation of SpikingRectifiedLinear."""
    def __init__(self, amplitude):
        super().__init__()
        self.amplitude = amplitude

    def forward(self, x):
        return F.relu(x) * self.amplitude


class _LeakyReLURateModule(nn.Module):
    """Leaky ReLU rate activation."""
    def __init__(self, negative_slope, amplitude):
        super().__init__()
        self.negative_slope = negative_slope
        self.amplitude = amplitude

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope) * self.amplitude


def neuron_type_to_module(neuron_type) -> nn.Module:
    """Convert a Nengo NeuronType to a differentiable PyTorch rate module."""
    from .neurons import SoftLIFRate, LeakyReLU, SpikingLeakyReLU

    if isinstance(neuron_type, nengo.LIF):
        return _LIFRateModule(neuron_type.tau_rc, neuron_type.tau_ref, neuron_type.amplitude)
    elif isinstance(neuron_type, nengo.LIFRate):
        return _LIFRateModule(neuron_type.tau_rc, neuron_type.tau_ref, neuron_type.amplitude)
    elif isinstance(neuron_type, SoftLIFRate):
        return _SoftLIFRateModule(
            neuron_type.tau_rc, neuron_type.tau_ref,
            neuron_type.amplitude, neuron_type.sigma
        )
    elif isinstance(neuron_type, nengo.RectifiedLinear):
        return _ReLURateModule(neuron_type.amplitude)
    elif isinstance(neuron_type, nengo.SpikingRectifiedLinear):
        return _SpikingReLURateModule(neuron_type.amplitude)
    elif isinstance(neuron_type, LeakyReLU):
        return _LeakyReLURateModule(neuron_type.negative_slope, neuron_type.amplitude)
    elif isinstance(neuron_type, SpikingLeakyReLU):
        return _LeakyReLURateModule(neuron_type.negative_slope, neuron_type.amplitude)
    elif isinstance(neuron_type, nengo.Sigmoid):
        class _SigmoidModule(nn.Module):
            def __init__(self, tau_ref, amplitude):
                super().__init__()
                self.tau_ref = tau_ref
                self.amplitude = amplitude
            def forward(self, x):
                return torch.sigmoid(x / self.tau_ref) * self.amplitude
        return _SigmoidModule(neuron_type.tau_ref, neuron_type.amplitude)
    elif isinstance(neuron_type, nengo.Tanh):
        class _TanhModule(nn.Module):
            def __init__(self, amplitude):
                super().__init__()
                self.amplitude = amplitude
            def forward(self, x):
                return torch.tanh(x) * self.amplitude
        return _TanhModule(neuron_type.amplitude)
    else:
        # Fallback: identity
        warnings.warn(
            f"Unsupported neuron type {type(neuron_type).__name__} for Layer; "
            "using identity activation."
        )
        return nn.Identity()


# ---------------------------------------------------------------------------
# Wrapper to handle spatial reshaping for conv layers
# ---------------------------------------------------------------------------

class _SpatialModule(nn.Module):
    """Wraps a spatial PyTorch module (e.g. Conv2d) for flat I/O.

    Nengo nodes operate on flat 1-D vectors. This wrapper:
    1. Reshapes the flat input to ``shape_in`` (HWC → NCHW)
    2. Passes through the module
    3. Flattens the output back to a 1-D vector
    """

    def __init__(self, module: nn.Module, shape_in: tuple):
        super().__init__()
        self.module = module
        self.shape_in = shape_in  # (H, W, C) or (C, H, W) or flat tuple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, flat_in)
        batch = x.shape[0]
        # Reshape to spatial: assume shape_in = (H, W, C) → NCHW for PyTorch
        if len(self.shape_in) == 3:
            h, w, c = self.shape_in
            x_spatial = x.reshape(batch, h, w, c).permute(0, 3, 1, 2)  # NHWC → NCHW
        elif len(self.shape_in) == 2:
            h, w = self.shape_in
            x_spatial = x.reshape(batch, 1, h, w)
        else:
            x_spatial = x.reshape(batch, *self.shape_in)

        out = self.module(x_spatial)

        # Flatten output: NCHW → N*(C*H*W)
        return out.reshape(batch, -1)


# ---------------------------------------------------------------------------
# SimTorchNode: operator
# ---------------------------------------------------------------------------

class SimTorchNode(Operator):
    """Operator for executing a TorchNode during simulation.

    Stores the nn.Module directly so the builder can call it with
    PyTorch tensors, enabling gradient flow through the module.
    """

    def __init__(self, node, module, output, input=None, t=None, tag=None):
        super().__init__(tag=tag)
        self.node = node
        self.module = module  # nn.Module to execute (may be None)
        self._input = input
        self._output = output
        self._t = t

        self.reads = [s for s in [t, input] if s is not None]
        self.sets = [output]
        self.updates = []
        self.incs = []

    def make_step(self, signals, dt, rng):
        raise NotImplementedError("SimTorchNode uses PyTorch execution path")


# ---------------------------------------------------------------------------
# SimTorchNodeBuilder: executes SimTorchNode with full PyTorch autograd
# ---------------------------------------------------------------------------

@Builder.register(SimTorchNode)
class SimTorchNodeBuilder(OpBuilder):
    """Executes TorchNode operators using PyTorch tensors with gradient flow."""

    def build_pre(self, ops, signals, config):
        self._op_list = []
        for op in ops:
            # Pre-build a smooth version of LIF rate modules for use during
            # training when lif_smoothing > 0. This enables non-zero gradients
            # everywhere (avoids dead neuron problem with hard threshold).
            smooth_module = None
            if config.lif_smoothing > 0 and isinstance(op.module, _LIFRateModule):
                smooth_module = _SoftLIFRateModule(
                    op.module.tau_rc, op.module.tau_ref,
                    op.module.amplitude, config.lif_smoothing,
                ).to(config.device)
            self._op_list.append({
                "module": op.module,
                "smooth_module": smooth_module,
                "input_sig": op._input,
                "output_sig": op._output,
                "t_sig": op._t,
            })

    def build_step(self, ops, signals, config):
        for op_info in self._op_list:
            t_sig = op_info["t_sig"]
            x_sig = op_info["input_sig"]
            out_sig = op_info["output_sig"]

            # Use smooth LIF module during training if lif_smoothing is set
            smooth = op_info["smooth_module"]
            if config.training and config.lif_smoothing > 0 and smooth is not None:
                module = smooth
            else:
                module = op_info["module"]

            t_val = signals.gather(t_sig)
            t = float(t_val.flatten()[0].item())

            if x_sig is not None:
                x = signals.gather(x_sig).to(config.dtype)
                # x: (batch, *shape) – ensure 2D (batch, flat)
                batch = x.shape[0]
                if x.dim() > 2:
                    x = x.reshape(batch, -1)

                if module is not None:
                    out = module(x)
                else:
                    out = x
            else:
                if module is not None:
                    # Source module: call with no input
                    dummy = torch.zeros(config.minibatch_size, 1,
                                       dtype=config.dtype, device=config.device)
                    out = module(dummy)
                else:
                    continue

            # out: (batch, flat_out) - scatter to output signal
            signals.scatter(out_sig, out.to(config.dtype), mode="set")


# ---------------------------------------------------------------------------
# TorchNode: Nengo Node wrapping an nn.Module
# ---------------------------------------------------------------------------

class TorchNode(nengo.Node):
    """A Nengo Node that wraps a PyTorch ``nn.Module`` or callable.

    The wrapped module is executed as part of the nengo-dl simulation graph
    and supports gradient-based training.

    Parameters
    ----------
    torch_func : nn.Module or callable
        A PyTorch module or callable. For spatial inputs (e.g. Conv2d), wrap
        with ``shape_in`` to enable automatic reshaping.
    size_in : int, optional
        Number of input dimensions.
    size_out : int, optional
        Number of output dimensions. If None, inferred via dummy forward pass.
    shape_in : tuple, optional
        Spatial input shape e.g. ``(28, 28, 1)`` for 28×28 greyscale.
    shape_out : tuple, optional
        Output shape (excluding batch).
    pass_time : bool, optional
        If True, pass simulation time as first argument to module.
    label : str, optional
        Human-readable name.
    """

    def __init__(
        self,
        torch_func: Union[nn.Module, Callable],
        size_in: Optional[int] = None,
        size_out: Optional[int] = None,
        shape_in: Optional[Tuple] = None,
        shape_out: Optional[Tuple] = None,
        pass_time: bool = False,
        label: Optional[str] = None,
    ):
        if size_in is None and shape_in is not None:
            size_in = int(np.prod(shape_in))
        if size_out is None and shape_out is not None:
            size_out = int(np.prod(shape_out))

        if size_out is None:
            raise ValueError(
                "TorchNode requires size_out or shape_out to determine output size."
            )

        self.torch_func = torch_func
        self.shape_in = shape_in if shape_in is not None else ((size_in,) if size_in else ())
        self.shape_out = shape_out if shape_out is not None else (size_out,)
        self.pass_time = pass_time

        # The module to register for parameter tracking
        self._module: Optional[nn.Module] = None
        if isinstance(torch_func, nn.Module):
            self._module = torch_func

        nengo_size_in = size_in if size_in and size_in > 0 else None

        def _output_fn(t, x=None):
            """Placeholder; actual execution done in nengo-dl."""
            return np.zeros(size_out, dtype=np.float32)

        super().__init__(
            output=_output_fn if nengo_size_in is not None else lambda t: np.zeros(size_out),
            size_in=nengo_size_in,
            size_out=size_out,
            label=label or f"TorchNode({type(torch_func).__name__})",
        )

    def get_module(self) -> Optional[nn.Module]:
        """Return the underlying nn.Module, or None."""
        return self._module


# ---------------------------------------------------------------------------
# Nengo builder integration: build_node override for TorchNode
# ---------------------------------------------------------------------------

@NengoBuilder.register(TorchNode)
def build_torch_node(model, node):
    """Build a TorchNode: register signals and create a SimTorchNode operator."""
    from nengo.builder.operator import SimPyFunc

    # Let the standard node builder create the signals and SimPyFunc operator
    build_node(model, node)

    # The standard builder adds a SimPyFunc. Find it and extract the signals,
    # then replace it with a SimTorchNode.
    out_sig = model.sig[node].get("out")
    if out_sig is None:
        return

    # Find the SimPyFunc that build_node added for this node
    found_op = None
    for op in model.operators:
        if isinstance(op, SimPyFunc) and hasattr(op, 'output') and op.output is out_sig:
            found_op = op
            break

    if found_op is None:
        # No SimPyFunc found; nothing to replace
        return

    # Extract t and x signals from the found SimPyFunc
    t_sig = found_op.t
    x_sig = found_op.x

    # Determine the module to execute
    module = node._module

    # Create the SimTorchNode operator
    torch_op = SimTorchNode(
        node=node,
        module=module,
        output=out_sig,
        input=x_sig,
        t=t_sig,
    )

    # Replace SimPyFunc with SimTorchNode in model.operators
    new_ops = []
    for op in model.operators:
        if op is found_op:
            new_ops.append(torch_op)
        else:
            new_ops.append(op)
    model.operators[:] = new_ops


# ---------------------------------------------------------------------------
# Layer: Keras-style functional API
# ---------------------------------------------------------------------------

class Layer:
    """Convenience wrapper for using a layer API in Nengo networks.

    Supports both ``nn.Module`` layers (Conv2d, Linear, etc.) and Nengo
    ``NeuronType`` objects (LIF, RectifiedLinear, etc.) as activation layers.

    Examples
    --------
    ::

        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(784))

            # Convolutional layer (PyTorch Conv2d)
            x = nengo_dl.Layer(nn.Conv2d(1, 32, 3))(inp, shape_in=(28, 28, 1))

            # Neuron activation layer
            x = nengo_dl.Layer(nengo.LIF(amplitude=0.01))(x)

            # Linear readout
            out = nengo_dl.Layer(nn.Linear(21632, 10))(x)
    """

    def __init__(self, layer):
        """
        Parameters
        ----------
        layer : nn.Module or NeuronType
            The layer or activation to apply.
        """
        self.layer = layer

    def __call__(
        self,
        nengo_input,
        shape_in: Optional[Tuple] = None,
        shape_out: Optional[Tuple] = None,
        synapse=None,
        label: Optional[str] = None,
        transform=1,
    ) -> TorchNode:
        """Apply the layer to a Nengo object.

        Parameters
        ----------
        nengo_input : nengo object
            Input connection source (Ensemble, Node, TorchNode, etc.).
        shape_in : tuple, optional
            Spatial input shape e.g. ``(28, 28, 1)`` for a 28×28 greyscale image.
            Required for convolutional layers.
        shape_out : tuple, optional
            Output shape. Inferred from layer if not given.
        synapse : float or Synapse, optional
            Synapse for the input connection.
        label : str, optional
            Label for the created node.
        transform : float or array, optional
            Transform for the input connection.

        Returns
        -------
        TorchNode
            The created node (which can be used as input to further layers).
        """
        layer = self.layer

        # Determine input size
        if shape_in is not None:
            size_in = int(np.prod(shape_in))
        elif hasattr(nengo_input, "size_out"):
            size_in = nengo_input.size_out
            shape_in = (size_in,)
        elif hasattr(nengo_input, "dimensions"):
            size_in = nengo_input.dimensions
            shape_in = (size_in,)
        else:
            raise ValueError(
                "Cannot determine input size. Provide shape_in explicitly."
            )

        # Handle NeuronType (activation layer)
        if isinstance(layer, nengo.neurons.NeuronType):
            module = neuron_type_to_module(layer)
            size_out = size_in
            shape_out = shape_in

            node = TorchNode(
                module,
                size_in=size_in,
                size_out=size_out,
                shape_in=shape_in,
                shape_out=shape_out,
                label=label or f"Layer({type(layer).__name__})",
            )
            nengo.Connection(nengo_input, node, synapse=synapse, transform=transform)
            return node

        # Handle nn.Module layers
        if isinstance(layer, nn.Module):
            # Check if this is a spatial layer (needs reshape)
            is_spatial = (
                len(shape_in) >= 2 and
                isinstance(layer, (nn.Conv2d, nn.Conv1d, nn.MaxPool2d,
                                   nn.AvgPool2d, nn.BatchNorm2d))
            )

            if is_spatial:
                # Wrap with spatial reshaping module
                module = _SpatialModule(layer, shape_in)
            else:
                module = layer

            # Compute output size via dummy forward pass
            if shape_out is None:
                dummy_in = torch.zeros(1, size_in)
                with torch.no_grad():
                    try:
                        dummy_out = module(dummy_in)
                        size_out = int(np.prod(dummy_out.shape[1:]))
                        shape_out = tuple(dummy_out.shape[1:])
                    except Exception as e:
                        raise ValueError(
                            f"Cannot infer output shape for layer {layer}. "
                            f"Error: {e}. Provide shape_out explicitly."
                        )
            else:
                size_out = int(np.prod(shape_out))

            node = TorchNode(
                module,
                size_in=size_in,
                size_out=size_out,
                shape_in=shape_in,
                shape_out=shape_out,
                label=label or f"Layer({type(layer).__name__})",
            )
            nengo.Connection(nengo_input, node, synapse=synapse, transform=transform)
            return node

        # Fallback: treat as callable, wrap in a Module
        if callable(layer):
            class _CallableModule(nn.Module):
                def __init__(self, fn):
                    super().__init__()
                    self.fn = fn
                def forward(self, x):
                    return self.fn(x)

            module = _CallableModule(layer)
            if shape_out is None:
                dummy = torch.zeros(1, size_in)
                with torch.no_grad():
                    dummy_out = module(dummy)
                size_out = int(np.prod(dummy_out.shape[1:]))
                shape_out = tuple(dummy_out.shape[1:])
            else:
                size_out = int(np.prod(shape_out))

            node = TorchNode(
                module,
                size_in=size_in,
                size_out=size_out,
                shape_in=shape_in,
                shape_out=shape_out,
                label=label or "Layer(callable)",
            )
            nengo.Connection(nengo_input, node, synapse=synapse, transform=transform)
            return node

        raise TypeError(
            f"Layer expects an nn.Module, NeuronType, or callable; got {type(layer)}"
        )
