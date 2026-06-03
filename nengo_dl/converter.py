"""Converter: transform PyTorch models into Nengo networks.

The ``Converter`` class analyzes a PyTorch ``nn.Module`` and produces
an equivalent Nengo network, replacing activation functions with the
appropriate Nengo neuron types (ReLU → ``RectifiedLinear``, etc.).

This allows trained PyTorch models to be run as spiking neural networks
by replacing the activation functions with spiking equivalents.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nengo

from .tensor_node import TorchNode, _SpatialModule


class ConversionError(Exception):
    """Raised when a PyTorch layer cannot be converted to Nengo."""


class Converter:
    """Convert a trained PyTorch model to an equivalent Nengo network.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to convert.
    allow_fallback : bool
        If True (default), unsupported layers are wrapped as ``nengo.Node``
        objects that call through to the PyTorch layer. If False, conversion
        fails for unsupported layers.
    scale_firing_rates : float or None
        If set, scale the input to neuron layers so that the maximum
        firing rate (in Hz) equals approximately this value.
        For ReLU neurons in Nengo, the firing rate equals the input
        current, so a scale of 500 means peak rate ≈ 500 Hz.
    synapse : float or nengo.Synapse or None
        Synapse to apply on all connections (default None for no filtering).
    activation_type : str
        Neuron type to use for activations.
        Options: ``'rectified_linear'``, ``'spiking_relu'``, ``'lif'``,
        ``'softlif'``.  Use ``'spiking_relu'`` for a network that behaves
        like ReLU in rate mode and emits discrete spikes in spiking mode.
    input_shape : tuple, optional
        Shape of model inputs before flattening. Required when the first
        converted layer is spatial, e.g. ``(1, 28, 28)`` for MNIST Conv2d
        models. ``(28, 28, 1)`` is also accepted when it matches the layer's
        channel count.

    Examples
    --------
    ::

        import torch
        import torch.nn as nn
        import nengo_dl

        # A simple feedforward model
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        # Load pretrained weights ...

        converter = nengo_dl.Converter(model, scale_firing_rates=500)
        with nengo_dl.Simulator(converter.net) as sim:
            sim.run(0.1, data={converter.inputs[model[0]]: x_test})
        print(sim.data[converter.outputs[model[-1]]])
    """

    def __init__(
        self,
        model: nn.Module,
        allow_fallback: bool = True,
        scale_firing_rates: Optional[float] = None,
        synapse=None,
        activation_type: str = "rectified_linear",
        dt: float = 0.001,
        input_shape: Optional[Tuple[int, ...]] = None,
    ):
        self.model = model
        self.allow_fallback = allow_fallback
        self.scale_firing_rates = scale_firing_rates
        self.synapse = synapse
        self.activation_type = activation_type
        self.dt = dt
        self.input_shape = tuple(input_shape) if input_shape is not None else None

        # Maps layer → nengo object
        self._layer_map: Dict[nn.Module, nengo.Node] = {}
        # Input nodes for each layer that has external inputs
        self.inputs: Dict[nn.Module, nengo.Node] = {}
        # Output probes/nodes for each layer
        self.outputs: Dict[nn.Module, nengo.Node] = {}

        self.net = nengo.Network(label=f"Converted({type(model).__name__})")
        self._convert(model)

    def _get_neuron_type(self) -> nengo.neurons.NeuronType:
        """Return the Nengo neuron type to use for activations."""
        a = self.activation_type.lower().replace("-", "_")
        # amplitude=1/scale_firing_rates ensures probe values match the original
        # activations in rate mode while firing rates are scaled up to scale Hz.
        amp = (1.0 / self.scale_firing_rates
               if self.scale_firing_rates is not None else 1.0)
        if a in ("relu", "rectified_linear", "rectifiedlinear"):
            return nengo.RectifiedLinear(amplitude=amp)
        elif a in ("spiking_relu", "spikingrectifiedlinear", "spiky_relu"):
            return nengo.SpikingRectifiedLinear(amplitude=amp)
        elif a == "lif":
            return nengo.LIF()
        elif a in ("softlif", "soft_lif"):
            from .neurons import SoftLIFRate
            return SoftLIFRate()
        elif a in ("sigmoid",):
            return nengo.Sigmoid()
        elif a in ("tanh",):
            return nengo.Tanh()
        else:
            warnings.warn(
                f"Unknown activation_type '{self.activation_type}'; "
                "using RectifiedLinear."
            )
            return nengo.RectifiedLinear(amplitude=amp)

    def _convert(self, model: nn.Module):
        """Walk the model and build the Nengo network."""
        with self.net:
            # Flatten all layers in order
            layers = list(_iter_layers(model))
            if not layers:
                layers = [(model.__class__.__name__, model)]

            prev_node = None
            prev_size = None
            prev_shape = None

            for name, layer in layers:
                prev_node, prev_size, prev_shape = self._convert_layer(
                    name, layer, prev_node, prev_size, prev_shape
                )
                if prev_node is not None:
                    self.outputs[layer] = prev_node

    def _convert_layer(self, name, layer, prev_node, prev_size, prev_shape):
        """Convert a single layer and return (output_node, output_size)."""

        if isinstance(layer, nn.Linear):
            return self._convert_linear(name, layer, prev_node, prev_size, prev_shape)

        elif isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
            return self._convert_activation(name, layer, prev_node, prev_size, prev_shape)

        elif isinstance(layer, nn.Conv2d):
            return self._convert_conv2d(name, layer, prev_node, prev_size, prev_shape)

        elif isinstance(layer, (nn.Flatten, nn.Identity)):
            return self._convert_flatten(name, layer, prev_node, prev_size, prev_shape)

        elif isinstance(layer, nn.Sequential):
            # Recursively convert sub-layers
            for sub_name, sub_layer in layer.named_children():
                prev_node, prev_size, prev_shape = self._convert_layer(
                    f"{name}/{sub_name}", sub_layer, prev_node, prev_size, prev_shape
                )
            return prev_node, prev_size, prev_shape

        else:
            if self.allow_fallback:
                return self._convert_fallback(name, layer, prev_node, prev_size, prev_shape)
            else:
                raise ConversionError(
                    f"No converter for layer type {type(layer).__name__}. "
                    "Set allow_fallback=True to wrap unsupported layers."
                )

    def _convert_linear(self, name, layer: nn.Linear, prev_node, prev_size, prev_shape):
        """Convert nn.Linear to a linear Nengo Node plus trainable connections."""
        in_size = layer.in_features
        out_size = layer.out_features
        weights = layer.weight.data.cpu().numpy()  # (out, in)
        bias = layer.bias.data.cpu().numpy() if layer.bias is not None else None

        if prev_node is None:
            # Create an input node
            node = nengo.Node(np.zeros(in_size), label=f"{name}_input")
            self.inputs[layer] = node
            self._layer_map[layer] = node
            prev_node = node
            prev_size = in_size
            prev_shape = (in_size,)

        out = nengo.Node(size_in=out_size, label=name)

        # Only filter spike-train outputs (from neuron ensembles); plain nodes
        # carry continuous signals and don't need a synaptic low-pass filter.
        conn_synapse = (
            self.synapse if isinstance(prev_node, nengo.ensemble.Neurons) else None
        )
        nengo.Connection(
            prev_node,
            out,
            transform=weights,
            synapse=conn_synapse,
        )
        if bias is not None:
            bias_node = nengo.Node([1.0], label=f"{name}_bias")
            nengo.Connection(
                bias_node,
                out,
                transform=bias.reshape(out_size, 1),
                synapse=None,
            )

        self._layer_map[layer] = out
        return out, out_size, (out_size,)

    def _convert_activation(self, name, layer, prev_node, prev_size, prev_shape):
        """Convert activation layer.

        Linear layers place their activation directly on the output Ensemble, so
        a following ReLU can be skipped. Spatial TorchNode layers output linear
        activations, so they need an explicit neuron Ensemble here.
        """
        if isinstance(prev_node, nengo.ensemble.Neurons):
            return prev_node, prev_size, prev_shape

        if prev_node is None or prev_size is None:
            return prev_node, prev_size, prev_shape

        scale = float(self.scale_firing_rates) if self.scale_firing_rates else 1.0
        ens = nengo.Ensemble(
            prev_size,
            dimensions=1,
            neuron_type=self._get_neuron_type(),
            gain=np.ones(prev_size),
            bias=np.zeros(prev_size),
            label=name,
        )
        # Apply scale_firing_rates as the connection transform rather than as
        # Ensemble gain. Connecting to ens.neurons injects current directly,
        # bypassing the Ensemble's encoding path (gain/bias), so the gain
        # parameter is silently ignored. The transform achieves the same
        # J_i = scale × input_i, making spiking rate = scale × conv_output and
        # amplitude = 1/scale so that filtered output recovers the original value.
        nengo.Connection(prev_node, ens.neurons, transform=scale, synapse=None)
        self._layer_map[layer] = ens.neurons
        return ens.neurons, prev_size, prev_shape

    def _convert_conv2d(self, name, layer: nn.Conv2d, prev_node, prev_size, prev_shape):
        """Convert Conv2d to a TorchNode with flat Nengo I/O."""
        if prev_node is None:
            if self.input_shape is None:
                raise ConversionError(
                    "Converter(input_shape=...) is required when the first "
                    "converted layer is Conv2d."
                )
            prev_shape = self.input_shape
            prev_size = int(np.prod(prev_shape))
            prev_node = nengo.Node(np.zeros(prev_size), label=f"{name}_input")
            self.inputs[layer] = prev_node
            self._layer_map[layer] = prev_node

        if prev_shape is None or len(prev_shape) != 3:
            raise ConversionError(
                f"Conv2d layer '{name}' requires a spatial input shape, "
                f"got {prev_shape!r}."
            )

        module = _SpatialModule(layer, prev_shape)
        if prev_shape[0] == layer.in_channels:
            c, h, w = prev_shape
        else:
            h, w, c = prev_shape
        try:
            layer_device = next(layer.parameters()).device
        except StopIteration:
            layer_device = torch.device("cpu")
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w, device=layer_device)
            out = layer(dummy)
        _, out_c, out_h, out_w = out.shape
        shape_out = (out_c, out_h, out_w)
        size_out = int(np.prod(shape_out))

        node = TorchNode(
            module,
            size_in=prev_size,
            size_out=size_out,
            shape_in=prev_shape,
            shape_out=(size_out,),
            label=name,
        )
        conn_synapse = (
            self.synapse if isinstance(prev_node, nengo.ensemble.Neurons) else None
        )
        nengo.Connection(prev_node, node, synapse=conn_synapse)
        self._layer_map[layer] = node
        return node, size_out, shape_out

    def _convert_flatten(self, name, layer, prev_node, prev_size, prev_shape):
        """Flatten is a no-op at the Nengo level (signals are always flat)."""
        return prev_node, prev_size, (prev_size,) if prev_size is not None else None

    def _convert_fallback(self, name, layer, prev_node, prev_size, prev_shape):
        """Wrap an unsupported layer as a passthrough nengo.Node."""
        warnings.warn(
            f"Layer {type(layer).__name__} ('{name}') is not natively supported; "
            "wrapping as a passthrough Node. Gradient flow may be limited."
        )

        if prev_node is None or prev_size is None:
            warnings.warn(f"Cannot create fallback node for {name}; skipping.")
            return prev_node, prev_size, prev_shape

        # Probe the layer to get output size
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, prev_size)
                out = layer(dummy)
                out_size = int(out.numel())
        except Exception:
            warnings.warn(f"Cannot determine output size for {name}; skipping.")
            return prev_node, prev_size, prev_shape

        def layer_fn(t, x):
            with torch.no_grad():
                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                y = layer(x_t)
                return y.squeeze(0).numpy()

        node = nengo.Node(layer_fn, size_in=prev_size, size_out=out_size, label=name)
        if prev_node is not None:
            conn_synapse = (
                self.synapse if isinstance(prev_node, nengo.ensemble.Neurons) else None
            )
            nengo.Connection(prev_node, node, synapse=conn_synapse)
        self._layer_map[layer] = node
        return node, out_size, (out_size,)


def _iter_layers(model: nn.Module, prefix: str = ""):
    """Yield (name, layer) for each leaf module in the model."""
    children = list(model.named_children())
    if not children:
        # Leaf module
        yield prefix or model.__class__.__name__, model
    else:
        for name, child in children:
            full_name = f"{prefix}/{name}" if prefix else name
            yield from _iter_layers(child, full_name)
