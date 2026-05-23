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
        Options: ``'rectified_linear'``, ``'lif'``, ``'softlif'``.

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
    ):
        self.model = model
        self.allow_fallback = allow_fallback
        self.scale_firing_rates = scale_firing_rates
        self.synapse = synapse
        self.activation_type = activation_type
        self.dt = dt

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
        a = self.activation_type.lower()
        if a in ("relu", "rectified_linear", "rectifiedlinear"):
            return nengo.RectifiedLinear()
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
            return nengo.RectifiedLinear()

    def _convert(self, model: nn.Module):
        """Walk the model and build the Nengo network."""
        with self.net:
            # Flatten all layers in order
            layers = list(_iter_layers(model))
            if not layers:
                layers = [(model.__class__.__name__, model)]

            prev_node = None
            prev_size = None

            for name, layer in layers:
                prev_node, prev_size = self._convert_layer(
                    name, layer, prev_node, prev_size
                )
                if prev_node is not None:
                    self.outputs[layer] = prev_node

    def _convert_layer(self, name, layer, prev_node, prev_size):
        """Convert a single layer and return (output_node, output_size)."""

        if isinstance(layer, nn.Linear):
            return self._convert_linear(name, layer, prev_node, prev_size)

        elif isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
            return self._convert_activation(name, layer, prev_node, prev_size)

        elif isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            return self._convert_conv(name, layer, prev_node, prev_size)

        elif isinstance(layer, (nn.Flatten, nn.Identity)):
            return self._convert_flatten(name, layer, prev_node, prev_size)

        elif isinstance(layer, nn.Sequential):
            # Recursively convert sub-layers
            for sub_name, sub_layer in layer.named_children():
                prev_node, prev_size = self._convert_layer(
                    f"{name}/{sub_name}", sub_layer, prev_node, prev_size
                )
            return prev_node, prev_size

        else:
            if self.allow_fallback:
                return self._convert_fallback(name, layer, prev_node, prev_size)
            else:
                raise ConversionError(
                    f"No converter for layer type {type(layer).__name__}. "
                    "Set allow_fallback=True to wrap unsupported layers."
                )

    def _convert_linear(self, name, layer: nn.Linear, prev_node, prev_size):
        """Convert nn.Linear to a Nengo Ensemble + connection."""
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

        # Scale weights if firing rate scaling is requested
        if self.scale_firing_rates is not None:
            scale = 1.0 / self.scale_firing_rates
            weights = weights * scale
            if bias is not None:
                bias = bias * scale

        # Output ensemble
        neuron_type = self._get_neuron_type()
        ens = nengo.Ensemble(
            out_size,
            dimensions=1,
            neuron_type=neuron_type,
            gain=np.ones(out_size),
            bias=bias if bias is not None else np.zeros(out_size),
            label=name,
        )

        # Connection with the weight matrix as transform
        nengo.Connection(
            prev_node,
            ens.neurons,
            transform=weights,
            synapse=self.synapse,
        )

        self._layer_map[layer] = ens.neurons
        return ens.neurons, out_size

    def _convert_activation(self, name, layer, prev_node, prev_size):
        """Convert activation layer (already handled by neuron type in ensemble)."""
        # Activation is incorporated into the neuron type; skip as separate layer
        return prev_node, prev_size

    def _convert_conv(self, name, layer, prev_node, prev_size):
        """Convert convolutional layer using a TorchNode fallback."""
        return self._convert_fallback(name, layer, prev_node, prev_size)

    def _convert_flatten(self, name, layer, prev_node, prev_size):
        """Flatten is a no-op at the Nengo level (signals are always flat)."""
        return prev_node, prev_size

    def _convert_fallback(self, name, layer, prev_node, prev_size):
        """Wrap an unsupported layer as a passthrough nengo.Node."""
        warnings.warn(
            f"Layer {type(layer).__name__} ('{name}') is not natively supported; "
            "wrapping as a passthrough Node. Gradient flow may be limited."
        )

        if prev_node is None or prev_size is None:
            warnings.warn(f"Cannot create fallback node for {name}; skipping.")
            return prev_node, prev_size

        # Probe the layer to get output size
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, prev_size)
                out = layer(dummy)
                out_size = int(out.numel())
        except Exception:
            warnings.warn(f"Cannot determine output size for {name}; skipping.")
            return prev_node, prev_size

        def layer_fn(t, x):
            with torch.no_grad():
                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                y = layer(x_t)
                return y.squeeze(0).numpy()

        node = nengo.Node(layer_fn, size_in=prev_size, size_out=out_size, label=name)
        if prev_node is not None:
            nengo.Connection(prev_node, node, synapse=self.synapse)
        self._layer_map[layer] = node
        return node, out_size


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
