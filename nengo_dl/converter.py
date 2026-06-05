"""Converter: transform PyTorch models into Nengo networks.

This implementation uses ``torch.fx`` to trace a model graph and convert it to
an equivalent Nengo network. That allows us to preserve graph structure such as
branches and residual connections, rather than flattening everything into a
single sequential chain.
"""

import copy
import operator
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule
from torch.fx.passes.shape_prop import ShapeProp

import nengo

from .tensor_node import TorchNode, _SpatialModule


class ConversionError(Exception):
    """Raised when a PyTorch layer cannot be converted to Nengo."""


@dataclass
class GraphValue:
    """Converted representation of an FX node output."""

    obj: Any
    size: int
    shape: Tuple[int, ...]


class Converter:
    """Convert a trained PyTorch model to an equivalent Nengo network."""

    def __init__(
        self,
        model: nn.Module,
        allow_fallback: bool = True,
        inference_only: bool = False,
        max_to_avg_pool: bool = False,
        scale_firing_rates: Optional[float] = None,
        synapse=None,
        activation_type: str = "rectified_linear",
        dt: float = 0.001,
        input_shape: Optional[Tuple[int, ...]] = None,
    ):
        self.model = model.eval()
        self.allow_fallback = allow_fallback
        self.inference_only = inference_only
        self.max_to_avg_pool = max_to_avg_pool
        self.scale_firing_rates = scale_firing_rates
        self.synapse = synapse
        self.activation_type = activation_type
        self.dt = dt
        self.input_shape = self._normalize_input_shape(input_shape)

        self._layer_map: Dict[Any, Any] = {}
        self.inputs: "OrderedDict[Any, nengo.Node]" = OrderedDict()
        self.outputs: "OrderedDict[Any, Any]" = OrderedDict()
        self._fx_values: Dict[Any, GraphValue] = {}
        self._fx_constants: Dict[Any, Any] = {}
        self._shape_model: Optional[nn.Module] = None
        self.output_structure: Any = None
        self._module_call_counts: Dict[str, int] = {}
        self._shared_module_targets: set[str] = set()
        self._shared_spatial_wrappers: Dict[Tuple[int, Tuple[int, ...]], nn.Module] = {}

        self.net = nengo.Network(label=f"Converted({type(model).__name__})")
        self._convert()

    def _normalize_input_shape(self, input_shape):
        if input_shape is None:
            return None
        if isinstance(input_shape, (list, tuple)) and input_shape and isinstance(
            input_shape[0], (list, tuple)
        ):
            return tuple(tuple(s) for s in input_shape)
        return tuple(input_shape)

    def _get_neuron_type(self) -> nengo.neurons.NeuronType:
        """Return the Nengo neuron type to use for activations."""
        a = self.activation_type.lower().replace("-", "_")
        amp = (1.0 / self.scale_firing_rates if self.scale_firing_rates is not None else 1.0)
        if a in ("relu", "rectified_linear", "rectifiedlinear"):
            return nengo.RectifiedLinear(amplitude=amp)
        if a in ("spiking_relu", "spikingrectifiedlinear", "spiky_relu"):
            return nengo.SpikingRectifiedLinear(amplitude=amp)
        if a == "lif":
            return nengo.LIF()
        if a in ("softlif", "soft_lif"):
            from .neurons import SoftLIFRate

            return SoftLIFRate()
        if a == "sigmoid":
            return nengo.Sigmoid()
        if a == "tanh":
            return nengo.Tanh()

        warnings.warn(
            f"Unknown activation_type '{self.activation_type}'; using RectifiedLinear."
        )
        return nengo.RectifiedLinear(amplitude=amp)

    def _convert(self):
        gm = self._trace_model()
        modules = dict(self.model.named_modules())
        self._module_call_counts = self._count_module_calls(gm)
        self._shared_module_targets = {
            target for target, count in self._module_call_counts.items() if count > 1
        }

        with self.net:
            for node in gm.graph.nodes:
                if node.op == "placeholder":
                    value = self._convert_placeholder(node)
                    self._fx_values[node] = value
                elif node.op == "get_attr":
                    self._fx_constants[node] = self._resolve_attr(node.target)
                elif node.op == "call_module":
                    module = modules[node.target]
                    value = self._convert_call_module(node, module)
                    self._fx_values[node] = value
                    self._layer_map[module] = value.obj
                    self.outputs[module] = value.obj
                    if node.target in self._shared_module_targets:
                        self.outputs[node.name] = value.obj
                elif node.op == "call_function":
                    value = self._convert_call_function(node)
                    self._fx_values[node] = value
                elif node.op == "call_method":
                    value = self._convert_call_method(node)
                    self._fx_values[node] = value
                elif node.op == "output":
                    self._convert_output(node)
                else:
                    raise ConversionError(f"Unsupported FX node type '{node.op}'")

    def _resolve_attr(self, target):
        attr = self.model
        for name in target.split("."):
            attr = getattr(attr, name)
        return attr

    def _count_module_calls(self, gm):
        counts: Dict[str, int] = {}
        for node in gm.graph.nodes:
            if node.op == "call_module":
                counts[node.target] = counts.get(node.target, 0) + 1
        return counts

    def _resolve_shape_attr(self, target):
        attr = self._shape_model if self._shape_model is not None else self.model
        for name in target.split("."):
            attr = getattr(attr, name)
        return attr

    def _trace_model(self) -> GraphModule:
        try:
            self._shape_model = copy.deepcopy(self.model).cpu().eval()
            gm = torch.fx.symbolic_trace(self._shape_model)
        except Exception as exc:
            raise ConversionError(
                f"Unable to trace model with torch.fx: {exc}"
            ) from exc

        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        dummy_inputs = self._make_dummy_inputs(gm, placeholder_nodes)

        try:
            ShapeProp(gm).propagate(*dummy_inputs)
        except Exception as exc:
            raise ConversionError(
                f"Unable to infer intermediate tensor shapes with torch.fx: {exc}"
            ) from exc

        return gm

    def _model_tensor_spec(self):
        model = self._shape_model if self._shape_model is not None else self.model
        for tensor in list(model.parameters()) + list(model.buffers()):
            return tensor.device, tensor.dtype
        return torch.device("cpu"), torch.float32

    def _make_dummy_inputs(self, gm, placeholder_nodes):
        if len(placeholder_nodes) == 0:
            raise ConversionError("Model has no inputs.")

        if self.input_shape is not None:
            if isinstance(self.input_shape[0], tuple):
                input_shapes = list(self.input_shape)
            else:
                input_shapes = [self.input_shape]
        else:
            inferred = self._infer_input_shape_from_modules(gm)
            input_shapes = [inferred]

        if len(input_shapes) != len(placeholder_nodes):
            raise ConversionError(
                f"Model has {len(placeholder_nodes)} inputs, but input_shape specifies "
                f"{len(input_shapes)} shapes."
            )

        device, dtype = self._model_tensor_spec()
        return [
            torch.zeros((1,) + tuple(shape), dtype=dtype, device=device)
            for shape in input_shapes
        ]

    def _infer_input_shape_from_modules(self, gm):
        modules = dict(gm.named_modules())
        for node in gm.graph.nodes:
            if node.op != "call_module":
                if node.op == "call_function" and self._is_linear_function(node.target):
                    weight = self._resolve_shape_attr(node.args[1].target)
                    return (int(weight.shape[1]),)
                continue
            module = modules[node.target]
            if isinstance(module, nn.Linear):
                return (module.in_features,)
            if isinstance(module, nn.Conv2d):
                raise ConversionError(
                    "Converter(input_shape=...) is required when the first converted "
                    "layer is Conv2d."
                )

        raise ConversionError(
            "Unable to infer input shape automatically; provide input_shape=..."
        )

    def _tensor_shape(self, node) -> Tuple[int, ...]:
        meta = node.meta.get("tensor_meta")
        if meta is None:
            raise ConversionError(f"No tensor metadata available for FX node '{node.name}'.")
        shape = tuple(meta.shape)
        return tuple(shape[1:])

    def _size_from_shape(self, shape: Tuple[int, ...]) -> int:
        return int(np.prod(shape)) if shape else 1

    def _get_value(self, arg) -> GraphValue:
        if not isinstance(arg, torch.fx.Node):
            raise ConversionError(f"Expected FX node input, got {type(arg).__name__}")
        return self._fx_values[arg]

    def _get_constant(self, arg):
        if not isinstance(arg, torch.fx.Node):
            return arg
        if arg not in self._fx_constants:
            raise ConversionError(f"Expected constant FX node, got '{arg.op}'.")
        return self._fx_constants[arg]

    def _ensure_single_input(self, node) -> GraphValue:
        if len(node.args) != 1:
            raise ConversionError(
                f"Node '{node.name}' expected 1 input, got {len(node.args)}."
            )
        return self._get_value(node.args[0])

    def _convert_placeholder(self, node) -> GraphValue:
        shape = self._tensor_shape(node)
        size = self._size_from_shape(shape)
        input_node = nengo.Node(np.zeros(size, dtype=np.float32), label=str(node.target))
        self.inputs[str(node.target)] = input_node
        return GraphValue(input_node, size, shape)

    def _connect(self, src, dst, transform=1.0, synapse=None):
        return nengo.Connection(src, dst, transform=transform, synapse=synapse)

    def _convert_call_module(self, node, layer) -> GraphValue:
        inp = self._ensure_single_input(node)
        name = str(node.target)

        if layer in self.inputs.values():
            return inp

        if node.target in self._shared_module_targets:
            return self._convert_shared_module_call(node, layer, inp)

        if isinstance(layer, nn.Linear):
            return self._convert_linear(name, layer, inp)
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            return self._convert_batchnorm(name, layer, inp, node)
        if isinstance(layer, (nn.AvgPool1d, nn.AvgPool2d, nn.MaxPool1d, nn.MaxPool2d)):
            return self._convert_pooling(name, layer, inp, node)
        if isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
            return self._convert_activation(name, layer, inp)
        if isinstance(layer, nn.Conv2d):
            return self._convert_conv2d(name, layer, inp)
        if isinstance(layer, (nn.Flatten, nn.Identity)):
            return self._convert_flatten(name, layer, inp)

        return self._convert_fallback_module(name, layer, inp, node)

    def _convert_shared_module_call(self, node, layer, inp: GraphValue) -> GraphValue:
        out_shape = self._tensor_shape(node)
        size_out = self._size_from_shape(out_shape)
        wrapped = self._shared_wrapper(layer, inp.shape)
        out = TorchNode(
            wrapped,
            size_in=inp.size,
            size_out=size_out,
            shape_in=inp.shape,
            shape_out=out_shape,
            label=str(node.name),
        )
        conn_synapse = self.synapse if isinstance(inp.obj, nengo.ensemble.Neurons) else None
        self._connect(inp.obj, out, synapse=conn_synapse)
        return GraphValue(out, size_out, out_shape)

    def _convert_linear(self, name, layer: nn.Linear, inp: GraphValue) -> GraphValue:
        out_size = layer.out_features
        weights = layer.weight.data.detach().cpu().numpy()
        bias = layer.bias.data.detach().cpu().numpy() if layer.bias is not None else None

        out = nengo.Node(size_in=out_size, label=name)
        conn_synapse = self.synapse if isinstance(inp.obj, nengo.ensemble.Neurons) else None
        self._connect(inp.obj, out, transform=weights, synapse=conn_synapse)

        if bias is not None:
            bias_node = nengo.Node([1.0], label=f"{name}_bias")
            self._connect(bias_node, out, transform=bias.reshape(out_size, 1), synapse=None)

        if isinstance(node_arg := inp.obj, nengo.Node) and node_arg in self.inputs.values():
            module_key = layer
            if module_key not in self.inputs:
                self.inputs[module_key] = node_arg

        return GraphValue(out, out_size, (out_size,))

    def _convert_activation(self, name, layer, inp: GraphValue) -> GraphValue:
        if isinstance(layer, nn.LeakyReLU):
            return self._convert_fallback_module(name, layer, inp, None)

        scale = float(self.scale_firing_rates) if self.scale_firing_rates else 1.0
        ens = nengo.Ensemble(
            inp.size,
            dimensions=1,
            neuron_type=self._get_neuron_type(),
            gain=np.ones(inp.size),
            bias=np.zeros(inp.size),
            label=name,
        )
        self._connect(inp.obj, ens.neurons, transform=scale, synapse=None)
        return GraphValue(ens.neurons, inp.size, inp.shape)

    def _convert_conv2d(self, name, layer: nn.Conv2d, inp: GraphValue) -> GraphValue:
        if len(inp.shape) != 3:
            raise ConversionError(
                f"Conv2d layer '{name}' requires a spatial input shape, got {inp.shape!r}."
            )

        module = _SpatialModule(layer, inp.shape)
        out_shape = self._tensor_shape_from_module(layer, inp.shape)
        size_out = self._size_from_shape(out_shape)
        node = TorchNode(
            module,
            size_in=inp.size,
            size_out=size_out,
            shape_in=inp.shape,
            shape_out=out_shape,
            label=name,
        )
        conn_synapse = self.synapse if isinstance(inp.obj, nengo.ensemble.Neurons) else None
        self._connect(inp.obj, node, synapse=conn_synapse)
        return GraphValue(node, size_out, out_shape)

    def _convert_batchnorm(self, name, layer, inp: GraphValue, node) -> GraphValue:
        if not self.inference_only:
            return self._convert_fallback_module(name, layer, inp, node)

        if layer.track_running_stats is False:
            return self._convert_fallback_module(name, layer, inp, node)

        running_mean = layer.running_mean.detach().cpu().numpy()
        running_var = layer.running_var.detach().cpu().numpy()
        gamma = (
            layer.weight.detach().cpu().numpy()
            if layer.affine and layer.weight is not None
            else np.ones_like(running_mean)
        )
        beta = (
            layer.bias.detach().cpu().numpy()
            if layer.affine and layer.bias is not None
            else np.zeros_like(running_mean)
        )

        stddev = np.sqrt(running_var + layer.eps)
        channel_scale = gamma / stddev
        channel_bias = beta - gamma * running_mean / stddev

        scale_vec, bias_vec = self._broadcast_channel_params(
            inp.shape, channel_scale, channel_bias
        )

        output = nengo.Node(size_in=inp.size, label=name)
        syn = self.synapse if isinstance(inp.obj, nengo.ensemble.Neurons) else None
        self._connect(
            inp.obj,
            output,
            transform=np.diag(scale_vec.astype(np.float32)),
            synapse=syn,
        )
        bias_node = nengo.Node(bias_vec.astype(np.float32), label=f"{name}_bias")
        self._connect(bias_node, output, synapse=None)
        return GraphValue(output, inp.size, inp.shape)

    def _broadcast_channel_params(self, shape, channel_scale, channel_bias):
        if len(shape) == 1:
            if len(channel_scale) != shape[0]:
                raise ConversionError(
                    f"BatchNorm channels ({len(channel_scale)}) do not match shape {shape}."
                )
            return channel_scale, channel_bias

        n_channels = shape[0]
        if len(channel_scale) != n_channels:
            raise ConversionError(
                f"BatchNorm channels ({len(channel_scale)}) do not match shape {shape}."
            )

        spatial = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        scale = np.repeat(channel_scale, spatial)
        bias = np.repeat(channel_bias, spatial)
        return scale, bias

    def _convert_pooling(self, name, layer, inp: GraphValue, node) -> GraphValue:
        out_shape = self._tensor_shape(node)
        if isinstance(layer, (nn.MaxPool1d, nn.MaxPool2d)):
            if not self.max_to_avg_pool:
                return self._convert_fallback_module(name, layer, inp, node)
            warnings.warn(
                f"Layer '{name}' ({type(layer).__name__}) is converted as average pooling "
                "because max_to_avg_pool=True; behavior will differ from PyTorch max pooling."
            )

        transform = self._pool_transform(layer, inp.shape, out_shape)
        output = nengo.Node(size_in=int(np.prod(out_shape)), label=name)
        syn = self.synapse if isinstance(inp.obj, nengo.ensemble.Neurons) else None
        self._connect(inp.obj, output, transform=transform, synapse=syn)
        return GraphValue(output, int(np.prod(out_shape)), out_shape)

    def _pool_transform(self, layer, in_shape, out_shape):
        if len(in_shape) == 2:
            return self._pool_transform_1d(layer, in_shape, out_shape)
        if len(in_shape) == 3:
            return self._pool_transform_2d(layer, in_shape, out_shape)
        raise ConversionError(
            f"Pooling layer {type(layer).__name__} requires 1D or 2D spatial input, got {in_shape}."
        )

    def _pool_transform_1d(self, layer, in_shape, out_shape):
        channels, width = in_shape
        out_channels, out_width = out_shape
        if channels != out_channels:
            raise ConversionError("Pooling should preserve channel count.")

        kernel = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
        stride = layer.stride if layer.stride is not None else kernel
        if isinstance(stride, tuple):
            stride = stride[0]
        padding = layer.padding if isinstance(layer.padding, int) else layer.padding[0]
        count_include_pad = getattr(layer, "count_include_pad", True)

        transform = np.zeros((channels * out_width, channels * width), dtype=np.float32)
        for c in range(channels):
            for ow in range(out_width):
                start = ow * stride - padding
                valid = []
                for kw in range(kernel):
                    iw = start + kw
                    if 0 <= iw < width:
                        valid.append(iw)
                denom = kernel if count_include_pad else max(len(valid), 1)
                out_idx = c * out_width + ow
                for iw in valid:
                    in_idx = c * width + iw
                    transform[out_idx, in_idx] = 1.0 / denom

        return transform

    def _pool_transform_2d(self, layer, in_shape, out_shape):
        channels, height, width = in_shape
        out_channels, out_height, out_width = out_shape
        if channels != out_channels:
            raise ConversionError("Pooling should preserve channel count.")

        kernel_h, kernel_w = self._pair(layer.kernel_size)
        stride_h, stride_w = self._pair(layer.stride if layer.stride is not None else layer.kernel_size)
        pad_h, pad_w = self._pair(layer.padding)
        count_include_pad = getattr(layer, "count_include_pad", True)

        transform = np.zeros(
            (channels * out_height * out_width, channels * height * width),
            dtype=np.float32,
        )
        for c in range(channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    start_h = oh * stride_h - pad_h
                    start_w = ow * stride_w - pad_w
                    valid = []
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            ih = start_h + kh
                            iw = start_w + kw
                            if 0 <= ih < height and 0 <= iw < width:
                                valid.append((ih, iw))
                    denom = kernel_h * kernel_w if count_include_pad else max(len(valid), 1)
                    out_idx = c * (out_height * out_width) + oh * out_width + ow
                    for ih, iw in valid:
                        in_idx = c * (height * width) + ih * width + iw
                        transform[out_idx, in_idx] = 1.0 / denom

        return transform

    def _pair(self, x):
        if isinstance(x, tuple):
            return x
        return (x, x)

    def _tensor_shape_from_module(self, module, in_shape):
        try:
            layer_device = next(module.parameters()).device
        except StopIteration:
            layer_device = torch.device("cpu")

        if len(in_shape) == 3 and in_shape[0] == module.in_channels:
            c, h, w = in_shape
        else:
            h, w, c = in_shape

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w, device=layer_device)
            out = module(dummy)
        return tuple(out.shape[1:])

    def _convert_flatten(self, name, layer, inp: GraphValue) -> GraphValue:
        return GraphValue(inp.obj, inp.size, inp.shape)

    def _convert_call_function(self, node) -> GraphValue:
        target = node.target

        if self._is_linear_function(target):
            return self._convert_function_linear(node)
        if target in (operator.add, torch.add):
            return self._convert_add(node)
        if target in (operator.sub, torch.sub):
            return self._convert_sub(node)
        if target in (operator.mul, torch.mul):
            return self._convert_mul(node)
        if target in (torch.relu, F.relu):
            return self._convert_function_activation(node, nn.ReLU())
        if target in (torch.sigmoid,):
            return self._convert_function_activation(node, nn.Sigmoid())
        if target in (torch.tanh,):
            return self._convert_function_activation(node, nn.Tanh())
        if target in (torch.reshape,):
            return self._convert_shape_only(node)
        if target in (torch.flatten,):
            return self._convert_shape_only(node)
        if target in (torch.permute,):
            return self._convert_permute(node)
        if target in (torch.transpose,):
            return self._convert_transpose(node)
        if target in (torch.cat,):
            return self._convert_concat(node)

        return self._convert_fallback_function(node)

    def _is_linear_function(self, target) -> bool:
        name = getattr(target, "__name__", "")
        return target is F.linear or name == "linear"

    def _convert_call_method(self, node) -> GraphValue:
        method = node.target
        if method in ("reshape", "view", "flatten"):
            return self._convert_shape_only(node)
        if method == "permute":
            return self._convert_permute(node)
        if method in ("transpose",):
            return self._convert_transpose(node)
        if method == "relu":
            return self._convert_function_activation(node, nn.ReLU())
        if method == "sigmoid":
            return self._convert_function_activation(node, nn.Sigmoid())
        if method == "tanh":
            return self._convert_function_activation(node, nn.Tanh())
        if method in ("sub", "__sub__", "__rsub__"):
            return self._convert_sub(node)
        if method in ("mul", "__mul__", "__rmul__"):
            return self._convert_mul(node)

        return self._convert_fallback_function(node)

    def _convert_function_activation(self, node, layer) -> GraphValue:
        inp = self._get_value(node.args[0])
        return self._convert_activation(node.name, layer, inp)

    def _convert_shape_only(self, node) -> GraphValue:
        inp = self._get_value(node.args[0])
        shape = self._tensor_shape(node)
        size = self._size_from_shape(shape)
        if size != inp.size:
            raise ConversionError(
                f"Shape operation '{node.name}' changes tensor size from {inp.size} to {size}."
            )
        return GraphValue(inp.obj, size, shape)

    def _convert_permute(self, node) -> GraphValue:
        inp = self._get_value(node.args[0])
        out_shape = self._tensor_shape(node)
        dims_args = node.args[1:]
        if not dims_args:
            dims_arg = node.kwargs.get("dims")
            if dims_arg is None:
                raise ConversionError(f"Permute node '{node.name}' is missing dims.")
            dims_args = (dims_arg,)

        if len(dims_args) == 1 and isinstance(dims_args[0], (list, tuple)):
            dims = tuple(int(d) for d in dims_args[0])
        elif len(dims_args) == 1 and not np.isscalar(dims_args[0]):
            dims = tuple(int(d) for d in dims_args[0])
        else:
            dims = tuple(int(d) for d in dims_args)

        if len(dims) == len(inp.shape) + 1:
            if dims[0] != 0:
                raise ConversionError(
                    f"Permute node '{node.name}' cannot permute batch dimension in converted models."
                )
            dims = tuple(d - 1 for d in dims[1:])

        if dims is None:
            raise ConversionError(f"Permute node '{node.name}' is missing dims.")

        dims = self._normalize_dims(dims, len(inp.shape), node.name)
        if tuple(inp.shape[d] for d in dims) != tuple(out_shape):
            raise ConversionError(
                f"Permute node '{node.name}' dims {dims!r} do not match output shape {out_shape!r}."
            )

        return self._convert_reindex(node.name, inp, out_shape, dims)

    def _convert_transpose(self, node) -> GraphValue:
        inp = self._get_value(node.args[0])
        out_shape = self._tensor_shape(node)
        if len(node.args) >= 3:
            dim0 = int(node.args[1])
            dim1 = int(node.args[2])
        else:
            dim0 = int(node.kwargs["dim0"])
            dim1 = int(node.kwargs["dim1"])

        ndims = len(inp.shape)
        if dim0 > 0 and dim1 > 0:
            dim0 -= 1
            dim1 -= 1
        elif dim0 == 0 or dim1 == 0:
            raise ConversionError(
                f"Transpose node '{node.name}' cannot move batch dimension in converted models."
            )
        dim0 = dim0 % ndims
        dim1 = dim1 % ndims
        dims = list(range(ndims))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]

        if tuple(inp.shape[d] for d in dims) != tuple(out_shape):
            raise ConversionError(
                f"Transpose node '{node.name}' dims {(dim0, dim1)!r} do not match output shape {out_shape!r}."
            )

        return self._convert_reindex(node.name, inp, out_shape, tuple(dims))

    def _convert_reindex(self, name, inp: GraphValue, out_shape, dims) -> GraphValue:
        out_size = self._size_from_shape(out_shape)
        if out_size != inp.size:
            raise ConversionError(
                f"Reindex operation '{name}' changes tensor size from {inp.size} to {out_size}."
            )

        index_grid = np.arange(inp.size, dtype=np.int64).reshape(inp.shape)
        reordered = np.transpose(index_grid, axes=dims).reshape(-1)
        transform = np.zeros((out_size, inp.size), dtype=np.float32)
        transform[np.arange(out_size), reordered] = 1.0

        out = nengo.Node(size_in=out_size, label=name)
        syn = self.synapse if isinstance(inp.obj, nengo.ensemble.Neurons) else None
        self._connect(inp.obj, out, transform=transform, synapse=syn)
        return GraphValue(out, out_size, out_shape)

    def _convert_add(self, node) -> GraphValue:
        a_arg, b_arg = node.args[0], node.args[1]

        if isinstance(a_arg, torch.fx.Node) and isinstance(b_arg, torch.fx.Node):
            a = self._get_value(a_arg)
            b = self._get_value(b_arg)
            if a.shape != b.shape:
                raise ConversionError(
                    f"Add node '{node.name}' requires matching input shapes, got "
                    f"{a.shape!r} and {b.shape!r}."
                )

            out = nengo.Node(size_in=a.size, label=node.name)
            syn_a = self.synapse if isinstance(a.obj, nengo.ensemble.Neurons) else None
            syn_b = self.synapse if isinstance(b.obj, nengo.ensemble.Neurons) else None
            self._connect(a.obj, out, synapse=syn_a)
            self._connect(b.obj, out, synapse=syn_b)
            return GraphValue(out, a.size, a.shape)

        value, const, _ = self._split_tensor_constant_args(node, "Add")
        if value is None:
            raise ConversionError(
                f"Add node '{node.name}' requires at least one tensor input."
            )

        const_arr = self._constant_vector(node.name, const, value)

        out = nengo.Node(size_in=value.size, label=node.name)
        syn = self.synapse if isinstance(value.obj, nengo.ensemble.Neurons) else None
        self._connect(value.obj, out, synapse=syn)
        bias_node = nengo.Node(np.array([1.0], dtype=np.float32), label=f"{node.name}_bias")
        self._connect(bias_node, out, transform=const_arr.reshape(value.size, 1), synapse=None)
        return GraphValue(out, value.size, value.shape)

    def _convert_sub(self, node) -> GraphValue:
        a_arg, b_arg = node.args[0], node.args[1]

        if isinstance(a_arg, torch.fx.Node) and isinstance(b_arg, torch.fx.Node):
            a = self._get_value(a_arg)
            b = self._get_value(b_arg)
            if a.shape != b.shape:
                raise ConversionError(
                    f"Sub node '{node.name}' requires matching input shapes, got "
                    f"{a.shape!r} and {b.shape!r}."
                )

            out = nengo.Node(size_in=a.size, label=node.name)
            syn_a = self.synapse if isinstance(a.obj, nengo.ensemble.Neurons) else None
            syn_b = self.synapse if isinstance(b.obj, nengo.ensemble.Neurons) else None
            self._connect(a.obj, out, synapse=syn_a)
            self._connect(b.obj, out, transform=-1.0, synapse=syn_b)
            return GraphValue(out, a.size, a.shape)

        value, const, tensor_is_first = self._split_tensor_constant_args(node, "Sub")
        if value is None:
            raise ConversionError(
                f"Sub node '{node.name}' requires at least one tensor input."
            )

        const_arr = self._constant_vector(node.name, const, value)
        out = nengo.Node(size_in=value.size, label=node.name)
        syn = self.synapse if isinstance(value.obj, nengo.ensemble.Neurons) else None
        if tensor_is_first:
            self._connect(value.obj, out, synapse=syn)
            bias = -const_arr
        else:
            self._connect(value.obj, out, transform=-1.0, synapse=syn)
            bias = const_arr

        bias_node = nengo.Node(np.array([1.0], dtype=np.float32), label=f"{node.name}_bias")
        self._connect(bias_node, out, transform=bias.reshape(value.size, 1), synapse=None)
        return GraphValue(out, value.size, value.shape)

    def _convert_mul(self, node) -> GraphValue:
        a_arg, b_arg = node.args[0], node.args[1]
        if isinstance(a_arg, torch.fx.Node) and isinstance(b_arg, torch.fx.Node):
            return self._convert_fallback_function(node)

        value, const, tensor_is_first = self._split_tensor_constant_args(node, "Mul")
        if value is None:
            raise ConversionError(
                f"Mul node '{node.name}' requires at least one tensor input."
            )

        const_arr = np.asarray(const, dtype=np.float32)
        if const_arr.ndim != 0:
            return self._convert_fallback_function(node)

        scale = float(const_arr)
        out = nengo.Node(size_in=value.size, label=node.name)
        syn = self.synapse if isinstance(value.obj, nengo.ensemble.Neurons) else None
        self._connect(value.obj, out, transform=scale, synapse=syn)
        return GraphValue(out, value.size, value.shape)

    def _convert_function_linear(self, node) -> GraphValue:
        inp = self._get_value(node.args[0])
        weight = self._get_constant(node.args[1]).detach().cpu().numpy()
        bias = None
        if len(node.args) > 2 and node.args[2] is not None:
            bias = self._get_constant(node.args[2]).detach().cpu().numpy()

        out_size = weight.shape[0]
        out = nengo.Node(size_in=out_size, label=node.name)
        conn_synapse = self.synapse if isinstance(inp.obj, nengo.ensemble.Neurons) else None
        self._connect(inp.obj, out, transform=weight, synapse=conn_synapse)

        if bias is not None:
            bias_node = nengo.Node([1.0], label=f"{node.name}_bias")
            self._connect(bias_node, out, transform=bias.reshape(out_size, 1), synapse=None)

        return GraphValue(out, out_size, (out_size,))

    def _convert_concat(self, node) -> GraphValue:
        tensors = node.args[0]
        if not isinstance(tensors, (list, tuple)):
            raise ConversionError(f"cat node '{node.name}' expected a tensor list.")
        dim = node.kwargs.get("dim", 0)
        out_shape = self._tensor_shape(node)

        if dim not in (-1, len(out_shape) - 1, 1):
            return self._convert_fallback_function(node)

        values = [self._get_value(t) for t in tensors]
        out = nengo.Node(size_in=sum(v.size for v in values), label=node.name)
        start = 0
        for val in values:
            transform = np.zeros((out.size_in, val.size), dtype=np.float32)
            transform[start : start + val.size, :] = np.eye(val.size, dtype=np.float32)
            syn = self.synapse if isinstance(val.obj, nengo.ensemble.Neurons) else None
            self._connect(val.obj, out, transform=transform, synapse=syn)
            start += val.size

        return GraphValue(out, sum(v.size for v in values), out_shape)

    def _convert_fallback_module(self, name, layer, inp: GraphValue, node) -> GraphValue:
        if not self.allow_fallback:
            raise ConversionError(
                f"No converter for layer type {type(layer).__name__}. "
                "Set allow_fallback=True to wrap unsupported layers."
            )

        self._warn_torchnode_fallback(
            kind="layer",
            name=name,
            detail=type(layer).__name__,
        )

        out_shape = self._tensor_shape(node) if node is not None else self._infer_module_output_shape(layer, inp)
        size_out = self._size_from_shape(out_shape)
        wrapped = self._maybe_wrap_spatial(layer, inp.shape)
        out = TorchNode(
            wrapped,
            size_in=inp.size,
            size_out=size_out,
            shape_in=inp.shape,
            shape_out=out_shape,
            label=name,
        )
        conn_synapse = self.synapse if isinstance(inp.obj, nengo.ensemble.Neurons) else None
        self._connect(inp.obj, out, synapse=conn_synapse)
        return GraphValue(out, size_out, out_shape)

    def _infer_module_output_shape(self, layer, inp: GraphValue):
        wrapped = self._maybe_wrap_spatial(layer, inp.shape)
        with torch.no_grad():
            dummy = torch.zeros((1,) + tuple(inp.shape), dtype=torch.float32)
            if isinstance(wrapped, _SpatialModule):
                dummy = dummy.reshape(1, inp.size)
            out = wrapped(dummy)
        return tuple(out.shape[1:])

    def _maybe_wrap_spatial(self, layer, shape_in):
        if (
            len(shape_in) >= 2
            and isinstance(layer, (nn.Conv2d, nn.Conv1d, nn.MaxPool2d, nn.AvgPool2d, nn.BatchNorm2d))
        ):
            return _SpatialModule(layer, shape_in)
        return layer

    def _split_tensor_constant_args(self, node, op_name):
        a_arg, b_arg = node.args[0], node.args[1]
        if isinstance(a_arg, torch.fx.Node) and not isinstance(b_arg, torch.fx.Node):
            return self._get_value(a_arg), b_arg, True
        if not isinstance(a_arg, torch.fx.Node) and isinstance(b_arg, torch.fx.Node):
            return self._get_value(b_arg), a_arg, False
        if isinstance(a_arg, torch.fx.Node) and isinstance(b_arg, torch.fx.Node):
            return None, None, True
        raise ConversionError(
            f"{op_name} node '{node.name}' requires at least one tensor input."
        )

    def _constant_vector(self, node_name, const, value: GraphValue):
        const_arr = np.asarray(const, dtype=np.float32)
        if const_arr.ndim == 0:
            return np.full((value.size,), float(const_arr), dtype=np.float32)
        if const_arr.size == value.size:
            return const_arr.reshape(value.size)
        raise ConversionError(
            f"Node '{node_name}' constant has incompatible size {const_arr.shape} "
            f"for tensor size {value.size}."
        )

    def _normalize_dims(self, dims, ndims, node_name):
        normalized = tuple(int(d) % ndims for d in dims)
        if len(normalized) != ndims:
            raise ConversionError(
                f"Node '{node_name}' expected {ndims} permutation dims, got {dims!r}."
            )
        if len(set(normalized)) != ndims:
            raise ConversionError(
                f"Node '{node_name}' permutation dims must be unique, got {dims!r}."
            )
        return normalized

    def _shared_wrapper(self, layer, shape_in):
        if (
            len(shape_in) >= 2
            and isinstance(layer, (nn.Conv2d, nn.Conv1d, nn.MaxPool2d, nn.AvgPool2d, nn.BatchNorm2d))
        ):
            key = (id(layer), tuple(shape_in))
            if key not in self._shared_spatial_wrappers:
                self._shared_spatial_wrappers[key] = _SpatialModule(layer, shape_in)
            return self._shared_spatial_wrappers[key]
        return layer

    def _convert_fallback_function(self, node) -> GraphValue:
        if not self.allow_fallback:
            raise ConversionError(
                f"No converter for function/method node '{node.name}' ({node.target!r})."
            )

        input_values = self._collect_input_values(node.args)
        out_shape = self._tensor_shape(node)
        size_out = self._size_from_shape(out_shape)

        self._warn_torchnode_fallback(
            kind="operation",
            name=node.name,
            detail=repr(node.target),
        )

        callable_fn = self._build_fallback_callable(node, input_values)
        total_in = sum(v.size for v in input_values)
        out = TorchNode(
            callable_fn,
            size_in=total_in,
            size_out=size_out,
            shape_in=(total_in,),
            shape_out=out_shape,
            label=node.name,
        )
        aggregator = nengo.Node(size_in=total_in, label=f"{node.name}_concat")
        start = 0
        for val in input_values:
            transform = np.zeros((aggregator.size_in, val.size), dtype=np.float32)
            transform[start : start + val.size, :] = np.eye(val.size, dtype=np.float32)
            syn = self.synapse if isinstance(val.obj, nengo.ensemble.Neurons) else None
            self._connect(val.obj, aggregator, transform=transform, synapse=syn)
            start += val.size
        self._connect(aggregator, out, synapse=None)
        return GraphValue(out, size_out, out_shape)

    def _collect_input_values(self, args) -> List[GraphValue]:
        values = []

        def visit(arg):
            if isinstance(arg, torch.fx.Node):
                values.append(self._get_value(arg))
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    visit(item)

        visit(args)
        return values

    def _build_fallback_callable(self, node, input_values):
        value_specs = [(val.size, val.shape) for val in input_values]

        def reconstruct(x):
            pieces = []
            start = 0
            for size, shape in value_specs:
                piece = x[:, start : start + size]
                pieces.append(piece.reshape((x.shape[0],) + tuple(shape)))
                start += size
            return pieces

        if node.op == "call_function":
            target = node.target

            def fn(x):
                pieces = reconstruct(x)
                args, kwargs = self._rebuild_args(node.args, node.kwargs, pieces)
                return target(*args, **kwargs)

            return fn

        if node.op == "call_method":
            method = node.target

            def fn(x):
                pieces = reconstruct(x)
                args, kwargs = self._rebuild_args(node.args, node.kwargs, pieces)
                receiver, *rest = args
                return getattr(receiver, method)(*rest, **kwargs)

            return fn

        raise ConversionError(f"Cannot build fallback callable for node '{node.name}'")

    def _rebuild_args(self, args, kwargs, pieces):
        piece_iter = iter(pieces)

        def replace(arg):
            if isinstance(arg, torch.fx.Node):
                return next(piece_iter)
            if isinstance(arg, tuple):
                return tuple(replace(a) for a in arg)
            if isinstance(arg, list):
                return [replace(a) for a in arg]
            return arg

        return replace(args), {k: replace(v) for k, v in kwargs.items()}

    def _convert_output(self, node):
        outputs = node.args[0]
        if isinstance(outputs, torch.fx.Node):
            self.output_structure = self._record_outputs(outputs, prefix="output_0")
        else:
            self.output_structure = self._record_outputs(outputs)

    def _record_outputs(self, outputs, prefix="output"):
        if isinstance(outputs, torch.fx.Node):
            value = self._get_value(outputs)
            key = prefix
            self.outputs[key] = value.obj
            return value.obj

        if isinstance(outputs, tuple):
            return tuple(
                self._record_outputs(out, prefix=f"{prefix}_{idx}")
                for idx, out in enumerate(outputs)
            )

        if isinstance(outputs, list):
            return [
                self._record_outputs(out, prefix=f"{prefix}_{idx}")
                for idx, out in enumerate(outputs)
            ]

        if isinstance(outputs, dict):
            recorded = {}
            for key, out in outputs.items():
                out_key = str(key)
                recorded[key] = self._record_outputs(out, prefix=out_key)
            return recorded

        raise ConversionError(
            f"Unsupported output structure element of type {type(outputs).__name__}."
        )

    def _warn_torchnode_fallback(self, kind, name, detail):
        warnings.warn(
            f"{kind.capitalize()} '{name}' ({detail}) uses TorchNode fallback; "
            "spiking fidelity may differ from native conversion."
        )
