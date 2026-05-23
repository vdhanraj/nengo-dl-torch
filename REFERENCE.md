# NengoDL Reference

This document describes the public API of the nengo-dl package (PyTorch backend).

---

## `nengo_dl.Simulator`

The main entry point for running and training Nengo networks.

```python
sim = nengo_dl.Simulator(network, dt=0.001, minibatch_size=1, device=None, seed=None)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `network` | `nengo.Network` | required | The Nengo network to simulate |
| `dt` | `float` | `0.001` | Simulation timestep in seconds |
| `minibatch_size` | `int` | `1` | Number of samples per mini-batch |
| `device` | `str` or `torch.device` | auto | Device to run on (`"cpu"`, `"cuda"`, etc.) |
| `seed` | `int` | `None` | Random seed for reproducibility |

### Methods

#### `sim.run(time_in_seconds, data=None, progress_bar=True)`

Run the simulation for the given duration.

- `time_in_seconds` — wall-clock duration to simulate (converted to steps via `dt`)
- `data` — dict mapping `nengo.Node` → array of shape `(n_steps, size)` for input injection
- `progress_bar` — whether to show a tqdm progress bar

#### `sim.run_steps(n_steps, data=None)`

Run for an exact number of timesteps. Lower-level than `run`.

#### `sim.reset(seed=None)`

Reset all simulation state (voltages, refractory times, probe buffers) to initial values.

#### `sim.fit(inputs, targets, n_epochs=1, optimizer=None, shuffle=True)`

Train the network via backpropagation through time.

- `inputs` — dict mapping `nengo.Node` → array of shape `(n_samples, n_steps, node_size)`
- `targets` — dict mapping `nengo.Probe` → array of shape `(n_samples, n_steps, probe_size)`
- `n_epochs` — number of passes over the dataset
- `optimizer` — a `torch.optim.Optimizer` instance (default: Adam, lr=0.001)
- `shuffle` — whether to shuffle samples between epochs

Returns a dict with key `"loss"` containing per-epoch loss values.

#### `sim.evaluate(inputs, targets, train_mode=False)`

Evaluate mean squared error (or custom loss) on the given data without updating parameters.

Returns a dict with key `"loss"`.

#### `sim.get_data(probe)`

Retrieve recorded probe data as a NumPy array of shape `(minibatch_size, n_steps, probe_size)`.

- `probe` — a `nengo.Probe` object that was added to the network

#### `sim.save_params(path)`

Save all trainable parameters to a file (NumPy `.npz` format).

#### `sim.load_params(path)`

Load trainable parameters from a `.npz` file saved by `save_params`.

#### `sim.freeze_params(objects)`

Freeze parameters of specific Nengo objects so they are not updated during training.

- `objects` — a single object or list of `nengo.Ensemble`, `nengo.Connection`, or `nengo.Node`

#### `sim.unfreeze_params(objects)`

Unfreeze parameters of the given objects (reverse of `freeze_params`).

#### `sim.compile(optimizer=None, loss=None, metrics=None)`

Configure the optimizer and loss function used by `fit` and `evaluate`.

- `optimizer` — a `torch.optim.Optimizer` (default: Adam)
- `loss` — a loss function or dict mapping probe → loss function (default: MSE)
- `metrics` — additional metric functions (not yet used)

#### Context manager usage

```python
with nengo_dl.Simulator(net) as sim:
    sim.run(1.0)
    data = sim.get_data(my_probe)
```

The context manager calls `sim.reset()` on entry and releases resources on exit.

---

## `nengo_dl.configure_settings`

```python
nengo_dl.configure_settings(
    trainable=None,
    inference_only=None,
    lif_smoothing=None,
    dtype=None,
    keep_history=None,
    stateful=None,
    use_loop=None,
)
```

Configure nengo-dl settings on the currently active `nengo.Network`. Must be called inside a `with nengo.Network() as net:` block.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainable` | `bool` | `True` | Whether network parameters are trainable by default |
| `inference_only` | `bool` | `False` | Skip training-specific ops (faster inference) |
| `lif_smoothing` | `float` | `0.0` | SoftLIF smoothing sigma (0 = use default 0.002, >0 = explicit value) |
| `dtype` | `str` | `"float32"` | Default float dtype (`"float32"` or `"float64"`) |
| `keep_history` | `bool` | `True` | Keep probe data from all timesteps |
| `stateful` | `bool` | `False`` | Preserve simulation state between `run_steps` calls |
| `use_loop` | `bool` | — | Unused; kept for API compatibility |

**Example**

```python
with nengo.Network() as net:
    nengo_dl.configure_settings(trainable=False, lif_smoothing=0.01)
    ens = nengo.Ensemble(100, dimensions=1)
    # ens weights are frozen (not trained)
```

---

## `nengo_dl.TorchNode`

Wraps an arbitrary `torch.nn.Module` as a Nengo `Node`, allowing PyTorch layers to be embedded in a Nengo network.

```python
node = nengo_dl.TorchNode(module, size_in=None, size_out=None, pass_time=False)
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `torch.nn.Module` | The PyTorch module to wrap |
| `size_in` | `int` | Number of input dimensions (inferred if `None`) |
| `size_out` | `int` | Number of output dimensions (inferred if `None`) |
| `pass_time` | `bool` | If `True`, prepend the current sim time to the input |

The module's `forward(x)` is called each timestep with `x` of shape `(batch, size_in)`.

**Example**

```python
import torch.nn as nn

mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1))

with nengo.Network() as net:
    node = nengo_dl.TorchNode(mlp, size_in=2, size_out=1)
    inp = nengo.Node(np.zeros(2))
    nengo.Connection(inp, node)
```

---

## `nengo_dl.Layer`

A convenience wrapper that applies a `torch.nn.Module` layer to an existing Nengo object, returning a new `Node`.

```python
output_node = nengo_dl.Layer(module)(input_node_or_ensemble, **connection_kwargs)
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `torch.nn.Module` or callable | Layer to apply |

**Example**

```python
dense = nengo_dl.Layer(nn.Linear(10, 5))
out = dense(inp_ensemble)
```

---

## `nengo_dl.Converter`

Converts a Keras model (or compatible architecture) to an equivalent Nengo network with `TorchNode` layers.

```python
converter = nengo_dl.Converter(model, allow_fallback=True, inference_only=False)
net = converter.net
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | Keras model | The model to convert |
| `allow_fallback` | `bool` | If `True`, unsupported layers fall back to `TorchNode` |
| `inference_only` | `bool` | If `True`, skip training-specific ops |

**Attributes**

- `converter.net` — the resulting `nengo.Network`
- `converter.inputs` — dict mapping Keras layer → input `nengo.Node`
- `converter.outputs` — dict mapping Keras layer → output `nengo.Probe`

---

## Neuron Types (`nengo_dl.neurons`)

### `SoftLIFRate`

A smoothed LIF rate neuron that is differentiable everywhere, avoiding the hard threshold discontinuity of `nengo.LIFRate`.

```python
neuron = SoftLIFRate(sigma=0.002, tau_rc=0.02, tau_ref=0.002, amplitude=1.0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma` | `0.002` | Smoothing parameter (larger = smoother, further from true LIF) |
| `tau_rc` | `0.02` | RC time constant in seconds |
| `tau_ref` | `0.002` | Refractory period in seconds |
| `amplitude` | `1.0` | Amplitude scaling factor |

Uses `softplus((J-1)/sigma)*sigma` in place of `max(J-1, 0)` to smooth the threshold.

### `SpikingLeakyReLU`

A spiking leaky ReLU neuron. Integrates current and fires when voltage exceeds 1. During training, replaced by the rate approximation `LeakyReLU`.

```python
neuron = SpikingLeakyReLU(negative_slope=0.01, amplitude=1.0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `negative_slope` | `0.01` | Slope for negative inputs |
| `amplitude` | `1.0` | Output amplitude scaling |

### `LeakyReLU`

Rate-mode leaky ReLU neuron (fully differentiable). Output = `leaky_relu(J) * amplitude`.

```python
neuron = LeakyReLU(negative_slope=0.01, amplitude=1.0)
```

---

## Loss Functions (`nengo_dl.losses`)

All loss classes follow the interface:

```python
loss_fn = LossClass(...)
loss_value = loss_fn(outputs, targets)  # tensors of shape (batch, n_steps, size)
```

### `MSE`

Mean squared error loss, averaged over batch, time, and output dimensions.

```python
loss = nengo_dl.losses.MSE()
```

### `Regularization`

L1 or L2 regularization on probe outputs (encourages sparse or small activations).

```python
loss = nengo_dl.losses.Regularization(order=2, weight=1e-4)
```

| Parameter | Description |
|-----------|-------------|
| `order` | Norm order: `1` for L1, `2` for L2 |
| `weight` | Scaling factor for the regularization term |

### `SimilarityLoss`

Cosine similarity loss. Encourages outputs to be aligned with targets in direction.

```python
loss = nengo_dl.losses.SimilarityLoss()
```

### `TargetedDropout`

Applies a loss that encourages a fixed fraction of neurons to be silent (for network sparsification).

```python
loss = nengo_dl.losses.TargetedDropout(drop_p=0.5, target_p=0.1, weight=1.0)
```

---

## Builder System

The builder system converts Nengo operators into PyTorch operations executed each timestep.

### `nengo_dl.builder.Builder`

Registry and executor for operator builders.

```python
builder = Builder(ops, signals, config)
builder.run_step()  # execute one timestep
```

#### `Builder.register(op_type)`

Class decorator to register an `OpBuilder` subclass for a given Nengo operator type.

```python
@Builder.register(nengo.builder.operator.Copy)
class CopyBuilder(OpBuilder):
    def build_pre(self, ops, signals, config): ...
    def build_step(self, ops, signals, config): ...
```

### `nengo_dl.builder.OpBuilder`

Base class for all operator builders. Subclass this to add support for new operator types.

| Method | Description |
|--------|-------------|
| `build_pre(ops, signals, config)` | Called once before simulation starts; allocate buffers here |
| `build_step(ops, signals, config)` | Called every timestep; implement the operator logic here |

### `nengo_dl.builder.BuildConfig`

Immutable configuration object passed to every `build_step` call.

| Attribute | Type | Description |
|-----------|------|-------------|
| `dt` | `float` | Simulation timestep |
| `minibatch_size` | `int` | Batch size |
| `training` | `bool` | Whether in training mode |
| `lif_smoothing` | `float` | LIF surrogate smoothing |
| `inference_only` | `bool` | Inference-only mode flag |
| `device` | `torch.device` | Target device |
| `dtype` | `torch.dtype` | Target dtype |
| `rng` | `np.random.Generator` | Seeded random generator |

---

## `nengo_dl.tensor_graph.TensorGraph`

The core `torch.nn.Module` that wraps a built Nengo model and drives simulation.

```python
tg = TensorGraph(
    model, dt=0.001, minibatch_size=1, device=None,
    dtype=torch.float32, lif_smoothing=0.0,
    inference_only=False, trainable=True
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nengo.builder.Model` | A built Nengo model |
| `dt` | `float` | Simulation timestep |
| `minibatch_size` | `int` | Batch size |
| `device` | `torch.device` | Target device |
| `dtype` | `torch.dtype` | Target dtype |
| `lif_smoothing` | `float` | LIF smoothing parameter |
| `inference_only` | `bool` | Skip training ops |
| `trainable` | `bool` | Whether readonly signals become `nn.Parameter` |

### `TensorGraph.forward(n_steps, input_data=None, training=False)`

Run the simulation for `n_steps` timesteps.

- `input_data` — dict mapping `nengo.Node` → tensor of shape `(batch, n_steps, size)`
- Returns dict mapping `nengo.Probe` → tensor of shape `(batch, n_steps, probe_size)`

### `TensorGraph.reset_state()`

Reset all time-varying signals to their initial values.

### `TensorGraph.get_weights()`

Returns all trainable parameters as a `dict[str, np.ndarray]`.

### `TensorGraph.set_weights(weights)`

Set trainable parameters from a `dict[str, np.ndarray]`.

---

## `nengo_dl.signals.SignalDict`

Manages the mapping from Nengo `Signal` objects to batched `torch.Tensor` values.

```python
signals = SignalDict(minibatch_size, device, dtype)
```

| Method | Description |
|--------|-------------|
| `add_signal(sig, trainable=False)` | Allocate a tensor for `sig`; wraps in `nn.Parameter` if `trainable=True` |
| `gather(sig)` → `Tensor(batch, *shape)` | Read a signal's current value |
| `scatter(sig, val, mode="set")` | Write `val` into a signal (`mode`: `"set"`, `"inc"`, `"mul"`) |
| `reset()` | Reset all signals to their initial values |
| `get_all_parameters()` | Returns `dict[str, nn.Parameter]` for all trainable signals |

---

## `nengo_dl.config.get_setting`

```python
value = nengo_dl.config.get_setting(network_or_model, setting, default=None)
```

Retrieve a nengo-dl setting from a `nengo.Network` or built `nengo.builder.Model`. Falls back to the global settings cache if the network config does not have the setting.

| Parameter | Description |
|-----------|-------------|
| `network_or_model` | A `nengo.Network` or `nengo.builder.Model` |
| `setting` | Setting name string (e.g., `"trainable"`, `"lif_smoothing"`) |
| `default` | Value to return if the setting is not found |

---

## Summary Table

| Symbol | Module | Description |
|--------|--------|-------------|
| `Simulator` | `nengo_dl` | Main simulation and training interface |
| `configure_settings` | `nengo_dl` | Set network-level nengo-dl options |
| `TorchNode` | `nengo_dl` | Embed a `nn.Module` in a Nengo network |
| `Layer` | `nengo_dl` | Apply a layer between two Nengo objects |
| `Converter` | `nengo_dl` | Convert Keras models to Nengo networks |
| `SoftLIFRate` | `nengo_dl.neurons` | Smoothed LIF rate neuron |
| `SpikingLeakyReLU` | `nengo_dl.neurons` | Spiking leaky ReLU neuron |
| `LeakyReLU` | `nengo_dl.neurons` | Rate leaky ReLU neuron |
| `losses.MSE` | `nengo_dl.losses` | Mean squared error |
| `losses.Regularization` | `nengo_dl.losses` | L1/L2 regularization |
| `losses.SimilarityLoss` | `nengo_dl.losses` | Cosine similarity loss |
| `losses.TargetedDropout` | `nengo_dl.losses` | Sparsification loss |
| `Builder` | `nengo_dl.builder` | Operator registry and executor |
| `OpBuilder` | `nengo_dl.builder` | Base class for operator builders |
| `BuildConfig` | `nengo_dl.builder` | Per-step configuration container |
| `TensorGraph` | `nengo_dl.tensor_graph` | Core `nn.Module` wrapping Nengo model |
| `SignalDict` | `nengo_dl.signals` | Signal → tensor mapping |
| `get_setting` | `nengo_dl.config` | Read a nengo-dl config setting |
