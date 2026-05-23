"""Main computation graph for nengo-dl (PyTorch backend).

``TensorGraph`` is an ``nn.Module`` that wraps a built Nengo model and
exposes a standard PyTorch ``forward()`` interface.  It handles:

- Mapping Nengo signals to torch tensors (``SignalDict``)
- Identifying trainable parameters (weights, biases, encoders, decoders)
- Running the simulation loop for *n_steps* timesteps
- Collecting probe outputs
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import nengo
import nengo.builder
from nengo.builder import Builder as NengoBuilder, Model as NengoModel
from nengo.builder.signal import Signal
from nengo.builder.operator import (
    Copy, DotInc, ElementwiseInc, Reset, SimPyFunc, TimeUpdate
)
from nengo.builder.probe import SimProbe
from nengo.builder.processes import SimProcess

from .builder import Builder, BuildConfig
from .signals import SignalDict
from .graph_optimizer import topo_sort

# Import all operator builders so they register themselves
from . import op_builders  # noqa: F401
from . import neuron_builders  # noqa: F401
from . import process_builders  # noqa: F401
from . import tensor_node  # noqa: F401  (registers SimTorchNode builder)


def _build_sig_owner_map(model: NengoModel):
    """Build a dict mapping signal → (owner_object, role_name)."""
    sig_owner = {}
    for obj, roles in model.sig.items():
        for role, sig in roles.items():
            if hasattr(sig, 'base'):
                sig_owner[sig.base] = (obj, role)
            sig_owner[sig] = (obj, role)
    return sig_owner


def _check_object_trainable(obj, model: NengoModel, default: bool) -> bool:
    """Return whether a Nengo object is trainable according to network config."""
    from .config import _global_settings

    # nengo.ensemble.Neurons stores biases/voltages but config is on the parent Ensemble
    if hasattr(obj, 'ensemble') and isinstance(getattr(obj, 'ensemble', None), nengo.Ensemble):
        obj = obj.ensemble

    try:
        network = model.toplevel
        if network is not None:
            cfg = network.config
            # Check instance-level config first
            try:
                inst_val = getattr(cfg[obj], 'trainable', None)
                if inst_val is not None:
                    return bool(inst_val)
            except Exception:
                pass
            # Check class-level config (e.g. net.config[nengo.Ensemble].trainable = True)
            try:
                cls_val = getattr(cfg[type(obj)], 'trainable', None)
                if cls_val is not None:
                    return bool(cls_val)
            except Exception:
                pass
            # Check network-level default (set via configure_settings)
            try:
                net_val = getattr(cfg[type(network)], 'trainable', None)
                if net_val is not None:
                    return bool(net_val)
            except Exception:
                pass
    except Exception:
        pass
    # Fall back to global settings (set by configure_settings when outside a network context,
    # or as a cache for the value set inside a network context)
    return bool(_global_settings.get('trainable', default))


def _is_trainable_signal(sig: Signal, model: NengoModel,
                          sig_owner: dict = None, global_trainable: bool = True) -> bool:
    """Return True if *sig* is a trainable parameter, respecting network config.

    A signal is trainable if:
    1. It is readonly (set by builder, not changed during simulation)
    2. It is not a scalar constant (ZERO, ONE, step, time)
    3. Its owning Nengo object is marked trainable in the network config
    """
    if not sig.readonly:
        return False
    if sig.size <= 1 and sig.name in ("ZERO", "ONE"):
        return False
    if sig.name in ("step", "time"):
        return False
    init = sig.initial_value
    if not isinstance(init, np.ndarray):
        init = np.array(init)
    if init.size <= 1:
        return False

    # Check trainable config for the owning object
    if sig_owner is not None:
        entry = sig_owner.get(sig)
        if entry is not None:
            obj, role = entry
            return _check_object_trainable(obj, model, default=global_trainable)

    return global_trainable


class TensorGraph(nn.Module):
    """PyTorch nn.Module wrapping a Nengo network for differentiable simulation.

    Parameters
    ----------
    model : nengo.builder.Model
        A built Nengo model (produced by ``NengoBuilder.build``).
    dt : float
        Simulation timestep in seconds.
    minibatch_size : int
        Number of samples per batch.
    device : str or torch.device, optional
        Device to run on (default: auto-detect CUDA).
    dtype : torch.dtype, optional
        Default float dtype (default: ``torch.float32``).
    lif_smoothing : float, optional
        LIF surrogate smoothing parameter (0 = pure spiking, >0 = smooth).
    inference_only : bool, optional
        If True, skip training-specific ops.
    rate_mode : bool, optional
        If True, use rate approximations for spiking neurons during inference.
    trainable : bool, optional
        Whether readonly signals are treated as trainable parameters.
    """

    def __init__(
        self,
        model: NengoModel,
        dt: float = 0.001,
        minibatch_size: int = 1,
        device=None,
        dtype=torch.float32,
        lif_smoothing: float = 0.0,
        inference_only: bool = False,
        trainable: bool = True,
    ):
        super().__init__()
        self.model = model
        self.dt = dt
        self.minibatch_size = minibatch_size
        self.lif_smoothing = lif_smoothing
        self.inference_only = inference_only

        # Device selection
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype

        # Build signal dict
        self.signals = SignalDict(minibatch_size, device, dtype)
        self._trainable = trainable
        self._probe_sig_map: Dict = {}  # Probe → Signal
        self._input_node_sigs: Dict = {}  # Node → Signal (for input injection)
        self._time_sig: Optional[Signal] = None
        self._step_sig: Optional[Signal] = None

        # Collect all signals from operators
        self._collect_signals()

        # Register parameters with nn.Module
        self._param_dict = nn.ParameterDict()
        for name, param in self.signals.get_all_parameters().items():
            safe_name = name.replace("-", "_").replace(".", "_").replace(" ", "_")
            self._param_dict[safe_name] = param

        # Register TorchNode modules as submodules so their parameters
        # are tracked by PyTorch and included in the optimizer.
        self._torch_modules = nn.ModuleList()
        self._torch_module_map: Dict = {}  # id(module) → index in ModuleList
        self._collect_torch_modules()

        # Build the operator executor
        rng = np.random.default_rng(0)
        config = BuildConfig(
            dt=dt,
            minibatch_size=minibatch_size,
            training=False,  # will be overridden per-call
            rate_mode=False,  # will be overridden per-call
            lif_smoothing=lif_smoothing,
            inference_only=inference_only,
            device=device,
            dtype=dtype,
            rng=rng,
        )
        # Sort operators into correct dependency order
        sorted_ops = topo_sort(model.operators)
        self._builder = Builder(sorted_ops, self.signals, config)
        self._config = config

        # Map probes to their signals
        for probe in model.probes:
            if probe in model.sig and "in" in model.sig[probe]:
                self._probe_sig_map[probe] = model.sig[probe]["in"]

        # Map input Nodes to their output signals
        for node in model.toplevel.all_nodes:
            if node in model.sig and "out" in model.sig[node]:
                self._input_node_sigs[node] = model.sig[node]["out"]

        # Find time and step signals
        for op in model.operators:
            if isinstance(op, TimeUpdate):
                self._time_sig = op.time
                self._step_sig = op.step
                break

    def _collect_signals(self):
        """Allocate all signals referenced by operators."""
        seen = {}  # id(sig.base) -> sig.base, first-seen order from topo-sorted ops
        for op in self.model.operators:
            for sig in op.reads + op.sets + op.incs + op.updates:
                k = id(sig.base)
                if k not in seen:
                    seen[k] = sig.base
        all_sigs = seen.values()

        # Build signal→owner map for trainable config lookup
        sig_owner = _build_sig_owner_map(self.model)

        # Register each base signal
        for sig in all_sigs:
            trainable = self._trainable and _is_trainable_signal(
                sig, self.model, sig_owner=sig_owner, global_trainable=self._trainable
            )
            if sig not in self.signals:
                self.signals.add_signal(sig, trainable=trainable)

    def _collect_torch_modules(self):
        """Discover nn.Module objects from SimTorchNode operators and register them."""
        from .tensor_node import SimTorchNode as _SimTorchNode
        idx = 0
        for op in self.model.operators:
            if isinstance(op, _SimTorchNode) and op.module is not None:
                module = op.module
                m_id = id(module)
                if m_id not in self._torch_module_map:
                    # Move module to correct device/dtype
                    module.to(device=self.device, dtype=self.dtype)
                    self._torch_modules.append(module)
                    self._torch_module_map[m_id] = idx
                    idx += 1

    def forward(
        self,
        n_steps: int,
        input_data: Optional[Dict] = None,
        training: bool = False,
        rate_mode: bool = False,
    ) -> Dict:
        """Run the simulation for *n_steps* timesteps.

        Parameters
        ----------
        n_steps : int
            Number of timesteps to simulate.
        input_data : dict, optional
            Maps ``nengo.Node`` objects to input tensors of shape
            ``(batch, n_steps, node_size)`` or ``(n_steps, node_size)``.
        training : bool, optional
            If True, use training-mode neuron dynamics (e.g. rate approx).
        rate_mode : bool, optional
            If True, use rate approximations for spiking neurons without
            enabling gradient/training behavior.

        Returns
        -------
        dict
            Maps ``nengo.Probe`` objects to tensors of shape
            ``(batch, n_steps, probe_size)``.
        """
        # Reset probe data buffers
        if not hasattr(self.signals, "_probe_data"):
            self.signals._probe_data = {}
        for sig in self._probe_sig_map.values():
            self.signals._probe_data[sig] = []

        # Update config training flag
        self._config.training = training
        self._config.rate_mode = rate_mode

        # Pre-process input data
        processed_inputs: Dict[Signal, torch.Tensor] = {}
        if input_data is not None:
            for node, data in input_data.items():
                sig = self._input_node_sigs.get(node)
                if sig is None:
                    warnings.warn(f"Node {node} has no output signal in model; ignoring.")
                    continue
                data_t = _to_tensor(data, self.dtype, self.device)
                if data_t.dim() == 2:
                    # (n_steps, size) → (1, n_steps, size)
                    data_t = data_t.unsqueeze(0)
                # data_t: (batch, n_steps, size)
                processed_inputs[sig] = data_t

        # Simulation loop
        for t in range(n_steps):
            # Inject input for this timestep
            for sig, data_t in processed_inputs.items():
                step_idx = min(t, data_t.shape[1] - 1)
                val = data_t[:, step_idx, :]  # (batch, size)
                self.signals.scatter(sig, val, mode="set")

            # Run one step of all operators
            self._builder.run_step()

        # Collect probe outputs
        results = {}
        for probe, sig in self._probe_sig_map.items():
            probe_steps = self.signals._probe_data.get(sig, [])
            if probe_steps:
                # Stack along time: list of (batch, *shape) → (batch, n_steps, *shape)
                stacked = torch.stack(probe_steps, dim=1)
                results[probe] = stacked
            else:
                results[probe] = torch.zeros(
                    self.minibatch_size, n_steps, *sig.shape,
                    dtype=self.dtype, device=self.device
                )

        return results

    def reset_state(self):
        """Reset all time-varying state to initial values."""
        self.signals.reset()

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return all trainable parameters as a dict of numpy arrays."""
        weights = {
            name: param.data.cpu().numpy()
            for name, param in self._param_dict.items()
        }

        for idx, module in enumerate(self._torch_modules):
            prefix = f"torch_module_{idx:04d}."
            for name, tensor in module.state_dict().items():
                if torch.is_tensor(tensor):
                    weights[prefix + name] = tensor.detach().cpu().numpy()

        return weights

    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set trainable parameters from a dict of numpy arrays."""
        for name, val in weights.items():
            if name in self._param_dict:
                with torch.no_grad():
                    self._param_dict[name].copy_(
                        torch.tensor(val, dtype=self.dtype, device=self.device)
                    )

        for idx, module in enumerate(self._torch_modules):
            prefix = f"torch_module_{idx:04d}."
            state = module.state_dict()
            changed = False

            for name, current in state.items():
                key = prefix + name
                if key in weights:
                    state[name] = torch.as_tensor(
                        weights[key],
                        dtype=current.dtype,
                        device=current.device,
                    )
                    changed = True

            if changed:
                module.load_state_dict(state, strict=False)

    def extra_repr(self) -> str:
        return (
            f"dt={self.dt}, minibatch_size={self.minibatch_size}, "
            f"device={self.device}, n_operators={len(self.model.operators)}"
        )


def _to_tensor(x, dtype, device) -> torch.Tensor:
    """Convert numpy array or tensor to the target dtype/device."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(np.asarray(x, dtype=np.float32), dtype=dtype, device=device)
