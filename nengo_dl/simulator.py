"""Main Simulator for nengo-dl (PyTorch backend).

Provides a Keras-like API (``fit``, ``predict``, ``evaluate``) backed by
PyTorch, with full Nengo network support.
"""

import warnings
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import nengo
import nengo.builder
from nengo.builder import Builder as NengoBuilder, Model as NengoModel

from .config import get_setting
from .tensor_graph import TensorGraph, _to_tensor


class SimulationData:
    """Dictionary-like interface for accessing simulation results.

    After ``Simulator.run_steps()``, access probe data via
    ``sim.data[probe]`` which returns a numpy array of shape
    ``(n_steps, probe_size)`` (batch dimension is squeezed when
    ``minibatch_size == 1``).

    Also supports accessing Nengo object parameters:
    ``sim.data[ensemble]``, ``sim.data[connection]``, etc.
    """

    def __init__(self, simulator: "Simulator"):
        self._sim = simulator
        self._probe_data: Dict[nengo.Probe, np.ndarray] = {}

    def _store_probe(self, probe: nengo.Probe, data: torch.Tensor):
        """Store probe results from a forward pass.

        Parameters
        ----------
        probe : nengo.Probe
            The probe object.
        data : torch.Tensor
            Shape ``(batch, n_steps, *probe_shape)``.
        """
        arr = data.detach().cpu().numpy()
        if self._sim.minibatch_size == 1:
            arr = arr[0]  # drop batch dim → (n_steps, probe_shape...)
        self._probe_data[probe] = arr

    def __getitem__(self, key):
        if isinstance(key, nengo.Probe):
            if key not in self._probe_data:
                raise KeyError(
                    f"Probe {key} has no data. Did you call run_steps() first?"
                )
            return self._probe_data[key]

        # Nengo object parameter access
        model = self._sim._model
        tg = self._sim.tensor_graph

        if isinstance(key, nengo.Ensemble):
            return self._get_ensemble_params(key)
        elif isinstance(key, nengo.Connection):
            return self._get_connection_params(key)
        elif isinstance(key, nengo.ensemble.Neurons):
            return self._get_neuron_params(key)
        else:
            raise KeyError(f"Cannot access data for {type(key).__name__}")

    def _get_ensemble_params(self, ens: nengo.Ensemble):
        model = self._sim._model
        params = model.params.get(ens)
        if params is None:
            return None
        result = {}
        for attr in ("gain", "bias", "scaled_encoders", "encoders"):
            val = getattr(params, attr, None)
            if val is not None:
                result[attr] = val
        # Try to get current trained values
        tg = self._sim.tensor_graph
        sigs = model.sig.get(ens, {})
        if "encoders" in sigs:
            sig = sigs["encoders"]
            current = tg.signals.gather(sig)
            result["scaled_encoders"] = current.detach().cpu().numpy()
        return result

    def _get_connection_params(self, conn: nengo.Connection):
        model = self._sim._model
        tg = self._sim.tensor_graph
        sigs = model.sig.get(conn, {})
        result = {}
        if "weights" in sigs:
            sig = sigs["weights"]
            if sig is not None:
                current = tg.signals.gather(sig)
                result["weights"] = current.detach().cpu().numpy()
        return result if result else model.params.get(conn)

    def _get_neuron_params(self, neurons):
        model = self._sim._model
        sigs = model.sig.get(neurons, {})
        result = {}
        for key, sig in sigs.items():
            current = self._sim.tensor_graph.signals.gather(sig)
            result[key] = current.detach().cpu().numpy()
        return result if result else model.params.get(neurons)

    def __contains__(self, key):
        if isinstance(key, nengo.Probe):
            return key in self._probe_data
        if isinstance(key, (nengo.Ensemble, nengo.Connection, nengo.ensemble.Neurons)):
            model = self._sim._model
            return key in model.sig or key in model.params
        return False

    def __repr__(self):
        return f"SimulationData(probes={list(self._probe_data.keys())})"


class Simulator:
    """nengo-dl Simulator (PyTorch backend).

    Drop-in (near) replacement for NengoDL's ``Simulator``, backed by PyTorch
    instead of TensorFlow/Keras.

    Parameters
    ----------
    network : nengo.Network
        The Nengo network to simulate.
    dt : float
        Simulation timestep in seconds (default 0.001).
    seed : int, optional
        Random seed for reproducibility.
    model : nengo.builder.Model, optional
        Pre-built model (if None, it is built automatically).
    device : str or torch.device, optional
        Computation device (default: auto-detect CUDA, else CPU).
    minibatch_size : int, optional
        Number of samples per minibatch for batched simulation (default 1).
    progress_bar : bool, optional
        Show progress during run (default True).

    Examples
    --------
    Basic simulation::

        with nengo.Network() as net:
            ens = nengo.Ensemble(100, 1)
            node = nengo.Node(np.sin)
            nengo.Connection(node, ens)
            p = nengo.Probe(ens, synapse=0.01)

        with nengo_dl.Simulator(net) as sim:
            sim.run(1.0)
        print(sim.data[p])

    Training::

        with nengo_dl.Simulator(net, minibatch_size=32) as sim:
            sim.compile(optimizer=torch.optim.Adam(sim.trainable_params(), lr=1e-3),
                        loss={p: torch.nn.MSELoss()})
            sim.fit(inputs={node: x_train}, targets={p: y_train}, epochs=10)
    """

    def __init__(
        self,
        network: nengo.Network,
        dt: float = 0.001,
        seed: Optional[int] = None,
        model: Optional[NengoModel] = None,
        device=None,
        minibatch_size: int = 1,
        progress_bar: bool = True,
    ):
        self.network = network
        self.dt = dt
        self.seed = seed
        self.minibatch_size = minibatch_size
        self.progress_bar = progress_bar

        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Build the Nengo model if not provided
        if model is None:
            model = NengoModel(dt=dt, label=str(network))
            NengoBuilder.build(model, network)
        self._model = model

        # Read settings from configure_settings (stored in network config / global)
        lif_smoothing = get_setting(network, "lif_smoothing", default=0.0)
        inference_only = get_setting(network, "inference_only", default=False)

        # Create the computation graph
        self.tensor_graph = TensorGraph(
            model=model,
            dt=dt,
            minibatch_size=minibatch_size,
            device=device,
            trainable=True,
            lif_smoothing=lif_smoothing,
            inference_only=inference_only,
        )

        # Simulation state
        self._n_steps = 0        # total steps simulated
        self._last_n_steps = 0   # steps in the most recent run_steps call
        self.data = SimulationData(self)

        # Training state
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._loss_fns: Dict[nengo.Probe, Callable] = {}
        self._loss_weights: Optional[Dict[nengo.Probe, float]] = None
        self._stateful = False
        self._closed = False

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Release resources."""
        if self._closed:
            return
        self._closed = True

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Simulator is closed.")

    def _progress(self, iterable, enabled: bool, **kwargs):
        if not enabled:
            return iterable

        try:
            from tqdm.auto import tqdm

            return tqdm(iterable, **kwargs)
        except Exception:
            return iterable

    def _run_inference_batches(
        self,
        n_steps: int,
        data,
        stateful: bool,
        rate_mode: bool,
        progress_bar: bool = False,
        desc: str = "run",
    ):
        """Run inference and return probe tensors with sample batching."""
        n_total = _count_samples(data, None)

        if n_total is not None and n_total > 0:
            if n_total < self.minibatch_size or n_total % self.minibatch_size != 0:
                raise ValueError(
                    f"Input sample count ({n_total}) must be divisible by "
                    f"minibatch_size ({self.minibatch_size})."
                )

        if n_total is not None and n_total > self.minibatch_size:
            chunk_results: Dict = {}
            starts = range(0, n_total, self.minibatch_size)
            starts = self._progress(
                starts,
                progress_bar,
                total=n_total // self.minibatch_size,
                desc=desc,
                leave=False,
            )
            for start in starts:
                end = min(start + self.minibatch_size, n_total)
                batch_data = {k: np.asarray(v)[start:end] for k, v in data.items()}
                if not stateful:
                    self.tensor_graph.reset_state()
                with torch.no_grad():
                    batch_results = self.tensor_graph.forward(
                        n_steps=n_steps,
                        input_data=batch_data,
                        training=False,
                        rate_mode=rate_mode,
                    )
                for probe, tensor in batch_results.items():
                    chunk_results.setdefault(probe, []).append(tensor.cpu())

            return {
                probe: torch.cat(tensors, dim=0)
                for probe, tensors in chunk_results.items()
            }

        if not stateful:
            self.tensor_graph.reset_state()

        with torch.no_grad():
            return self.tensor_graph.forward(
                n_steps=n_steps,
                input_data=data,
                training=False,
                rate_mode=rate_mode,
            )

    # ------------------------------------------------------------------
    # Running simulations
    # ------------------------------------------------------------------

    def run(self, time_in_seconds: float, data: Optional[Dict] = None,
            progress_bar: Optional[bool] = None,
            inference_mode: str = "spiking"):
        """Run the simulation for a specified amount of time.

        Parameters
        ----------
        time_in_seconds : float
            How long to simulate (seconds).
        data : dict, optional
            Input data. See ``run_steps``.
        progress_bar : bool, optional
            Show progress bar (overrides instance setting).
        inference_mode : {"spiking", "rate"}, optional
            If ``"spiking"``, spiking neurons emit discrete spikes. If
            ``"rate"``, spiking neurons use their rate approximation.
        """
        self._ensure_open()
        n_steps = int(np.round(time_in_seconds / self.dt))
        self.run_steps(
            n_steps,
            data=data,
            progress_bar=progress_bar,
            inference_mode=inference_mode,
        )

    def run_steps(
        self,
        n_steps: int,
        data: Optional[Dict] = None,
        progress_bar: Optional[bool] = None,
        stateful: bool = False,
        inference_mode: str = "spiking",
    ):
        """Run the simulation for a fixed number of timesteps.

        Parameters
        ----------
        n_steps : int
            Number of timesteps to simulate.
        data : dict, optional
            Maps ``nengo.Node`` objects to input arrays of shape
            ``(n_steps, node_size)`` or ``(batch, n_steps, node_size)``.
        progress_bar : bool, optional
            Overrides the instance-level ``progress_bar`` setting.
        stateful : bool, optional
            If True, preserve simulation state between calls.
        inference_mode : {"spiking", "rate"}, optional
            If ``"spiking"``, spiking neurons emit discrete spikes. If
            ``"rate"``, spiking neurons use their rate approximation.
        """
        self._ensure_open()
        show_pbar = self.progress_bar if progress_bar is None else progress_bar
        rate_mode = _inference_mode_to_rate(inference_mode)
        data = _normalize_data_dict(data, n_steps=n_steps)
        results = self._run_inference_batches(
            n_steps=n_steps,
            data=data,
            stateful=stateful,
            rate_mode=rate_mode,
            progress_bar=show_pbar,
            desc="run_steps",
        )
        for probe, tensor in results.items():
            self.data._store_probe(probe, tensor)

        # Always accumulate total step count; stateful only controls signal state
        self._n_steps += n_steps
        self._last_n_steps = n_steps  # remember for trange()

    def predict(
        self,
        x: Optional[Dict] = None,
        n_steps: int = 1,
        stateful: bool = False,
        batch_size: Optional[int] = None,
        inference_mode: str = "spiking",
    ) -> Dict[nengo.Probe, np.ndarray]:
        """Run inference and return probe data.

        Parameters
        ----------
        x : dict, optional
            Input data for Nodes.
        n_steps : int
            Number of timesteps.
        stateful : bool
            Preserve state between calls.
        inference_mode : {"spiking", "rate"}
            Inference mode for spiking neurons.

        Returns
        -------
        dict
            Maps probes to numpy arrays.
        """
        self._ensure_open()
        rate_mode = _inference_mode_to_rate(inference_mode)
        data = _normalize_data_dict(x, n_steps=n_steps)

        if batch_size is None:
            results = self._run_inference_batches(
                n_steps=n_steps,
                data=data,
                stateful=stateful,
                rate_mode=rate_mode,
                progress_bar=self.progress_bar,
                desc="predict",
            )
        else:
            if batch_size < self.minibatch_size:
                raise ValueError(
                    f"predict(batch_size={batch_size}) must be >= minibatch_size "
                    f"({self.minibatch_size})."
                )
            if batch_size % self.minibatch_size != 0:
                raise ValueError(
                    f"predict(batch_size={batch_size}) must be divisible by "
                    f"minibatch_size ({self.minibatch_size})."
                )
            n_total = _count_samples(data, None)
            if n_total and n_total % self.minibatch_size != 0:
                raise ValueError(
                    f"Input sample count ({n_total}) must be divisible by "
                    f"minibatch_size ({self.minibatch_size})."
                )

            chunk_results: Dict = {}
            starts = range(0, n_total, batch_size) if n_total else [0]
            starts = self._progress(
                starts,
                self.progress_bar,
                total=(n_total + batch_size - 1) // batch_size if n_total else 1,
                desc="predict",
                leave=False,
            )
            for start in starts:
                end = min(start + batch_size, n_total)
                batch_x = (
                    {k: np.asarray(v)[start:end] for k, v in data.items()}
                    if n_total
                    else data
                )
                batch_results = self._run_inference_batches(
                    n_steps=n_steps,
                    data=batch_x,
                    stateful=stateful,
                    rate_mode=rate_mode,
                    progress_bar=False,
                    desc="predict",
                )
                for probe, tensor in batch_results.items():
                    chunk_results.setdefault(probe, []).append(tensor.cpu())

            results = {
                probe: torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
                for probe, tensors in chunk_results.items()
            }

        for probe, tensor in results.items():
            self.data._store_probe(probe, tensor)
        return {p: self.data[p] for p in self._model.probes}

    def predict_on_batch(
        self,
        x: Optional[Dict] = None,
        n_steps: int = 1,
        inference_mode: str = "spiking",
    ) -> Dict[nengo.Probe, np.ndarray]:
        """Run inference on a single batch."""
        return self.predict(
            x=x,
            n_steps=n_steps,
            stateful=False,
            inference_mode=inference_mode,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compile(
        self,
        optimizer=None,
        loss: Optional[Dict] = None,
        loss_weights: Optional[Dict] = None,
    ):
        """Configure the model for training.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer or str, optional
            PyTorch optimizer, or a string like ``'adam'`` or ``'sgd'``.
            If a string, creates the optimizer with default settings.
        loss : dict, optional
            Maps ``nengo.Probe`` to a loss function (callable or
            ``torch.nn.Module``). If a single callable is given, it is
            applied to all probes.
        loss_weights : dict, optional
            Maps probes to scalar loss weights (default 1.0 each).
        """
        self._ensure_open()
        if isinstance(optimizer, str):
            params = self.trainable_params()
            optimizer_map = {
                "adam": lambda p: torch.optim.Adam(p, lr=1e-3),
                "sgd": lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9),
                "rmsprop": lambda p: torch.optim.RMSprop(p, lr=1e-3),
                "adamw": lambda p: torch.optim.AdamW(p, lr=1e-3),
            }
            key = optimizer.lower()
            if key not in optimizer_map:
                raise ValueError(
                    f"Unknown optimizer string '{optimizer}'. "
                    f"Choose from: {list(optimizer_map)}"
                )
            self._optimizer = optimizer_map[key](params)
        else:
            self._optimizer = optimizer

        if loss is not None:
            # Resolve string loss names to callables
            loss_name_map = {
                "mse": torch.nn.MSELoss(),
                "mae": torch.nn.L1Loss(),
                "crossentropy": torch.nn.CrossEntropyLoss(),
                "bce": torch.nn.BCELoss(),
            }
            if isinstance(loss, str):
                key = loss.lower().replace("_", "").replace("-", "")
                if key not in loss_name_map:
                    raise ValueError(
                        f"Unknown loss string '{loss}'. "
                        f"Choose from: {list(loss_name_map)}"
                    )
                loss_fn = loss_name_map[key]
                self._loss_fns = {p: loss_fn for p in self._model.probes}
            elif callable(loss) and not isinstance(loss, dict):
                self._loss_fns = {p: loss for p in self._model.probes}
            else:
                # dict mapping probe → loss fn or string
                resolved = {}
                for probe, fn in dict(loss).items():
                    if isinstance(fn, str):
                        key = fn.lower().replace("_", "").replace("-", "")
                        resolved[probe] = loss_name_map.get(key, torch.nn.MSELoss())
                    else:
                        resolved[probe] = fn
                self._loss_fns = resolved
        self._loss_weights = loss_weights

    def fit(
        self,
        x: Optional[Dict] = None,
        y: Optional[Dict] = None,
        n_steps: int = 1,
        epochs: int = 1,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        validation_split: float = 0.0,
        stateful: bool = False,
        verbose: int = 1,
    ) -> Dict:
        """Train the network parameters.

        Parameters
        ----------
        x : dict
            Input data: maps ``nengo.Node`` to array of shape
            ``(n_samples, n_steps, node_size)`` or
            ``(n_steps, node_size)`` for a single sample.
        y : dict
            Target data: maps ``nengo.Probe`` to array of shape
            ``(n_samples, n_steps, probe_size)``.
        n_steps : int
            Number of simulation timesteps per sample.
        epochs : int
            Number of passes through the training data.
        batch_size : int, optional
            Batch size (defaults to ``self.minibatch_size``).
        shuffle : bool
            Shuffle data between epochs.
        validation_split : float
            Fraction of data to use for validation.
        stateful : bool
            Preserve state across batches within an epoch.
        verbose : int
            Verbosity (0 = silent, 1 = epoch summary, 2 = batch-level).

        Returns
        -------
        dict
            Training history: ``{'loss': [...], 'val_loss': [...]}``.
        """
        self._ensure_open()
        if self._optimizer is None:
            raise RuntimeError(
                "No optimizer set. Call sim.compile(optimizer=...) first."
            )

        bs = batch_size if batch_size is not None else self.minibatch_size
        x = _normalize_data_dict(x, n_steps=n_steps)
        y = _normalize_data_dict(y, n_steps=n_steps)

        # Determine sample count
        n_samples = _count_samples(x, y)
        if n_samples < bs or n_samples % bs != 0:
            raise ValueError(
                f"Sample count ({n_samples}) must be divisible by batch_size ({bs})."
            )

        # Validation split
        val_idx = int(n_samples * (1 - validation_split))
        train_x, val_x = _split_data(x, val_idx)
        train_y, val_y = _split_data(y, val_idx)
        n_train = val_idx
        n_val = n_samples - val_idx

        history = {"loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(n_train) if shuffle else np.arange(n_train)

            epoch_losses = []
            for start in range(0, n_train, bs):
                end = min(start + bs, n_train)
                idx = perm[start:end]
                cur_bs = len(idx)

                batch_x = _index_data(train_x, idx)
                batch_y = _index_data(train_y, idx)

                if not stateful:
                    self.tensor_graph.reset_state()

                self._optimizer.zero_grad()

                # Forward pass (training mode)
                results = self.tensor_graph.forward(
                    n_steps=n_steps,
                    input_data=batch_x,
                    training=True,
                )

                # Compute loss
                total_loss = self._compute_loss(results, batch_y, cur_bs)
                if total_loss is not None:
                    total_loss.backward()
                    self._optimizer.step()
                    epoch_losses.append(total_loss.item())

            mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
            history["loss"].append(mean_loss)

            # Validation
            if n_val > 0 and val_y:
                val_loss = self._compute_val_loss(val_x, val_y, n_steps, bs)
                history["val_loss"].append(val_loss)
                if verbose >= 1:
                    print(
                        f"Epoch {epoch + 1}/{epochs} — "
                        f"loss: {mean_loss:.4f} — val_loss: {val_loss:.4f}"
                    )
            elif verbose >= 1:
                print(f"Epoch {epoch + 1}/{epochs} — loss: {mean_loss:.4f}")

        return history

    def _compute_loss(self, results, targets, batch_size):
        """Compute total training loss."""
        if not self._loss_fns:
            return None

        total = torch.tensor(0.0, device=self.tensor_graph.device)
        for probe, loss_fn in self._loss_fns.items():
            if probe not in results:
                continue
            pred = results[probe]  # (batch, n_steps, *shape)
            target = targets.get(probe) if targets else None
            if target is None:
                continue
            if not isinstance(target, torch.Tensor):
                target = _to_tensor(target, self.tensor_graph.dtype, self.tensor_graph.device)
            target = _align_target_to_pred(target, pred)
            weight = 1.0
            if self._loss_weights:
                weight = self._loss_weights.get(probe, 1.0)
            loss = loss_fn(pred, target) * weight
            total = total + loss

        return total

    def _compute_val_loss(
        self,
        val_x,
        val_y,
        n_steps,
        bs,
        inference_mode: str = "spiking",
    ):
        """Compute validation loss without gradients."""
        losses = []
        val_x = _normalize_data_dict(val_x, n_steps=n_steps)
        val_y = _normalize_data_dict(val_y, n_steps=n_steps)
        n_val = _count_samples(val_x, val_y)
        rate_mode = _inference_mode_to_rate(inference_mode)
        if n_val > 0 and (n_val < bs or n_val % bs != 0):
            raise ValueError(
                f"Validation sample count ({n_val}) must be divisible by batch_size ({bs})."
            )
        with torch.no_grad():
            for start in range(0, n_val, bs):
                end = min(start + bs, n_val)
                idx = np.arange(start, end)
                batch_x = _index_data(val_x, idx)
                batch_y = _index_data(val_y, idx)
                self.tensor_graph.reset_state()
                results = self.tensor_graph.forward(
                    n_steps,
                    batch_x,
                    training=False,
                    rate_mode=rate_mode,
                )
                loss = self._compute_loss(results, batch_y, len(idx))
                if loss is not None:
                    losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    def evaluate(
        self,
        x: Optional[Dict] = None,
        y: Optional[Dict] = None,
        n_steps: int = 1,
        batch_size: Optional[int] = None,
        inference_mode: str = "spiking",
    ) -> Dict:
        """Evaluate the model on test data.

        Returns
        -------
        dict
            ``{'loss': float}``
        """
        self._ensure_open()
        bs = batch_size if batch_size is not None else self.minibatch_size
        val_loss = self._compute_val_loss(
            x,
            y,
            n_steps,
            bs,
            inference_mode=inference_mode,
        )
        return {"loss": val_loss}

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return all trainable parameters as a dict of numpy arrays."""
        self._ensure_open()
        return self.tensor_graph.get_weights()

    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set trainable parameters from a dict of numpy arrays."""
        self._ensure_open()
        self.tensor_graph.set_weights(weights)

    def reset_state(self):
        """Reset all time-varying state signals to initial values."""
        self._ensure_open()
        self.tensor_graph.reset_state()
        self._n_steps = 0
        self._last_n_steps = 0

    def trainable_params(self) -> List[torch.nn.Parameter]:
        """Return list of trainable parameters for use with an optimizer.

        Includes both Nengo signal parameters (weights/biases/encoders) and
        any PyTorch nn.Module parameters added via ``Layer`` / ``TorchNode``.
        """
        self._ensure_open()
        params = list(self.tensor_graph.parameters())  # uses nn.Module.parameters()
        if not params:
            # Fallback to explicit collections
            params = (
                list(self.tensor_graph._param_dict.parameters()) +
                list(self.tensor_graph._torch_modules.parameters())
            )
        return params

    def get_nengo_params(
        self,
        nengo_objects=None,
        include_trainable: bool = True,
        include_non_trainable: bool = False,
    ) -> Dict:
        """Extract current parameter values from the simulation.

        Parameters
        ----------
        nengo_objects : list, optional
            Objects to extract parameters for. If None, returns all.

        Returns
        -------
        dict
            Maps objects to dicts of parameter values.
        """
        self._ensure_open()
        model = self._model
        result = {}

        objects = nengo_objects
        if objects is None:
            objects = (
                list(self.network.all_ensembles) +
                list(self.network.all_connections) +
                list(self.network.all_nodes)
            )

        for obj in objects:
            obj_params = {}
            if isinstance(obj, nengo.Ensemble):
                ens_params = model.params.get(obj)
                if ens_params is not None:
                    for key in ("gain", "bias", "encoders", "scaled_encoders"):
                        val = getattr(ens_params, key, None)
                        if val is not None:
                            obj_params[key] = np.asarray(val)

                ens_sigs = model.sig.get(obj, {})
                if "encoders" in ens_sigs:
                    try:
                        current = self.tensor_graph.signals.gather(ens_sigs["encoders"])
                        obj_params["scaled_encoders"] = current.detach().cpu().numpy()
                    except Exception:
                        pass

                neuron_sigs = model.sig.get(obj.neurons, {})
                if "bias" in neuron_sigs:
                    try:
                        current = self.tensor_graph.signals.gather(neuron_sigs["bias"])
                        obj_params["bias"] = current.detach().cpu().numpy()
                    except Exception:
                        pass
            else:
                sigs = model.sig.get(obj, {})
                for key, sig in sigs.items():
                    try:
                        val = self.tensor_graph.signals.gather(sig)
                        obj_params[key] = val.detach().cpu().numpy()
                    except Exception:
                        pass
            if obj_params:
                result[obj] = obj_params

        return result

    def freeze_params(self, nengo_objects=None):
        """Copy trained parameter values back to the Nengo objects.

        This allows you to save the trained model as a standard Nengo
        network that can be run without nengo-dl.

        Parameters
        ----------
        nengo_objects : list, optional
            Objects to freeze. If None, freezes all.
        """
        self._ensure_open()
        params = self.get_nengo_params(nengo_objects)
        model = self._model

        for obj, obj_params in params.items():
            if isinstance(obj, nengo.Ensemble):
                ens_params = model.params.get(obj)
                gain = obj_params.get("gain")
                bias = obj_params.get("bias")
                scaled = obj_params.get("scaled_encoders")
                encoders = obj_params.get("encoders")

                if gain is not None:
                    gain = np.asarray(gain)
                    obj.gain = gain
                    if ens_params is not None:
                        try:
                            ens_params.gain = gain
                        except Exception:
                            pass

                if bias is not None:
                    bias = np.asarray(bias)
                    obj.bias = bias
                    if ens_params is not None:
                        try:
                            ens_params.bias = bias
                        except Exception:
                            pass

                if scaled is not None:
                    scaled = np.asarray(scaled)
                    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
                    safe_norms = np.where(norms > 1e-12, norms, 1.0)
                    encoders = scaled / safe_norms
                    derived_gain = (safe_norms[:, 0] * obj.radius).astype(scaled.dtype, copy=False)

                    if ens_params is not None and getattr(ens_params, "encoders", None) is not None:
                        existing = np.asarray(ens_params.encoders)
                        zero_mask = norms[:, 0] <= 1e-12
                        if np.any(zero_mask):
                            encoders[zero_mask] = existing[zero_mask]

                    gain = derived_gain
                    obj.gain = gain
                    if ens_params is not None:
                        try:
                            ens_params.gain = gain
                        except Exception:
                            pass

                if encoders is not None:
                    encoders = np.asarray(encoders)
                    obj.encoders = encoders
                    if ens_params is not None:
                        try:
                            ens_params.encoders = encoders
                        except Exception:
                            pass
                        if gain is not None:
                            try:
                                ens_params.scaled_encoders = encoders * (
                                    np.asarray(gain)[:, None] / obj.radius
                                )
                            except Exception:
                                pass
            elif isinstance(obj, nengo.Connection):
                conn_params = model.params.get(obj)
                if conn_params is not None and "weights" in obj_params:
                    weights = np.asarray(obj_params["weights"])
                    try:
                        conn_params.weights = weights
                    except Exception:
                        pass
                    try:
                        solver_uses_weights = (
                            isinstance(obj.pre_obj, nengo.ensemble.Neurons)
                            or isinstance(obj.post_obj, nengo.ensemble.Neurons)
                        )
                        solver_values = weights if solver_uses_weights else weights.T
                        obj.solver = nengo.solvers.NoSolver(
                            solver_values,
                            weights=solver_uses_weights,
                        )
                    except Exception:
                        pass

    def save_params(self, path: str):
        """Save all trainable parameters to a file.

        Parameters
        ----------
        path : str
            File path. If the path does not end in ``.npz``, that extension
            is added automatically (matching ``np.savez`` behaviour).
        """
        self._ensure_open()
        # Ensure consistent extension so load_params can find the file
        if not path.endswith(".npz"):
            path = path + ".npz"
        weights = self.tensor_graph.get_weights()
        np.savez(path, **weights)

    def load_params(self, path: str):
        """Load parameters from a file saved by ``save_params``.

        Parameters
        ----------
        path : str
            Path to the saved parameter file. The ``.npz`` extension may be
            omitted; it is added automatically if the bare path is not found.
        """
        self._ensure_open()
        import os
        if not os.path.exists(path) and not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path, allow_pickle=False)
        self.tensor_graph.set_weights(dict(data), strict=False)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def check_gradients(
        self,
        x: Optional[Dict] = None,
        n_steps: int = 1,
        atol: float = 1e-4,
    ):
        """Check gradients using PyTorch's gradcheck (for debugging).

        Returns True if gradients are correct.
        """
        self._ensure_open()
        import torch.autograd
        params = self.trainable_params()
        if not params:
            warnings.warn("No trainable parameters found.")
            return True

        # Run one forward/backward pass and check for NaN/Inf gradients
        self.tensor_graph.reset_state()
        results = self.tensor_graph.forward(n_steps, x, training=True)

        # Sum all outputs as a dummy loss
        total = sum(v.sum() for v in results.values())
        total.backward()

        for i, p in enumerate(params):
            if p.grad is None:
                warnings.warn(f"Parameter {i} has no gradient.")
            elif torch.isnan(p.grad).any():
                warnings.warn(f"NaN gradient in parameter {i}.")
            elif torch.isinf(p.grad).any():
                warnings.warn(f"Inf gradient in parameter {i}.")

        return True

    @property
    def n_steps(self) -> int:
        """Total number of simulation steps completed."""
        return self._n_steps

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self._n_steps * self.dt

    def trange(self, n_steps: Optional[int] = None, dt: Optional[float] = None):
        """Return timestep values matching the last simulation run.

        Parameters
        ----------
        n_steps : int, optional
            Number of steps (defaults to last run length).
        dt : float, optional
            Timestep (defaults to simulator dt).

        Returns
        -------
        numpy.ndarray
            Array of time values in seconds.
        """
        steps = n_steps if n_steps is not None else self._last_n_steps
        timestep = dt if dt is not None else self.dt
        return np.arange(1, steps + 1) * timestep

    def reset(self, seed: Optional[int] = None):
        """Reset simulation state and optionally set a new seed.

        Parameters
        ----------
        seed : int, optional
            New random seed (also re-initialises parameters).
        """
        self.tensor_graph.reset_state()
        self._n_steps = 0
        self._last_n_steps = 0
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)
            np.random.seed(seed)

    def __repr__(self):
        return (
            f"Simulator(network={self.network}, dt={self.dt}, "
            f"minibatch_size={self.minibatch_size}, "
            f"device={self.tensor_graph.device})"
        )


# ---------------------------------------------------------------------------
# Helper functions for data management
# ---------------------------------------------------------------------------

def _inference_mode_to_rate(inference_mode):
    """Return whether a public inference mode should use rate neurons."""
    if isinstance(inference_mode, bool):
        return inference_mode

    mode = str(inference_mode).lower().replace("_", "-")
    if mode in {"spiking", "spike", "spikes"}:
        return False
    if mode in {"rate", "rates", "rate-based", "ratebased"}:
        return True

    raise ValueError(
        "inference_mode must be 'spiking' or 'rate', "
        f"got {inference_mode!r}"
    )

def _count_samples(x, y):
    """Count number of samples from input/target dicts."""
    counts = []
    for data in (x, y):
        if not data:
            continue
        for v in data.values():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                counts.append(v.shape[0])
        if counts and len(set(counts)) != 1:
            raise ValueError(f"Inconsistent sample counts in data: {sorted(set(counts))}")
    if x and y and counts:
        x_count = next(v.shape[0] for v in x.values() if isinstance(v, (np.ndarray, torch.Tensor)))
        y_count = next(v.shape[0] for v in y.values() if isinstance(v, (np.ndarray, torch.Tensor)))
        if x_count != y_count:
            raise ValueError(f"Input and target sample counts differ: {x_count} vs {y_count}")
    return counts[0] if counts else 0


def _normalize_data_dict(data, n_steps=None):
    """Canonicalize arrays/tensors to ``(samples, n_steps, ...)`` form."""
    if not data:
        return data

    normalized = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            arr = v
            if arr.ndim == 1:
                arr = arr.reshape(1, 1, *arr.shape)
            elif arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            normalized[k] = arr
        elif isinstance(v, torch.Tensor):
            arr = v
            if arr.dim() == 1:
                arr = arr.reshape(1, 1, *arr.shape)
            elif arr.dim() == 2:
                arr = arr.unsqueeze(0)
            normalized[k] = arr
        else:
            normalized[k] = v

    return normalized


def _split_data(data, idx):
    """Split data dict at index idx."""
    if not data:
        return data, data
    a = {k: v[:idx] for k, v in data.items() if isinstance(v, (np.ndarray, torch.Tensor))}
    b = {k: v[idx:] for k, v in data.items() if isinstance(v, (np.ndarray, torch.Tensor))}
    return (a if a else None), (b if b else None)


def _index_data(data, idx):
    """Index data dict by a list of sample indices."""
    if not data:
        return data
    return {
        k: v[idx] if isinstance(v, (np.ndarray, torch.Tensor)) else v
        for k, v in data.items()
    }


def _align_target_to_pred(target, pred):
    """Expand singleton batch/time dimensions in targets to match predictions."""
    while target.dim() < pred.dim():
        target = target.unsqueeze(0)

    if target.shape[0] == 1 and pred.shape[0] != 1:
        expand_shape = list(target.shape)
        expand_shape[0] = pred.shape[0]
        target = target.expand(*expand_shape)

    if target.shape[1] == 1 and pred.shape[1] != 1:
        expand_shape = list(target.shape)
        expand_shape[1] = pred.shape[1]
        target = target.expand(*expand_shape)

    if target.shape != pred.shape:
        raise ValueError(
            f"Target shape {tuple(target.shape)} is incompatible with prediction "
            f"shape {tuple(pred.shape)}."
        )

    return target
