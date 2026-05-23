"""Signal management for nengo-dl (PyTorch backend).

Maps Nengo Signals to PyTorch tensors. Maintains separate stores for
trainable parameters (nn.Parameter) and simulation state (regular tensors).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional
from nengo.builder.signal import Signal


class SignalDict:
    """Manages Nengo Signals as PyTorch tensors during simulation.

    Signals are stored in two categories:
    - ``params``: An ``nn.ParameterDict`` containing trainable parameters
      (weights, biases, encoders). These have shape ``(*sig.shape)`` with
      no batch dimension since they are shared across the minibatch.
    - ``state``: A dict of regular tensors representing time-varying state
      (voltages, refractory times, activations). These have shape
      ``(minibatch_size, *sig.shape)``.

    Parameters
    ----------
    minibatch_size : int
        Number of samples processed in parallel.
    device : torch.device
        Device for all tensors.
    dtype : torch.dtype
        Default float dtype.
    """

    def __init__(self, minibatch_size: int, device: torch.device, dtype=torch.float32):
        self.minibatch_size = minibatch_size
        self.device = device
        self.dtype = dtype
        # Maps id(signal.base) -> signal (the base signal object)
        self._base_signals: Dict[int, Signal] = {}
        # Maps id(signal.base) -> bool (is this a trainable parameter?)
        self._is_param: Dict[int, bool] = {}
        # Maps id(signal.base) -> torch.Tensor (the actual data)
        # For params: shape = sig.shape; for state: shape = (batch, *sig.shape)
        self._data: Dict[int, torch.Tensor] = {}

    def add_signal(self, sig: Signal, trainable: bool = False) -> None:
        """Register a signal and allocate its tensor.

        Parameters
        ----------
        sig : Signal
            The Nengo signal to register. Must be a base signal (not a view).
        trainable : bool
            If True, the signal is stored as an nn.Parameter.
        """
        assert not sig.is_view, f"Only base signals can be added directly: {sig.name}"
        base_id = id(sig)
        if base_id in self._data:
            return  # already registered

        self._base_signals[base_id] = sig
        self._is_param[base_id] = trainable

        # Convert initial value to tensor
        init_val = sig.initial_value
        if not isinstance(init_val, np.ndarray):
            init_val = np.array(init_val)

        # Choose dtype
        if np.issubdtype(init_val.dtype, np.floating):
            t_dtype = self.dtype
        elif np.issubdtype(init_val.dtype, np.integer):
            t_dtype = torch.int64
        else:
            t_dtype = self.dtype

        tensor = torch.tensor(init_val, dtype=t_dtype, device=self.device)

        if trainable:
            # Parameters: no batch dim, wrapped in nn.Parameter
            self._data[base_id] = nn.Parameter(tensor, requires_grad=True)
        else:
            # State: add batch dimension
            self._data[base_id] = tensor.unsqueeze(0).expand(
                self.minibatch_size, *tensor.shape
            ).clone()

    def gather(self, sig: Signal) -> torch.Tensor:
        """Read the current value of a signal.

        Parameters
        ----------
        sig : Signal
            The signal to read.

        Returns
        -------
        torch.Tensor
            For parameters: shape ``(*sig.shape)``, no batch dim.
            For state signals: shape ``(batch, *sig.shape)``.
        """
        base = sig.base
        base_id = id(base)
        data = self._data[base_id]

        if sig.is_view:
            # The view occupies a contiguous range in the base
            start = sig.elemoffset
            end = sig.elemoffset + sig.size
            is_param = self._is_param[base_id]
            if is_param:
                flat = data.reshape(-1)[start:end]
                return flat.reshape(sig.shape)
            else:
                flat = data.reshape(self.minibatch_size, -1)[:, start:end]
                return flat.reshape(self.minibatch_size, *sig.shape)
        else:
            return data

    def scatter(self, sig: Signal, val: torch.Tensor, mode: str = "set") -> None:
        """Write a value to a signal.

        Parameters
        ----------
        sig : Signal
            The signal to write.
        val : torch.Tensor
            The value to write.
        mode : str
            ``'set'`` replaces the current value; ``'inc'`` increments it.
        """
        base = sig.base
        base_id = id(base)
        is_param = self._is_param[base_id]
        current = self._data[base_id]

        # Normalize val to correct batch shape: need (batch, *sig.shape)
        if not is_param:
            expected = (self.minibatch_size,) + tuple(sig.shape)
            if tuple(val.shape) == expected:
                pass  # already correct
            elif tuple(val.shape) == tuple(sig.shape):
                # No batch dim: add it
                val = val.unsqueeze(0).expand(expected)
            elif val.dim() == len(sig.shape) + 1 and val.shape[0] == 1:
                # Batch dim = 1: expand to full batch
                val = val.expand(expected)
            else:
                # Last resort: try reshape first, then expand
                try:
                    val = val.reshape(expected)
                except RuntimeError:
                    try:
                        val = val.expand(expected)
                    except RuntimeError:
                        pass  # let it fail with a clear error below

        if sig.is_view:
            start = sig.elemoffset
            end = sig.elemoffset + sig.size

            if is_param:
                flat_val = val.reshape(-1)
                flat_cur = current.reshape(-1)
                if mode == "set":
                    flat_cur[start:end] = flat_val
                else:
                    flat_cur[start:end] = flat_cur[start:end] + flat_val
                self._data[base_id] = flat_cur.reshape(current.shape)
            else:
                flat_val = val.reshape(self.minibatch_size, -1)
                new_data = current.clone()
                flat_data = new_data.reshape(self.minibatch_size, -1)
                if mode == "set":
                    flat_data[:, start:end] = flat_val
                else:
                    flat_data[:, start:end] = flat_data[:, start:end] + flat_val
                self._data[base_id] = flat_data.reshape(current.shape)
        else:
            if mode == "set":
                if is_param:
                    with torch.no_grad():
                        current.copy_(val)
                else:
                    self._data[base_id] = val
            else:  # inc
                if is_param:
                    with torch.no_grad():
                        current.add_(val)
                else:
                    self._data[base_id] = current + val

    def reset(self) -> None:
        """Reset all state signals to their initial values.

        Parameters (trainable signals) are NOT reset; only time-varying
        state is reset.
        """
        for base_id, sig in self._base_signals.items():
            if not self._is_param[base_id]:
                init_val = sig.initial_value
                if not isinstance(init_val, np.ndarray):
                    init_val = np.array(init_val)

                if np.issubdtype(init_val.dtype, np.floating):
                    t_dtype = self.dtype
                elif np.issubdtype(init_val.dtype, np.integer):
                    t_dtype = torch.int64
                else:
                    t_dtype = self.dtype

                tensor = torch.tensor(init_val, dtype=t_dtype, device=self.device)
                self._data[base_id] = tensor.unsqueeze(0).expand(
                    self.minibatch_size, *tensor.shape
                ).clone()

    def get_parameter(self, sig: Signal) -> Optional[nn.Parameter]:
        """Return the nn.Parameter for a trainable signal, or None."""
        base_id = id(sig.base)
        if self._is_param.get(base_id, False):
            return self._data[base_id]
        return None

    def get_all_parameters(self) -> Dict[str, nn.Parameter]:
        """Return a dict of all trainable parameters keyed by stable index.

        Keys are ``param_0000``, ``param_0001``, … in the order signals were
        added (which is deterministic for a fixed network structure).  This
        avoids embedding Python object ids in the key, making save/load work
        across different Simulator instances built from the same network.
        """
        result = {}
        idx = 0
        for base_id, sig in self._base_signals.items():
            if self._is_param[base_id]:
                result[f"param_{idx:04d}"] = self._data[base_id]
                idx += 1
        return result

    def __contains__(self, sig: Signal) -> bool:
        return id(sig.base) in self._data

    def __repr__(self) -> str:
        n_params = sum(1 for v in self._is_param.values() if v)
        n_state = sum(1 for v in self._is_param.values() if not v)
        return f"SignalDict(params={n_params}, state={n_state}, batch={self.minibatch_size})"
