"""Utility functions for nengo-dl (PyTorch backend)."""

import contextlib
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

class ProgressBar:
    """Simple text-based progress bar."""

    def __init__(self, total: int, label: str = "", width: int = 40):
        self.total = total
        self.label = label
        self.width = width
        self._start = time.time()
        self._n = 0

    def update(self, n: int = 1):
        self._n += n
        frac = self._n / max(self.total, 1)
        filled = int(self.width * frac)
        bar = "=" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self._start
        print(
            f"\r{self.label} [{bar}] {self._n}/{self.total} ({elapsed:.1f}s)",
            end="",
            flush=True,
        )
        if self._n >= self.total:
            print()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self._n < self.total:
            print()


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def to_numpy(x) -> np.ndarray:
    """Convert a tensor or array-like to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_tensor(x, dtype=torch.float32, device=None) -> torch.Tensor:
    """Convert array-like or tensor to a torch.Tensor."""
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(np.asarray(x, dtype=np.float32), dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t.to(dtype)


# ---------------------------------------------------------------------------
# Model inspection
# ---------------------------------------------------------------------------

def layer_count_params(module: nn.Module) -> int:
    """Return the total number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def print_model_summary(tensor_graph):
    """Print a summary of TensorGraph parameters."""
    total = 0
    print("=" * 60)
    print("nengo-dl TensorGraph Parameter Summary")
    print("=" * 60)
    for name, param in tensor_graph._param_dict.items():
        n = param.numel()
        total += n
        print(f"  {name[:50]:50s}  {str(param.shape):20s}  {n:>10,d}")
    print("-" * 60)
    print(f"  {'Total':50s}  {'':20s}  {total:>10,d}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Spike encoding helpers
# ---------------------------------------------------------------------------

def rate_to_spikes(rates: np.ndarray, dt: float, seed: int = 0) -> np.ndarray:
    """Convert rate-coded values to binary spike trains.

    Parameters
    ----------
    rates : ndarray
        Firing rates in Hz, shape ``(n_steps, n_neurons)`` or
        ``(batch, n_steps, n_neurons)``.
    dt : float
        Timestep in seconds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ndarray
        Binary spike array of the same shape as ``rates``.
    """
    rng = np.random.default_rng(seed)
    probs = np.clip(rates * dt, 0, 1)
    return (rng.uniform(size=probs.shape) < probs).astype(np.float32)


def decode_spikes(
    spikes: np.ndarray,
    dt: float,
    tau: float = 0.01,
) -> np.ndarray:
    """Low-pass filter a spike train to estimate firing rate.

    Parameters
    ----------
    spikes : ndarray
        Spike train, shape ``(n_steps, n_neurons)``.
    dt : float
        Timestep in seconds.
    tau : float
        Filter time constant in seconds.

    Returns
    -------
    ndarray
        Estimated firing rates.
    """
    alpha = np.exp(-dt / tau)
    output = np.zeros_like(spikes)
    y = np.zeros(spikes.shape[1:] if spikes.ndim > 1 else (1,))
    for t in range(spikes.shape[0]):
        y = alpha * y + (1 - alpha) * spikes[t]
        output[t] = y
    return output


# ---------------------------------------------------------------------------
# Signal name sanitisation
# ---------------------------------------------------------------------------

def sanitize_name(name: str, max_len: int = 64) -> str:
    """Make a Nengo signal name safe for use as a Python identifier."""
    import re
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s[:max_len]


# ---------------------------------------------------------------------------
# Batch-first / time-first reshaping helpers
# ---------------------------------------------------------------------------

def batch_first(x: np.ndarray) -> np.ndarray:
    """Convert (n_steps, batch, size) → (batch, n_steps, size)."""
    return np.transpose(x, (1, 0) + tuple(range(2, x.ndim)))


def time_first(x: np.ndarray) -> np.ndarray:
    """Convert (batch, n_steps, size) → (n_steps, batch, size)."""
    return np.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
