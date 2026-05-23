"""nengo-dl: PyTorch backend for Nengo deep learning.

A modern reimplementation of NengoDL using PyTorch instead of TensorFlow.
Provides the same high-level API for training and running Nengo networks
with gradient-based optimization.

Key classes
-----------
Simulator
    Run and train Nengo networks with PyTorch.
TorchNode
    Embed arbitrary PyTorch modules inside a Nengo network.
Converter
    Convert PyTorch models to Nengo spiking networks.

Key functions
-------------
configure_settings
    Set simulation/training hyperparameters on a Nengo Network.

Custom neuron types
-------------------
SoftLIFRate
    Differentiable LIF rate approximation (ideal for training).
SpikingLeakyReLU
    Spiking variant of the leaky ReLU neuron.
LeakyReLU
    Rate-coded leaky ReLU neuron.
"""

from .version import version as __version__

from .simulator import Simulator
from .tensor_node import TorchNode, Layer
from .converter import Converter
from .config import configure_settings, get_setting
from .neurons import SoftLIFRate, SpikingLeakyReLU, LeakyReLU
from . import losses

# Convenience re-export so users can write ``nengo_dl.Builder``
from .builder import Builder, OpBuilder, BuildConfig

# Signal dict exposed for advanced use
from .signals import SignalDict

# TensorGraph for direct access
from .tensor_graph import TensorGraph

__all__ = [
    "__version__",
    # Core
    "Simulator",
    # Nodes
    "TorchNode",
    "Layer",
    # Conversion
    "Converter",
    # Config
    "configure_settings",
    "get_setting",
    # Neurons
    "SoftLIFRate",
    "SpikingLeakyReLU",
    "LeakyReLU",
    # Losses
    "losses",
    # Advanced
    "Builder",
    "OpBuilder",
    "BuildConfig",
    "SignalDict",
    "TensorGraph",
]
