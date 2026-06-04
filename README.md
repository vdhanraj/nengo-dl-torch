# nengo-dl (PyTorch Backend)

A PyTorch-based reimplementation of [NengoDL](https://www.nengo.ai/nengo-dl/), the deep-learning extension for the [Nengo](https://www.nengo.ai/) neural simulator. The original NengoDL uses TensorFlow as its backend; this project replaces that with PyTorch while preserving the same high-level API for building, training, and simulating spiking and rate-mode neural networks.

## What is NengoDL?

[Nengo](https://www.nengo.ai/) is a Python library for building and simulating large-scale neural models. [NengoDL](https://www.nengo.ai/nengo-dl/) extends it with deep-learning capabilities — GPU acceleration, automatic differentiation, and the ability to convert deep-learning models (originally Keras/TensorFlow, here PyTorch) directly into Nengo networks. This lets you train spiking neural networks using standard gradient-based optimizers, convert pre-trained PyTorch models to spiking equivalents, and run hybrid rate/spiking simulations.

**Reference links:**
- NengoDL documentation: https://www.nengo.ai/nengo-dl/
- NengoDL GitHub (original TensorFlow version): https://github.com/nengo/nengo-dl
- Nengo documentation: https://www.nengo.ai/nengo/
- Nengo GitHub: https://github.com/nengo/nengo

---

## Requirements

- Python 3.14.4 (tested; ≥ 3.8 should work)
- See [`requirements.txt`](requirements.txt) for the full pinned dependency list.

---

## Installation

### Option 1 — Conda environment (recommended)

```bash
# Create and activate a new environment
conda create -n nengo_dl python=3.14.4
conda activate nengo_dl

# Install all dependencies
pip install -r requirements.txt

# Install nengo-dl itself in editable mode
pip install -e .
```

### Option 2 — pip only (no conda)

```bash
pip install -r requirements.txt
pip install -e .
```

> If you are on a GPU machine, `torch` will pull in the matching CUDA libraries automatically. The `nvidia-*` and `cuda-*` lines in `requirements.txt` pin the exact versions used during development; you can omit them and let pip resolve them from the torch wheel.

### Option 3 — Virtual environment (venv)

```bash
# Create and activate the virtual environment
python3.14 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies and the package
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

To deactivate the environment when you are done:

```bash
deactivate
```

---

## Running the tests

```bash
pytest nengo_dl/tests/
```

To run a specific test file:

```bash
pytest nengo_dl/tests/test_simulator.py -v
```

---

## Project structure

```
nengo_dl/
├── simulator.py        # Main Simulator class (mirrors nengo_dl.Simulator API)
├── tensor_graph.py     # TensorGraph: PyTorch nn.Module wrapping a built Nengo model
├── tensor_node.py      # TorchNode: wraps an nn.Module as a Nengo Node
├── op_builders.py      # PyTorch implementations of Nengo operators
├── graph_optimizer.py  # Operator dependency graph and topological sort
├── converter.py        # Converter: PyTorch model → Nengo network
├── config.py           # configure_settings() and build-time configuration
├── neurons.py          # Differentiable neuron implementations
├── processes.py        # Synaptic filter (Lowpass, Alpha) implementations
└── tests/              # Full test suite
docs/
└── examples/           # Jupyter notebook examples
```

---

## Quick example

```python
import nengo
import nengo_dl
import numpy as np

with nengo.Network() as net:
    inp = nengo.Node(np.array([1.0, 0.5]))
    ens = nengo.Ensemble(50, dimensions=2,
                         neuron_type=nengo.RectifiedLinear())
    nengo.Connection(inp, ens)
    p = nengo.Probe(ens.neurons, synapse=0.005)

with nengo_dl.Simulator(net, minibatch_size=32) as sim:
    sim.run_steps(100)
    output = sim.data[p]   # shape: (32, 100, 50)
```

### Converting a PyTorch model to a spiking network

```python
import torch.nn as nn
import nengo_dl

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

converter = nengo_dl.Converter(model, scale_firing_rates=500,
                               activation_type="spiking_relu")

with nengo_dl.Simulator(converter.net) as sim:
    sim.run_steps(30, inference_mode="spiking")
```
