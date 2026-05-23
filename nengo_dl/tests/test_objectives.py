"""Tests for objective / loss functions used in nengo-dl training.

These complement test_losses.py by focusing on how objectives compose with
sim.compile() and sim.fit() rather than testing the loss modules in isolation.
"""

import numpy as np
import pytest
import torch
import nengo
import nengo_dl
from nengo_dl.losses import MSE as MSELoss, ProbeObjective


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trainable_net(seed=0):
    """Minimal trainable network: input → ReLU ensemble → output node."""
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(np.zeros(1))
        ens = nengo.Ensemble(20, 1, neuron_type=nengo.RectifiedLinear(), seed=seed)
        nengo.Connection(inp, ens, synapse=None)
        out = nengo.Node(size_in=1)
        nengo.Connection(ens, out, function=lambda x: x, synapse=None)
        p = nengo.Probe(out, synapse=None)
    return net, inp, p


def _rng_data(n=32, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, (n, 1, 1)).astype(np.float32)
    y = np.zeros_like(x)
    return x, y


# ---------------------------------------------------------------------------
# Compile with different objective specifications
# ---------------------------------------------------------------------------

class TestCompileObjectives:
    def test_compile_string_mse(self):
        net, inp, p = _trainable_net()
        with nengo_dl.Simulator(net, minibatch_size=4, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})

    def test_compile_string_mae(self):
        net, inp, p = _trainable_net()
        with nengo_dl.Simulator(net, minibatch_size=4, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mae"})

    def test_compile_callable_loss(self):
        net, inp, p = _trainable_net()
        def my_loss(y_pred, y_true):
            return torch.mean((y_pred - y_true) ** 2)

        with nengo_dl.Simulator(net, minibatch_size=4, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: my_loss})

    def test_compile_module_loss(self):
        net, inp, p = _trainable_net()
        with nengo_dl.Simulator(net, minibatch_size=4, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: MSELoss()})

    def test_compile_multiple_probes(self):
        """Multiple probe objectives should all be registered."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            ens1 = nengo.Ensemble(10, 2, neuron_type=nengo.RectifiedLinear(), seed=0)
            ens2 = nengo.Ensemble(10, 2, neuron_type=nengo.RectifiedLinear(), seed=1)
            nengo.Connection(inp, ens1, synapse=None)
            nengo.Connection(ens1, ens2, synapse=None)
            p1 = nengo.Probe(ens1, synapse=None)
            p2 = nengo.Probe(ens2, synapse=None)

        with nengo_dl.Simulator(net, minibatch_size=4, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p1: "mse", p2: "mse"})


# ---------------------------------------------------------------------------
# Loss values are finite and reasonable
# ---------------------------------------------------------------------------

class TestObjectiveValues:
    def test_initial_loss_is_finite(self):
        """Loss before training must be a finite number."""
        net, inp, p = _trainable_net()
        x, y = _rng_data()

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            hist = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=1)

        assert np.isfinite(hist["loss"][0])

    def test_loss_decreases_after_training(self):
        """Training should reduce the MSE loss."""
        net, inp, p = _trainable_net()
        rng = np.random.RandomState(0)
        x = rng.uniform(-1, 1, (64, 1, 1)).astype(np.float32)
        y = np.zeros_like(x)

        with nengo_dl.Simulator(net, minibatch_size=16, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            hist = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=15)

        assert hist["loss"][-1] < hist["loss"][0]

    def test_zero_target_loss_near_zero_after_training(self):
        """Training toward constant zero target should bring loss close to 0."""
        net, inp, p = _trainable_net()
        x = np.zeros((32, 1, 1), dtype=np.float32)
        y = np.zeros_like(x)

        with nengo_dl.Simulator(net, minibatch_size=16, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            hist = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=20)

        assert hist["loss"][-1] < 0.5


# ---------------------------------------------------------------------------
# ProbeObjective wrapper
# ---------------------------------------------------------------------------

class TestProbeObjective:
    def test_probe_objective_wraps_loss(self):
        """ProbeObjective should wrap an arbitrary loss and be callable."""
        obj = ProbeObjective(MSELoss())
        y_pred = torch.zeros(4, 1, 1)
        y_true = torch.zeros(4, 1, 1)
        loss = obj(y_pred, y_true)
        assert torch.isfinite(loss)

    def test_probe_objective_nonzero_for_nonzero_error(self):
        obj = ProbeObjective(MSELoss())
        y_pred = torch.ones(4, 1, 1)
        y_true = torch.zeros(4, 1, 1)
        loss = obj(y_pred, y_true)
        assert loss.item() > 0

    def test_probe_objective_in_compile(self):
        net, inp, p = _trainable_net()
        with nengo_dl.Simulator(net, minibatch_size=4, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: ProbeObjective(MSELoss())})


# ---------------------------------------------------------------------------
# Evaluate after fit
# ---------------------------------------------------------------------------

class TestEvaluateObjective:
    def test_evaluate_returns_scalar_loss(self):
        net, inp, p = _trainable_net()
        x, y = _rng_data(n=16)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=3)
            result = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)

        val_loss = result["loss"]
        assert np.isfinite(val_loss)
        assert val_loss >= 0.0

    def test_evaluate_loss_consistent_with_fit(self):
        """Val loss from evaluate should be close to final training loss."""
        net, inp, p = _trainable_net()
        x, y = _rng_data(n=32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            hist = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=5)
            result = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)

        val_loss = result["loss"]
        # Evaluate and training loss should be in the same ballpark
        assert abs(val_loss - hist["loss"][-1]) < 1.0
