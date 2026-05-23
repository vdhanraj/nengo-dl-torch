"""Tests for the nengo-dl training / evaluation API (PyTorch backend).

This replaces the TF/Keras-specific ``test_keras.py`` from the original
nengo-dl with tests targeting the PyTorch ``sim.compile`` / ``sim.fit`` /
``sim.evaluate`` interface.
"""

import numpy as np
import pytest
import torch
import nengo
import nengo_dl
from nengo_dl.tests import dummies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trainable_net(seed=0, n_hidden=20):
    """Single-hidden-layer ReLU network for training tests."""
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(np.zeros(1))
        ens = nengo.Ensemble(
            n_hidden, 1, neuron_type=nengo.RectifiedLinear(), seed=seed
        )
        nengo.Connection(inp, ens, synapse=None)
        out = nengo.Node(size_in=1)
        nengo.Connection(ens, out, function=lambda x: x, synapse=None)
        p = nengo.Probe(out, synapse=None)
    return net, inp, p


def _xy(n=32, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, (n, 1, 1)).astype(np.float32)
    y = np.zeros_like(x)
    return x, y


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------

class TestCompile:
    def test_compile_mse_string(self):
        net, inp, p = _trainable_net()
        with nengo_dl.Simulator(net, minibatch_size=4, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})

    def test_compile_mae_string(self):
        net, inp, p = _trainable_net()
        with nengo_dl.Simulator(net, minibatch_size=4, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mae"})

    def test_compile_callable(self):
        net, inp, p = _trainable_net()
        loss_fn = lambda y_pred, y_true: torch.mean((y_pred - y_true) ** 2)
        with nengo_dl.Simulator(net, minibatch_size=4, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: loss_fn})

    def test_compile_sgd_optimizer(self):
        net, inp, p = _trainable_net()
        with nengo_dl.Simulator(net, minibatch_size=4, seed=0) as sim:
            sim.compile(
                optimizer=torch.optim.SGD(
                    [p for p in sim.tensor_graph.parameters()], lr=1e-3
                ),
                loss={p: "mse"},
            )

    def test_compile_multiple_probes(self):
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
# fit
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_reduces_loss(self):
        net, inp, p = _trainable_net()
        x, y = _xy(n=64)

        with nengo_dl.Simulator(net, minibatch_size=16, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            hist = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=15)

        assert hist["loss"][-1] < hist["loss"][0]

    def test_fit_returns_history_dict(self):
        net, inp, p = _trainable_net()
        x, y = _xy(n=16)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            hist = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=3)

        assert isinstance(hist, dict)
        assert "loss" in hist
        assert len(hist["loss"]) == 3

    def test_fit_loss_finite(self):
        net, inp, p = _trainable_net()
        x, y = _xy(n=16)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            hist = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=2)

        assert all(np.isfinite(v) for v in hist["loss"])

    def test_fit_changes_weights(self):
        net, inp, p = _trainable_net()
        x, y = _xy(n=32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            w_before = {k: v.copy() for k, v in sim.get_weights().items()}
            sim.compile(optimizer="adam", loss={p: "mse"})
            sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=10)
            w_after = sim.get_weights()

        changed = any(
            not np.allclose(w_before[k], w_after[k]) for k in w_before
        )
        assert changed

    def test_fit_n_steps_gt_1(self):
        """fit() with n_steps > 1 (multi-step unroll)."""
        net, inp, p = _trainable_net()
        x = np.zeros((16, 3, 1), dtype=np.float32)
        y = np.zeros((16, 3, 1), dtype=np.float32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            hist = sim.fit(x={inp: x}, y={p: y}, n_steps=3, epochs=2)

        assert "loss" in hist

    def test_fit_two_input_nodes(self):
        """fit() with two separate input nodes."""
        with nengo.Network(seed=0) as net:
            inp_a = nengo.Node(np.zeros(1))
            inp_b = nengo.Node(np.zeros(1))
            combined = nengo.Node(size_in=2)
            nengo.Connection(inp_a, combined[0], synapse=None)
            nengo.Connection(inp_b, combined[1], synapse=None)
            ens = nengo.Ensemble(10, 2, neuron_type=nengo.RectifiedLinear(), seed=0)
            nengo.Connection(combined, ens, synapse=None)
            out = nengo.Node(size_in=2)
            nengo.Connection(ens, out, function=lambda x: x, synapse=None)
            p = nengo.Probe(out, synapse=None)

        n = 16
        xa = np.zeros((n, 1, 1), dtype=np.float32)
        xb = np.zeros((n, 1, 1), dtype=np.float32)
        y = np.zeros((n, 1, 2), dtype=np.float32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            hist = sim.fit(x={inp_a: xa, inp_b: xb}, y={p: y}, n_steps=1, epochs=2)

        assert "loss" in hist


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_returns_dict(self):
        net, inp, p = _trainable_net()
        x, y = _xy(n=16)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            result = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)

        assert isinstance(result, dict)
        assert "loss" in result

    def test_evaluate_loss_finite(self):
        net, inp, p = _trainable_net()
        x, y = _xy(n=16)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            result = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)

        assert np.isfinite(result["loss"])

    def test_evaluate_after_fit(self):
        """Evaluation loss after training should be lower than before."""
        net, inp, p = _trainable_net()
        x, y = _xy(n=32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            before = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)["loss"]
            sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=10)
            after = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)["loss"]

        assert after <= before + 0.1  # allow small tolerance

    def test_evaluate_zero_target_near_zero(self):
        """If input and target are both zero, loss should be nearly 0."""
        net, inp, p = _trainable_net()
        x = np.zeros((16, 1, 1), dtype=np.float32)
        y = np.zeros((16, 1, 1), dtype=np.float32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            result = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)

        assert result["loss"] < 1.0


# ---------------------------------------------------------------------------
# save / load parameters
# ---------------------------------------------------------------------------

class TestSaveLoadParams:
    def test_get_set_weights_roundtrip(self):
        """set_weights(get_weights()) should leave weights unchanged."""
        net, inp, p = _trainable_net()

        with nengo_dl.Simulator(net, seed=0) as sim:
            w = sim.get_weights()
            sim.set_weights(w)
            w2 = sim.get_weights()

        for k in w:
            np.testing.assert_allclose(w[k], w2[k])

    def test_weights_change_after_training(self):
        """Weights after fit should differ from initial weights."""
        net, inp, p = _trainable_net()
        x, y = _xy(n=32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            w0 = {k: v.copy() for k, v in sim.get_weights().items()}
            sim.compile(optimizer="adam", loss={p: "mse"})
            sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=10)
            w1 = sim.get_weights()

        assert any(not np.allclose(w0[k], w1[k]) for k in w0)

    def test_transfer_weights_between_simulators(self, tmp_path):
        """Weights from one simulator can be loaded into another."""
        net, inp, p = _trainable_net()
        x, y = _xy(n=16)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim1:
            sim1.compile(optimizer="adam", loss={p: "mse"})
            sim1.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=5)
            w_trained = sim1.get_weights()

        with nengo_dl.Simulator(net, minibatch_size=8, seed=99) as sim2:
            w_initial = sim2.get_weights()
            # verify different initial weights
            assert any(not np.allclose(w_initial[k], w_trained[k]) for k in w_initial)
            sim2.set_weights(w_trained)
            w_loaded = sim2.get_weights()

        for k in w_trained:
            np.testing.assert_allclose(w_trained[k], w_loaded[k])


# ---------------------------------------------------------------------------
# linear_net from dummies
# ---------------------------------------------------------------------------

class TestLinearNet:
    def test_linear_net_runs(self):
        net, a, p = dummies.linear_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(3)
        np.testing.assert_allclose(sim.data[p][:, 0], 1.0, atol=1e-5)

    def test_trainable_net_fit(self):
        """A network with trainable params should fit without error."""
        net, inp, p = _trainable_net()
        y = np.zeros((8, 1, 1), dtype=np.float32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            hist = sim.fit(
                x={inp: np.zeros((8, 1, 1), dtype=np.float32)},
                y={p: y},
                n_steps=1,
                epochs=5,
            )
        assert "loss" in hist
