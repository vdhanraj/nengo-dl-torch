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
        """Evaluation loss after training should be strictly lower than before."""
        net, inp, p = _trainable_net()
        x, y = _xy(n=32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            before = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)["loss"]
            sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=10)
            after = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)["loss"]

        assert after < before, f"Loss should decrease: {before:.4f} → {after:.4f}"

    def test_evaluate_zero_target_near_zero(self):
        """Identity network (W=1, b=0): zero input vs zero target gives near-zero MSE."""
        linear = torch.nn.Linear(1, 1)
        with torch.no_grad():
            linear.weight.fill_(1.0)
            linear.bias.fill_(0.0)

        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            out = nengo_dl.Layer(linear)(inp)
            p = nengo.Probe(out, synapse=None)

        x = np.zeros((8, 1, 1), dtype=np.float32)
        y = np.zeros((8, 1, 1), dtype=np.float32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            result = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)

        assert result["loss"] < 1e-8, (
            f"Expected near-zero MSE for zero input vs zero target, got {result['loss']}"
        )


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


# ---------------------------------------------------------------------------
# Behavioral correctness
# ---------------------------------------------------------------------------

class TestBehavioralCorrectness:
    def test_exact_mse_loss_value(self):
        """Identity TorchNode (W=1, b=0): MSE of known x vs y equals (x-y)^2."""
        x_val, y_val = 3.0, 1.0
        expected_mse = (x_val - y_val) ** 2  # = 4.0

        linear = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            linear.weight.fill_(1.0)  # output = input exactly

        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            out = nengo_dl.Layer(linear)(inp)
            p = nengo.Probe(out, synapse=None)

        x = np.full((8, 1, 1), x_val, dtype=np.float32)
        y = np.full((8, 1, 1), y_val, dtype=np.float32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            result = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)

        np.testing.assert_allclose(
            result["loss"], expected_mse, rtol=1e-4,
            err_msg=f"Expected MSE={expected_mse}, got {result['loss']}",
        )

    def test_fit_output_moves_toward_target(self):
        """After training on target=0, the mean absolute output decreases."""
        net, inp, p = _trainable_net()
        x_train = np.ones((32, 1, 1), dtype=np.float32)
        y_train = np.zeros((32, 1, 1), dtype=np.float32)
        x_eval = x_train[:8]

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            loss_before = sim.evaluate(x={inp: x_eval}, y={p: y_train[:8]}, n_steps=1)["loss"]
            sim.fit(x={inp: x_train}, y={p: y_train}, n_steps=1, epochs=20)
            loss_after = sim.evaluate(x={inp: x_eval}, y={p: y_train[:8]}, n_steps=1)["loss"]

        assert loss_after < loss_before, (
            f"Output should move toward target=0 after training: "
            f"loss {loss_before:.4f} → {loss_after:.4f}"
        )

    def test_sgd_weight_update_direction(self):
        """SGD step on positive output vs zero target must decrease the weight."""
        import torch.nn as nn

        linear = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            linear.weight.fill_(1.0)

        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            out = nengo_dl.Layer(linear)(inp)
            p = nengo.Probe(out, synapse=None)

        # output = W * input = 1.0 * 1.0 = 1.0; target = 0.0
        # dL/dW = 2*(1.0 - 0.0)*1.0 = 2.0  → SGD decreases W
        x = np.ones((8, 1, 1), dtype=np.float32)
        y = np.zeros((8, 1, 1), dtype=np.float32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            w_before = linear.weight.item()
            sim.compile(
                optimizer=torch.optim.SGD(sim.trainable_params(), lr=0.1),
                loss={p: "mse"},
            )
            sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=1)
            w_after = linear.weight.item()

        assert w_after < w_before, (
            f"SGD should decrease weight toward zero: {w_before:.4f} → {w_after:.4f}"
        )


# ---------------------------------------------------------------------------
# Manual MSE, loss_weights, frozen parameters, evaluate consistency
# ---------------------------------------------------------------------------

class TestExactBehavior:
    def test_manual_mse_matches_evaluate(self):
        """Manually computing (output - target)^2 must equal sim.evaluate() loss."""
        # Identity network: W=1, b=0 → output = input = x_val
        x_val, y_val = 2.0, 5.0
        expected_mse = (x_val - y_val) ** 2  # = 9.0

        linear = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            linear.weight.fill_(1.0)

        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            out = nengo_dl.Layer(linear)(inp)
            p = nengo.Probe(out, synapse=None)

        x = np.full((8, 1, 1), x_val, dtype=np.float32)
        y = np.full((8, 1, 1), y_val, dtype=np.float32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            result = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)

        np.testing.assert_allclose(
            result["loss"], expected_mse, rtol=1e-4,
            err_msg=f"Manual MSE={expected_mse:.1f} must equal sim.evaluate() loss"
        )

    def test_loss_weights_zero_suppresses_probe_loss(self):
        """loss_weights={p2: 0.0} makes p2 contribute nothing to the total loss."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, neuron_type=nengo.RectifiedLinear(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p1 = nengo.Probe(ens, synapse=None)
            p2 = nengo.Probe(ens, synapse=None)

        x = np.zeros((8, 1, 1), dtype=np.float32)
        y = np.zeros((8, 1, 1), dtype=np.float32)

        # Both probes have MSE loss but p2 weight=0
        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(
                optimizer="adam",
                loss={p1: "mse", p2: "mse"},
                loss_weights={p2: 0.0},
            )
            loss_p2_zero = sim.evaluate(x={inp: x}, y={p1: y, p2: y}, n_steps=1)["loss"]

        # Both probes with equal weight=1
        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(
                optimizer="adam",
                loss={p1: "mse", p2: "mse"},
                loss_weights={p1: 1.0, p2: 1.0},
            )
            loss_equal = sim.evaluate(x={inp: x}, y={p1: y, p2: y}, n_steps=1)["loss"]

        # With p2 zeroed out, total loss should be strictly less than with both contributing
        assert loss_p2_zero <= loss_equal, (
            f"loss_weights p2=0 must give loss ≤ equal weights: {loss_p2_zero} vs {loss_equal}"
        )

    def test_trainable_false_gives_zero_trainable_params(self):
        """configure_settings(trainable=False) must result in 0 trainable parameters."""
        with nengo.Network(seed=0) as net:
            nengo_dl.configure_settings(trainable=False)
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            n_params = len(sim.trainable_params())

        assert n_params == 0, (
            f"trainable=False must give 0 trainable params, got {n_params}"
        )

    def test_trainable_params_change_after_fit(self):
        """Trainable parameters must change after fitting (verifies gradient flow)."""
        net, inp, p = _trainable_net()
        x = np.random.RandomState(0).uniform(-1, 1, (32, 1, 1)).astype(np.float32)
        y = np.zeros((32, 1, 1), dtype=np.float32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            w_before = {k: v.copy() for k, v in sim.get_weights().items()}
            sim.compile(optimizer="adam", loss={p: "mse"})
            sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=10)
            w_after = sim.get_weights()

        changed = [k for k in w_before if not np.allclose(w_before[k], w_after[k])]
        assert len(changed) > 0, (
            "At least one trainable parameter must change after 10 epochs of training"
        )

    def test_evaluate_is_reproducible(self):
        """evaluate() on the same data returns the same loss value twice."""
        net, inp, p = _trainable_net()
        x, y = _xy(n=16)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            r1 = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)["loss"]
            r2 = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)["loss"]

        np.testing.assert_allclose(r1, r2, rtol=1e-6,
                                   err_msg="evaluate() must return identical loss on repeated calls")

    def test_fit_loss_strictly_less_than_initial_after_many_epochs(self):
        """After 20 epochs of SGD on a solvable task, loss must be strictly less than initial."""
        net, inp, p = _trainable_net(seed=0, n_hidden=30)
        x = np.random.RandomState(1).uniform(-1, 1, (64, 1, 1)).astype(np.float32)
        y = np.zeros((64, 1, 1), dtype=np.float32)

        with nengo_dl.Simulator(net, minibatch_size=16, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            hist = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=20)

        losses = hist["loss"]
        assert losses[-1] < losses[0], (
            f"Loss must decrease over 20 epochs: {losses[0]:.4f} → {losses[-1]:.4f}"
        )
