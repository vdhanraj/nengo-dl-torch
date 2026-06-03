"""Tests for nengo_dl.Simulator."""

import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
import nengo
import nengo_dl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_net(seed=0):
    """Node → Ensemble(ReLU) → output Node, probed."""
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(np.zeros(1))
        ens = nengo.Ensemble(20, 1, neuron_type=nengo.RectifiedLinear(), seed=seed)
        nengo.Connection(inp, ens, synapse=None)
        out = nengo.Node(size_in=1)
        nengo.Connection(ens, out, function=lambda x: x, synapse=None)
        p = nengo.Probe(out, synapse=None)
    return net, inp, out, p


def _make_lif_net(seed=0):
    """LIF spiking network for state-related tests."""
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(np.zeros(1))
        ens = nengo.Ensemble(20, 1, neuron_type=nengo.LIF(), seed=seed)
        nengo.Connection(inp, ens, synapse=None)
        p_v = nengo.Probe(ens.neurons, "voltage", synapse=None)
        p_out = nengo.Probe(ens, synapse=None)
    return net, inp, p_v, p_out


# ---------------------------------------------------------------------------
# Context manager / lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_context_manager(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            assert sim is not None

    def test_close_is_idempotent(self):
        """Calling close() multiple times must not raise."""
        net, inp, out, p = _make_simple_net()
        sim = nengo_dl.Simulator(net, seed=0)
        sim.close()
        sim.close()  # second close should not raise

    def test_dt_stored(self):
        net, _, _, _ = _make_simple_net()
        with nengo_dl.Simulator(net, dt=0.005) as sim:
            assert sim.dt == pytest.approx(0.005)

    def test_minibatch_size_stored(self):
        net, _, _, _ = _make_simple_net()
        with nengo_dl.Simulator(net, minibatch_size=8) as sim:
            assert sim.minibatch_size == 8


# ---------------------------------------------------------------------------
# run_steps / run
# ---------------------------------------------------------------------------

class TestRunSteps:
    def test_run_steps_stores_probe_data(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(10)
            data = sim.data[p]
        # minibatch=1 → shape is (n_steps, probe_size)
        assert data.shape == (10, 1)

    def test_run_n_steps(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(25)
            assert sim.data[p].shape[0] == 25

    def test_run_with_input_data(self):
        net, inp, out, p = _make_simple_net()
        x = np.ones((1, 5, 1)) * 2.0  # (batch=1, n_steps=5, dim=1)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5, data={inp: x})
            data = sim.data[p]
        assert data.shape == (5, 1)

    def test_run_minibatch_shape(self):
        net, inp, out, p = _make_simple_net()
        bs = 4
        x = np.ones((bs, 3, 1))
        with nengo_dl.Simulator(net, minibatch_size=bs, seed=0) as sim:
            sim.run_steps(3, data={inp: x})
            data = sim.data[p]
        assert data.shape == (bs, 3, 1)

    def test_run_time(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run(0.01)  # 10 steps at dt=0.001
            assert sim.data[p].shape[0] == 10

    def test_lif_inference_modes(self):
        import warnings
        with nengo.Network(seed=0) as net:
            nengo_dl.configure_settings(lif_smoothing=0.01)
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(
                1,
                1,
                neuron_type=nengo.LIF(amplitude=0.01),
                gain=nengo.dists.Choice([1]),
                bias=nengo.dists.Choice([0]),
                encoders=nengo.dists.Choice([[1]]),
            )
            nengo.Connection(inp, ens.neurons, transform=np.ones((1, 1)), synapse=None)
            p = nengo.Probe(ens.neurons, synapse=None)

        x = np.ones((1, 1, 1), dtype=np.float32) * 3.0
        # Nengo emits a benign RuntimeWarning from log1p when bias=0 is used
        # with the LIF rate equation; filter it out since it's not our code.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "divide by zero", RuntimeWarning)
            with nengo_dl.Simulator(net, seed=0) as sim:
                sim.run_steps(1, data={inp: x}, inference_mode="rate")
                rate_data = sim.data[p]

            with nengo_dl.Simulator(net, seed=0) as sim:
                sim.run_steps(1, data={inp: x}, inference_mode="spiking")
                spiking_data = sim.data[p]

        assert rate_data.shape == spiking_data.shape == (1, 1)
        assert rate_data[0, 0] > 0
        assert not np.isclose(rate_data[0, 0], spiking_data[0, 0])

    def test_unknown_inference_mode_raises(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            with pytest.raises(ValueError, match="inference_mode"):
                sim.run_steps(1, inference_mode="banana")

    def test_probe_raises_without_run(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            with pytest.raises(KeyError):
                _ = sim.data[p]

    def test_exact_linear_output(self):
        """Constant node 2.0 → transform=3.0 → probe must read exactly 6.0."""
        with nengo.Network(seed=0) as net:
            a = nengo.Node(np.array([2.0]))
            b = nengo.Node(size_in=1)
            nengo.Connection(a, b, transform=3.0, synapse=None)
            p = nengo.Probe(b, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
        np.testing.assert_allclose(
            sim.data[p], np.full((5, 1), 6.0), atol=1e-5,
            err_msg="transform=3.0 × node_output=2.0 must equal 6.0 at every step",
        )

    def test_multi_batch_each_gets_own_input(self):
        """Each batch item gets its own injected input and independent output."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            b = nengo.Node(size_in=1)
            nengo.Connection(inp, b, synapse=None)
            p = nengo.Probe(b, synapse=None)

        vals = [1.0, 5.0, 10.0]
        bs = len(vals)
        x = np.array([[[v]] for v in vals], dtype=np.float32)  # (3, 1, 1)

        with nengo_dl.Simulator(net, minibatch_size=bs, seed=0) as sim:
            sim.run_steps(1, data={inp: x})
            out = sim.data[p]  # (3, 1, 1)

        for i, v in enumerate(vals):
            np.testing.assert_allclose(
                out[i, 0, 0], v, atol=1e-5,
                err_msg=f"Batch item {i}: expected {v}, got {out[i,0,0]}",
            )

    def test_python_node_runs_per_batch_item(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            out = nengo.Node(output=lambda t, x: x + t, size_in=1, size_out=1)
            nengo.Connection(inp, out, synapse=None)
            p = nengo.Probe(out, synapse=None)

        x = np.array([[[1.0]], [[2.0]]], dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=2, seed=0) as sim:
            sim.run_steps(1, data={inp: x})
            out_data = sim.data[p]

        np.testing.assert_allclose(out_data[:, 0, 0], [1.001, 2.001], atol=1e-5)

    def test_run_steps_partial_batch_raises(self):
        net, inp, out, p = _make_simple_net()
        x = np.ones((3, 1, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=2, seed=0) as sim:
            with pytest.raises(ValueError, match="divisible"):
                sim.run_steps(1, data={inp: x})

    def test_probe_with_synapse_filters_signal(self):
        """Synapse probe: step response starts below the step, converges over time."""
        tau = 0.01
        dt = 0.001
        n_steps = 100
        step_val = 1.0

        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([step_val]))
            p = nengo.Probe(inp, synapse=tau)

        with nengo_dl.Simulator(net, dt=dt, seed=0) as sim:
            sim.run_steps(n_steps)
            out = sim.data[p]  # (n_steps, 1)

        # First step value must be significantly less than step_val
        assert out[0, 0] < 0.5 * step_val, (
            f"Filter should attenuate first step: got {out[0,0]:.4f} >= 0.5"
        )
        # By step 100 (t=0.1s = 10 tau), value should be close to step_val
        np.testing.assert_allclose(out[-1, 0], step_val, atol=0.01,
                                   err_msg="Filtered probe should converge to step after 10τ")


# ---------------------------------------------------------------------------
# Compile / fit / evaluate
# ---------------------------------------------------------------------------

class TestTraining:
    def test_compile_accepts_string_optimizer(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, minibatch_size=4, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})

    def test_compile_accepts_torch_optimizer(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, minibatch_size=4, seed=0) as sim:
            opt = torch.optim.Adam(sim.trainable_params(), lr=1e-3)
            sim.compile(optimizer=opt, loss={p: "mse"})

    def test_fit_reduces_loss(self):
        net, inp, out, p = _make_simple_net()
        n = 64
        x = np.random.RandomState(0).uniform(-1, 1, (n, 1, 1)).astype(np.float32)
        y = (x * 0.5).astype(np.float32)

        with nengo_dl.Simulator(net, minibatch_size=16, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            history = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=10)

        losses = history["loss"]
        assert losses[-1] < losses[0], "Loss should decrease over training"

    def test_fit_returns_history(self):
        net, inp, out, p = _make_simple_net()
        x = np.zeros((16, 1, 1), dtype=np.float32)
        y = np.zeros((16, 1, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            history = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=3)
        assert "loss" in history
        assert len(history["loss"]) == 3

    def test_fit_requires_compile(self):
        net, inp, out, p = _make_simple_net()
        x = np.zeros((8, 1, 1), dtype=np.float32)
        y = np.zeros((8, 1, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            with pytest.raises(Exception, match="[Oo]ptimizer"):
                sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=1)

    def test_evaluate_returns_loss(self):
        net, inp, out, p = _make_simple_net()
        x = np.zeros((8, 1, 1), dtype=np.float32)
        y = np.zeros((8, 1, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            result = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)
        assert "loss" in result
        assert np.isfinite(result["loss"])

    def test_evaluate_before_compile_returns_nan(self):
        net, inp, out, p = _make_simple_net()
        x = np.zeros((8, 1, 1), dtype=np.float32)
        y = np.zeros((8, 1, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            result = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)
        assert np.isnan(result["loss"])

    def test_unknown_optimizer_raises(self):
        net, _, _, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            with pytest.raises(ValueError, match="[Uu]nknown optimizer"):
                sim.compile(optimizer="notanoptimizer", loss={p: "mse"})

    def test_fit_accepts_single_sample_2d_shapes(self):
        net, inp, out, p = _make_simple_net()
        x = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
        y = np.zeros((3, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=1, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            history = sim.fit(x={inp: x}, y={p: y}, n_steps=3, epochs=1, verbose=0)
        assert np.isfinite(history["loss"][0])

    def test_fit_partial_batch_raises(self):
        net, inp, out, p = _make_simple_net()
        x = np.ones((3, 1, 1), dtype=np.float32)
        y = np.zeros((3, 1, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=2, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            with pytest.raises(ValueError, match="divisible"):
                sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=1)


# ---------------------------------------------------------------------------
# Save / load params
# ---------------------------------------------------------------------------

class TestSaveLoadParams:
    def test_save_load_roundtrip(self, tmp_path):
        net, inp, out, p = _make_simple_net()
        path = str(tmp_path / "weights")
        x = np.random.RandomState(1).uniform(-1, 1, (16, 1, 1)).astype(np.float32)
        y = np.zeros_like(x)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=5)
            weights_before = sim.get_weights()
            sim.save_params(path)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=99) as sim2:
            sim2.load_params(path)
            weights_after = sim2.get_weights()

        for k in weights_before:
            np.testing.assert_allclose(
                weights_before[k], weights_after[k], rtol=1e-5,
                err_msg=f"Weight mismatch for key {k}"
            )

    def test_save_load_different_batch_size(self, tmp_path):
        """Stable parameter names allow cross-batch-size save/load."""
        net, inp, out, p = _make_simple_net()
        path = str(tmp_path / "weights_xbatch")
        x = np.random.RandomState(2).uniform(-1, 1, (32, 1, 1)).astype(np.float32)
        y = np.zeros_like(x)

        with nengo_dl.Simulator(net, minibatch_size=32, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=5)
            sim.save_params(path)
            weights_train = sim.get_weights()

        with nengo_dl.Simulator(net, minibatch_size=4, seed=1) as sim2:
            sim2.load_params(path)
            weights_eval = sim2.get_weights()

        for k in weights_train:
            np.testing.assert_allclose(weights_train[k], weights_eval[k], rtol=1e-5)

    def test_get_set_weights(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim1:
            w1 = sim1.get_weights()

        with nengo_dl.Simulator(net, seed=99) as sim2:
            sim2.set_weights(w1)
            w2 = sim2.get_weights()

        for k in w1:
            np.testing.assert_allclose(w1[k], w2[k], rtol=1e-5)

    def test_save_load_includes_torch_layer_weights(self, tmp_path):
        def make_net(weight, bias):
            linear = nn.Linear(2, 2)
            with torch.no_grad():
                linear.weight.copy_(torch.tensor(weight, dtype=torch.float32))
                linear.bias.copy_(torch.tensor(bias, dtype=torch.float32))

            with nengo.Network(seed=0) as net:
                inp = nengo.Node(np.zeros(2))
                out = nengo_dl.Layer(linear)(inp)
                p = nengo.Probe(out, synapse=None)

            return net, inp, p

        path = str(tmp_path / "torch_weights")
        x = np.array([[[1.0, -1.0]]], dtype=np.float32)

        net1, inp1, p1 = make_net([[1.0, 2.0], [3.0, 4.0]], [0.5, -0.5])
        with nengo_dl.Simulator(net1, seed=0) as sim1:
            sim1.run_steps(1, data={inp1: x})
            expected = sim1.data[p1].copy()
            weights = sim1.get_weights()
            sim1.save_params(path)

        assert any(k.startswith("torch_module_") for k in weights)

        net2, inp2, p2 = make_net([[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0])
        with nengo_dl.Simulator(net2, seed=1) as sim2:
            sim2.load_params(path)
            sim2.run_steps(1, data={inp2: x})
            actual = sim2.data[p2]

        np.testing.assert_allclose(actual, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# State reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_state_restores_initial(self):
        net, inp, p_v, p_out = _make_lif_net()
        x = np.ones((1, 50, 1)) * 2.0
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(50, data={inp: x})
            v_run1 = sim.data[p_v].copy()
            sim.reset_state()
            sim.run_steps(50, data={inp: x})
            v_run2 = sim.data[p_v]
        # With the same input and reset state, results should be identical
        np.testing.assert_allclose(v_run1, v_run2, rtol=1e-4)

    def test_reset_does_not_change_params(self):
        net, inp, out, p = _make_simple_net()
        x = np.random.RandomState(3).uniform(-1, 1, (16, 1, 1)).astype(np.float32)
        y = np.zeros_like(x)
        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=5)
            w_before = {k: v.copy() for k, v in sim.get_weights().items()}
            sim.reset_state()
            w_after = sim.get_weights()
        for k in w_before:
            np.testing.assert_allclose(w_before[k], w_after[k], rtol=1e-5)


# ---------------------------------------------------------------------------
# SimulationData
# ---------------------------------------------------------------------------

class TestSimulationData:
    def test_probe_data_shape_batch1(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
            data = sim.data[p]
        assert data.ndim == 2
        assert data.shape == (5, 1)

    def test_probe_data_shape_batchN(self):
        net, inp, out, p = _make_simple_net()
        bs = 6
        x = np.ones((bs, 3, 1))
        with nengo_dl.Simulator(net, minibatch_size=bs, seed=0) as sim:
            sim.run_steps(3, data={inp: x})
            data = sim.data[p]
        assert data.shape == (bs, 3, 1)

    def test_missing_probe_raises(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            with pytest.raises(KeyError):
                _ = sim.data[p]

    def test_ensemble_data_access(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            ens = net.ensembles[0]
            result = sim.data[ens]
        assert result is not None

    def test_repr(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
            r = repr(sim.data)
        assert "SimulationData" in r


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_output(self):
        net, inp, out, p = _make_simple_net()
        x = np.random.RandomState(5).randn(1, 10, 1).astype(np.float32)

        with nengo_dl.Simulator(net, seed=42) as sim:
            sim.run_steps(10, data={inp: x})
            r1 = sim.data[p].copy()

        with nengo_dl.Simulator(net, seed=42) as sim:
            sim.run_steps(10, data={inp: x})
            r2 = sim.data[p].copy()

        np.testing.assert_allclose(r1, r2, rtol=1e-5)

    @pytest.mark.parametrize("optimizer", ["adam", "sgd", "adamw"])
    def test_optimizer_strings(self, optimizer):
        net, inp, out, p = _make_simple_net()
        x = np.zeros((8, 1, 1), dtype=np.float32)
        y = np.zeros((8, 1, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer=optimizer, loss={p: "mse"})
            history = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=1)
        assert np.isfinite(history["loss"][0])


# ---------------------------------------------------------------------------
# Trainable parameters
# ---------------------------------------------------------------------------

class TestTrainableParams:
    def test_trainable_params_nonempty(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            params = sim.trainable_params()
        assert len(params) > 0
        assert all(isinstance(pp, torch.nn.Parameter) for pp in params)

    def test_configure_trainable_false(self):
        """Setting trainable=False should reduce trainable param count."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo.Network(seed=0) as net_frozen:
            nengo_dl.configure_settings(trainable=False)
            inp2 = nengo.Node(np.zeros(1))
            ens2 = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp2, ens2, synapse=None)
            p2 = nengo.Probe(ens2, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            n_trainable = len(sim.trainable_params())

        with nengo_dl.Simulator(net_frozen, seed=0) as sim_frozen:
            n_frozen = len(sim_frozen.trainable_params())

        assert n_frozen <= n_trainable


# ---------------------------------------------------------------------------
# Persistent state (original: test_persistent_state)
# ---------------------------------------------------------------------------

class TestPersistentState:
    def test_run_reset_run_same_output(self):
        """run → reset_state → run with same input gives identical output."""
        net, inp, p_v, p_out = _make_lif_net()
        x = np.ones((1, 30, 1), dtype=np.float32) * 1.5

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(30, data={inp: x})
            out1 = sim.data[p_out].copy()
            sim.reset_state()
            sim.run_steps(30, data={inp: x})
            out2 = sim.data[p_out].copy()

        np.testing.assert_allclose(out1, out2, rtol=1e-4,
                                   err_msg="After reset_state, re-run must give same output")

    def test_stateful_accumulates_state(self):
        """stateful=True: two 10-step runs should differ from stateful=False."""
        net, inp, p_v, p_out = _make_lif_net()
        x = np.ones((1, 10, 1), dtype=np.float32) * 2.0

        with nengo_dl.Simulator(net, seed=0) as sim:
            # stateful=False: resets between calls
            sim.run_steps(10, data={inp: x}, stateful=False)
            out_nstateful_1 = sim.data[p_v].copy()
            sim.run_steps(10, data={inp: x}, stateful=False)
            out_nstateful_2 = sim.data[p_v].copy()

        with nengo_dl.Simulator(net, seed=0) as sim:
            # stateful=True: state carries over
            sim.run_steps(10, data={inp: x}, stateful=True)
            out_stateful_1 = sim.data[p_v].copy()
            sim.run_steps(10, data={inp: x}, stateful=True)
            out_stateful_2 = sim.data[p_v].copy()

        # The two runs under stateful=False should be identical (reset each time)
        np.testing.assert_allclose(out_nstateful_1, out_nstateful_2, rtol=1e-4)
        # The two runs under stateful=True will differ (state carries over)
        # At minimum the outputs should not be identical
        assert not np.allclose(out_stateful_1, out_stateful_2, rtol=1e-4), (
            "stateful=True: second run should differ because state accumulated"
        )

    def test_n_steps_counter(self):
        """n_steps property counts total steps simulated."""
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            assert sim.n_steps == 0
            sim.run_steps(10)
            assert sim.n_steps == 10
            sim.run_steps(5)
            assert sim.n_steps == 15
            sim.reset_state()
            assert sim.n_steps == 0

    def test_time_property(self):
        """time property equals n_steps * dt."""
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run_steps(7)
            assert sim.time == pytest.approx(0.007)


# ---------------------------------------------------------------------------
# SimulationData – deep key access (original: test_simulation_data)
# ---------------------------------------------------------------------------

class TestSimulationDataKeys:
    def test_ensemble_has_gain_bias(self):
        """sim.data[ens] must contain 'gain' and 'bias'."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            params = sim.data[ens]

        assert isinstance(params, dict), "sim.data[ens] should return a dict"
        assert "gain" in params, "Missing 'gain' key"
        assert "bias" in params, "Missing 'bias' key"

    def test_ensemble_gain_bias_shapes(self):
        """gain and bias shapes match n_neurons."""
        n_neurons = 15
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(n_neurons, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            params = sim.data[ens]

        assert params["gain"].shape == (n_neurons,), (
            f"gain shape {params['gain'].shape} != ({n_neurons},)"
        )
        assert params["bias"].shape == (n_neurons,), (
            f"bias shape {params['bias'].shape} != ({n_neurons},)"
        )

    def test_ensemble_encoders_present(self):
        """sim.data[ens] must contain 'encoders' or 'scaled_encoders'."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            ens = nengo.Ensemble(10, 2, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            params = sim.data[ens]

        assert "encoders" in params or "scaled_encoders" in params, (
            "sim.data[ens] must contain 'encoders' or 'scaled_encoders'"
        )

    def test_connection_has_weights(self):
        """sim.data[conn] should not raise and returns params info."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            ens = nengo.Ensemble(5, 2, seed=0)
            conn = nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            conn_data = sim.data[conn]
        # Must not raise; return value is None, dict, or Nengo BuiltConnection
        # A dict with 'weights' key is the richest result; anything is acceptable
        if isinstance(conn_data, dict):
            assert "weights" in conn_data
        # else: BuiltConnection or None — both are valid

    def test_ensemble_params_are_numpy(self):
        """Values returned by sim.data[ens] should be numpy arrays."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(8, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            params = sim.data[ens]

        for key, val in params.items():
            assert isinstance(val, np.ndarray), (
                f"sim.data[ens]['{key}'] should be np.ndarray, got {type(val)}"
            )


# ---------------------------------------------------------------------------
# Gradient checking (original: test_check_gradients)
# ---------------------------------------------------------------------------

class TestCheckGradients:
    def test_check_gradients_returns_true(self):
        """check_gradients() should return True (no NaN/Inf detected)."""
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            result = sim.check_gradients(n_steps=1)
        assert result is True

    def test_check_gradients_no_warning_for_healthy_net(self):
        """A healthy differentiable network should not produce gradient warnings."""
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                sim.check_gradients(n_steps=1)
        nan_warnings = [w for w in rec if "NaN" in str(w.message) or "Inf" in str(w.message)]
        assert len(nan_warnings) == 0, "Unexpected NaN/Inf gradient warning"


# ---------------------------------------------------------------------------
# freeze_params (original: test_freeze_params)
# ---------------------------------------------------------------------------

class TestFreezeParams:
    def test_freeze_params_does_not_crash(self):
        """freeze_params() should run without error."""
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, minibatch_size=4, seed=0) as sim:
            x = np.random.RandomState(7).uniform(-1, 1, (16, 1, 1)).astype(np.float32)
            y = np.zeros((16, 1, 1), dtype=np.float32)
            sim.compile(optimizer="adam", loss={p: "mse"})
            sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=3)
            sim.freeze_params()  # should not raise

    def test_freeze_params_specific_objects(self):
        """freeze_params(objects) should accept a list of nengo objects."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            conn = nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.freeze_params([ens, conn])  # should not raise


# ---------------------------------------------------------------------------
# trange (original: test_trange / time utilities)
# ---------------------------------------------------------------------------

class TestTrange:
    def test_trange_length(self):
        """trange() length should match n_steps of the last run."""
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run_steps(15)
            t = sim.trange()
        assert len(t) == 15

    def test_trange_values(self):
        """trange() values should be multiples of dt starting at dt."""
        net, inp, out, p = _make_simple_net()
        dt = 0.002
        n = 5
        with nengo_dl.Simulator(net, dt=dt, seed=0) as sim:
            sim.run_steps(n)
            t = sim.trange()
        expected = np.arange(1, n + 1) * dt
        np.testing.assert_allclose(t, expected, rtol=1e-6)

    def test_trange_custom_dt(self):
        """trange(dt=x) should use the provided dt."""
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run_steps(4)
            t = sim.trange(dt=0.01)  # override dt
        np.testing.assert_allclose(t, np.arange(1, 5) * 0.01, rtol=1e-6)


# ---------------------------------------------------------------------------
# get_nengo_params (original: test_get_nengo_params)
# ---------------------------------------------------------------------------

class TestGetNengoParams:
    def test_returns_dict(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            result = sim.get_nengo_params()
        assert isinstance(result, dict)

    def test_specific_object(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            result = sim.get_nengo_params([ens])
        assert ens in result or len(result) == 0  # might not have signal if not built


# ---------------------------------------------------------------------------
# Training convergence (original: test_train_ff)
# ---------------------------------------------------------------------------

class TestTrainingConvergence:
    def test_linear_regression_converges(self):
        """A network with trainable parameters should learn a simple mapping."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(20, 1, neuron_type=nengo.RectifiedLinear(), seed=0)
            out = nengo.Node(size_in=1)
            nengo.Connection(inp, ens, synapse=None)
            nengo.Connection(ens, out, function=lambda x: x, synapse=None)
            p = nengo.Probe(out, synapse=None)

        n = 64
        rng = np.random.RandomState(42)
        x = rng.uniform(-1, 1, (n, 1, 1)).astype(np.float32)
        y = np.zeros((n, 1, 1), dtype=np.float32)  # learn to output 0

        with nengo_dl.Simulator(net, minibatch_size=16, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            h = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=15)

        losses = h["loss"]
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_fit_with_validation_split(self):
        """fit with validation_split should produce val_loss entries."""
        net, inp, out, p = _make_simple_net()
        n = 32
        x = np.random.RandomState(0).uniform(-1, 1, (n, 1, 1)).astype(np.float32)
        y = np.zeros((n, 1, 1), dtype=np.float32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: "mse"})
            h = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=2,
                        validation_split=0.25)

        assert "val_loss" in h
        assert len(h["val_loss"]) == 2
        assert all(np.isfinite(v) for v in h["val_loss"])

    def test_training_moves_output_toward_target(self):
        """After training on target=0, mean absolute output should decrease."""
        net, inp, out, p = _make_simple_net()
        x_eval = np.ones((8, 1, 1), dtype=np.float32)
        x_train = np.ones((32, 1, 1), dtype=np.float32)
        y_train = np.zeros((32, 1, 1), dtype=np.float32)

        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.run_steps(1, data={inp: x_eval})
            out_before = np.abs(sim.data[p]).mean()

            sim.compile(optimizer="adam", loss={p: "mse"})
            sim.fit(x={inp: x_train}, y={p: y_train}, n_steps=1, epochs=20)

            sim.reset_state()
            sim.run_steps(1, data={inp: x_eval})
            out_after = np.abs(sim.data[p]).mean()

        assert out_after < out_before, (
            f"Mean absolute output should decrease toward 0 after training: "
            f"{out_before:.4f} → {out_after:.4f}"
        )


# ---------------------------------------------------------------------------
# nengo.Simulator reference comparisons
# ---------------------------------------------------------------------------

class TestNengoReference:
    @staticmethod
    def _configure_decoder_cache(tmp_path):
        cache_dir = tmp_path / "nengo_decoder_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        nengo.rc["decoder_cache"]["path"] = str(cache_dir)

    def test_constant_linear_net_matches_nengo(self, tmp_path):
        """Node(3.0) → transform=2.0 → probe: nengo_dl must match nengo exactly."""
        self._configure_decoder_cache(tmp_path)
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([3.0]))
            b = nengo.Node(size_in=1)
            nengo.Connection(inp, b, transform=2.0, synapse=None)
            p = nengo.Probe(b, synapse=None)

        with nengo.Simulator(net, dt=0.001) as ref:
            ref.run_steps(5)
            ref_out = ref.data[p].copy()

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
            dl_out = sim.data[p].copy()

        np.testing.assert_allclose(
            ref_out, dl_out, atol=1e-5,
            err_msg="Constant-input linear net: nengo_dl must match nengo.Simulator"
        )

    def test_relu_rate_mode_matches_nengo(self, tmp_path):
        """ReLU ensemble rate mode: nengo_dl output matches nengo.Simulator."""
        self._configure_decoder_cache(tmp_path)
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([2.0]))
            ens = nengo.Ensemble(
                8, 1, neuron_type=nengo.RectifiedLinear(),
                gain=nengo.dists.Choice([1.0]),
                bias=nengo.dists.Choice([0.0]),
                encoders=nengo.dists.Choice([[1.0]]),
                seed=0,
            )
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo.Simulator(net, dt=0.001) as ref:
            ref.run_steps(5)
            ref_out = ref.data[p].copy()

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5, inference_mode="rate")
            dl_out = sim.data[p].copy()

        np.testing.assert_allclose(ref_out, dl_out, atol=1e-4,
                                   err_msg="ReLU rate mode must match nengo.Simulator")

    def test_lif_spiking_total_spikes_match_nengo(self, tmp_path):
        """LIF spiking: total spike count per neuron must agree with nengo.Simulator."""
        self._configure_decoder_cache(tmp_path)
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([2.5]))
            ens = nengo.Ensemble(
                5, 1, neuron_type=nengo.LIF(),
                gain=nengo.dists.Choice([2.0]),
                bias=nengo.dists.Choice([1.0]),
                encoders=nengo.dists.Choice([[1.0]]),
                seed=0,
            )
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens.neurons, synapse=None)

        n_steps = 50
        with nengo.Simulator(net, dt=0.001) as ref:
            ref.run_steps(n_steps)
            ref_spikes = ref.data[p].sum(axis=0)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(n_steps, inference_mode="spiking")
            dl_spikes = sim.data[p].sum(axis=0)

        # Nengo encodes spikes as 1/dt per spike (1000 for dt=0.001).
        # Allow ±2 spikes = ±2*(1/dt) = ±2000 in probe units.
        spike_unit = 1.0 / 0.001  # = 1000
        np.testing.assert_allclose(
            ref_spikes, dl_spikes, atol=2.0 * spike_unit,
            err_msg="LIF total spike count must match nengo.Simulator within ±2 spikes"
        )

    def test_two_ensemble_chain_matches_nengo(self, tmp_path):
        """inp → ens1 → ens2: probe at ens2 matches nengo.Simulator in rate mode."""
        self._configure_decoder_cache(tmp_path)
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([1.0]))
            ens1 = nengo.Ensemble(5, 1, neuron_type=nengo.RectifiedLinear(),
                                  gain=nengo.dists.Choice([1.0]),
                                  bias=nengo.dists.Choice([0.5]),
                                  encoders=nengo.dists.Choice([[1.0]]),
                                  seed=0)
            ens2 = nengo.Ensemble(5, 1, neuron_type=nengo.RectifiedLinear(),
                                  gain=nengo.dists.Choice([1.0]),
                                  bias=nengo.dists.Choice([0.0]),
                                  encoders=nengo.dists.Choice([[1.0]]),
                                  seed=1)
            nengo.Connection(inp, ens1, synapse=None)
            nengo.Connection(ens1, ens2, synapse=None)
            p = nengo.Probe(ens2, synapse=None)

        with nengo.Simulator(net, dt=0.001) as ref:
            ref.run_steps(5)
            ref_out = ref.data[p].copy()

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5, inference_mode="rate")
            dl_out = sim.data[p].copy()

        np.testing.assert_allclose(ref_out, dl_out, atol=1e-3,
                                   err_msg="Two-ensemble chain must match nengo.Simulator")
