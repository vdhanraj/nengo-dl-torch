"""Tests for nengo_dl.Simulator."""

import numpy as np
import pytest
import torch
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

    def test_probe_raises_without_run(self):
        net, inp, out, p = _make_simple_net()
        with nengo_dl.Simulator(net, seed=0) as sim:
            with pytest.raises(KeyError):
                _ = sim.data[p]


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
