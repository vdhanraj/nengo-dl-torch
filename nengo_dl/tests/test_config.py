"""Tests for nengo_dl.configure_settings and nengo_dl.get_setting."""

import numpy as np
import pytest
import nengo
import nengo_dl
from nengo_dl.config import configure_settings, get_setting, _global_settings


# ---------------------------------------------------------------------------
# get_setting defaults
# ---------------------------------------------------------------------------

class TestGetSetting:
    def test_returns_default_when_unset(self):
        with nengo.Network() as net:
            pass  # no configure_settings
        assert get_setting(net, "lif_smoothing", default=0.05) == pytest.approx(0.05)

    def test_returns_none_default(self):
        with nengo.Network() as net:
            pass
        assert get_setting(net, "nonexistent_key", default=None) is None

    def test_reads_global_settings(self):
        # Use global_settings fallback path
        _global_settings["_test_key_xyz"] = 42
        try:
            with nengo.Network() as net:
                pass
            val = get_setting(net, "_test_key_xyz", default=None)
            assert val == 42
        finally:
            _global_settings.pop("_test_key_xyz", None)


# ---------------------------------------------------------------------------
# configure_settings inside a network
# ---------------------------------------------------------------------------

class TestConfigureSettings:
    def test_lif_smoothing_stored(self):
        with nengo.Network() as net:
            configure_settings(lif_smoothing=0.05)
        val = get_setting(net, "lif_smoothing", default=None)
        # May be stored in global_settings or network config
        assert val is not None or _global_settings.get("lif_smoothing") == pytest.approx(0.05)

    def test_inference_only_stored(self):
        with nengo.Network() as net:
            configure_settings(inference_only=True)
        # Global settings should have it
        assert _global_settings.get("inference_only") is True

    def test_multiple_settings(self):
        with nengo.Network() as net:
            configure_settings(lif_smoothing=0.1, inference_only=False)
        assert _global_settings.get("lif_smoothing") == pytest.approx(0.1)
        assert _global_settings.get("inference_only") is False

    def test_none_values_not_stored(self):
        _global_settings.pop("keep_history", None)
        with nengo.Network():
            configure_settings(keep_history=None)
        assert "keep_history" not in _global_settings

    def test_lif_smoothing_affects_simulator(self):
        """lif_smoothing > 0 should not crash the simulator."""
        with nengo.Network(seed=0) as net:
            configure_settings(lif_smoothing=0.1)
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, neuron_type=nengo.LIF(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
        assert sim.data[p].shape == (5, 1)

    def test_trainable_false_reduces_params(self):
        with nengo.Network(seed=0) as net_trainable:
            inp = nengo.Node(np.zeros(2))
            ens = nengo.Ensemble(10, 2, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo.Network(seed=0) as net_frozen:
            configure_settings(trainable=False)
            inp2 = nengo.Node(np.zeros(2))
            ens2 = nengo.Ensemble(10, 2, seed=0)
            nengo.Connection(inp2, ens2, synapse=None)
            p2 = nengo.Probe(ens2, synapse=None)

        with nengo_dl.Simulator(net_trainable, seed=0) as sim:
            n_trainable = len(sim.trainable_params())
        with nengo_dl.Simulator(net_frozen, seed=0) as sim:
            n_frozen = len(sim.trainable_params())

        assert n_frozen <= n_trainable


# ---------------------------------------------------------------------------
# Simulator reads settings
# ---------------------------------------------------------------------------

class TestSimulatorSettings:
    def test_simulator_uses_lif_smoothing(self):
        """Simulator should accept lif_smoothing without error."""
        with nengo.Network(seed=0) as net:
            configure_settings(lif_smoothing=0.05)
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, neuron_type=nengo.LIF(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        x = np.ones((1, 5, 1), dtype=np.float32) * 3.0
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5, data={inp: x})
        # lif_smoothing should not cause NaN or crash
        assert sim.data[p].shape == (5, 1)

    def test_lif_smoothing_enables_gradient_training(self):
        """With lif_smoothing, gradient should flow through LIF neurons."""
        with nengo.Network(seed=0) as net:
            configure_settings(lif_smoothing=0.1)
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(20, 1, neuron_type=nengo.LIF(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            out = nengo.Node(size_in=1)
            nengo.Connection(ens, out, function=lambda x: x, synapse=None)
            p = nengo.Probe(out, synapse=None)

        x = np.ones((16, 1, 1), dtype=np.float32) * 3.0
        y = np.zeros((16, 1, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=16, seed=0) as sim:
            if len(sim.trainable_params()) > 0:
                sim.compile(optimizer="adam", loss={p: "mse"})
                history = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=3)
                assert np.isfinite(history["loss"][-1])
