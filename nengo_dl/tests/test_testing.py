"""Tests that verify nengo-dl Simulator configuration and defaults.

Adapted from the original test_testing.py (which tested TF-specific fixture
parameters); this version tests the PyTorch Simulator's device, dtype, and
configure_settings behaviour.
"""

import numpy as np
import pytest
import torch
import nengo
import nengo_dl
from nengo_dl.config import configure_settings, get_setting


class TestSimulatorDefaults:
    def test_default_dtype_float32(self):
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net) as sim:
            assert sim.tensor_graph.dtype == torch.float32

    def test_default_device_is_available(self):
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net) as sim:
            device = sim.tensor_graph.device
            assert isinstance(device, torch.device)
            # device is either cpu or cuda — both are valid
            assert device.type in ("cpu", "cuda")

    def test_explicit_cpu_device(self):
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, device="cpu") as sim:
            assert sim.tensor_graph.device == torch.device("cpu")

    def test_default_minibatch_size_1(self):
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net) as sim:
            assert sim.minibatch_size == 1

    def test_custom_minibatch_size(self):
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, minibatch_size=8) as sim:
            assert sim.minibatch_size == 8

    def test_custom_dt(self):
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, dt=0.005) as sim:
            assert np.isclose(sim.dt, 0.005)


class TestConfigureSettings:
    def test_lif_smoothing_stored(self):
        with nengo.Network() as net:
            configure_settings(lif_smoothing=0.05)
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net) as sim:
            assert get_setting(sim._model, "lif_smoothing") == 0.05

    def test_inference_only_stored(self):
        with nengo.Network() as net:
            configure_settings(inference_only=True)
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net) as sim:
            assert get_setting(sim._model, "inference_only") is True

    def test_trainable_false_reduces_params(self):
        with nengo.Network(seed=0) as net:
            configure_settings(trainable=False)
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net) as sim:
            n_params = sum(
                p.numel()
                for p in sim.tensor_graph.parameters()
                if p.requires_grad
            )
        assert n_params == 0

    def test_manually_specified_device_not_overridden(self):
        """Explicitly passing device= to Simulator should take effect."""
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, device="cpu") as sim:
            assert sim.tensor_graph.device == torch.device("cpu")


class TestSimulatorLifecycle:
    def test_context_manager(self):
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net) as sim:
            sim.run_steps(1)
        assert sim.data[p].shape[0] == 1

    def test_close_is_safe(self):
        with nengo.Network() as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        sim = nengo_dl.Simulator(net)
        sim.__enter__()
        sim.run_steps(1)
        sim.__exit__(None, None, None)

    def test_multiple_simulators_independent(self):
        """Two Simulators built from the same Network should be independent."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim1:
            sim1.run_steps(3)
            d1 = sim1.data[p].copy()

        with nengo_dl.Simulator(net, seed=0) as sim2:
            sim2.run_steps(3)
            d2 = sim2.data[p].copy()

        np.testing.assert_array_equal(d1, d2)
