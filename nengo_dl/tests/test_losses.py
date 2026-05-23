"""Tests for nengo_dl.losses."""

import numpy as np
import pytest
import torch
import nengo
import nengo_dl
import nengo_dl.losses as losses_mod
from nengo_dl.losses import (
    MSE, MAE, CrossEntropy, SpikeRateLoss, TargetFiringRate, ProbeObjective
)


def _t(*shape, fill=0.0):
    return torch.full(shape, fill, dtype=torch.float32)


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------

class TestMSE:
    def test_perfect_prediction_zero_loss(self):
        loss = MSE()
        pred = _t(4, 2, 3, fill=1.0)
        target = _t(4, 2, 3, fill=1.0)
        assert loss(pred, target).item() == pytest.approx(0.0)

    def test_unit_offset_gives_one(self):
        loss = MSE()
        pred = torch.zeros(2, 1, 4)
        target = torch.ones(2, 1, 4)
        assert loss(pred, target).item() == pytest.approx(1.0)

    def test_mse_formula(self):
        loss = MSE()
        pred = torch.tensor([[[1.0, 2.0]]])
        target = torch.tensor([[[3.0, 4.0]]])
        expected = ((1 - 3) ** 2 + (2 - 4) ** 2) / 2
        assert loss(pred, target).item() == pytest.approx(expected)

    def test_scalar_output(self):
        loss = MSE()
        out = loss(_t(8, 5, 3, fill=0.0), _t(8, 5, 3, fill=0.5))
        assert out.shape == ()


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

class TestMAE:
    def test_zero_loss(self):
        loss = MAE()
        x = _t(3, 2, 5, fill=0.7)
        assert loss(x, x.clone()).item() == pytest.approx(0.0)

    def test_unit_offset(self):
        loss = MAE()
        pred = torch.zeros(2, 1, 4)
        target = torch.ones(2, 1, 4)
        assert loss(pred, target).item() == pytest.approx(1.0)

    def test_scalar_output(self):
        loss = MAE()
        out = loss(_t(4, 3, 2, fill=0.0), _t(4, 3, 2, fill=1.0))
        assert out.shape == ()


# ---------------------------------------------------------------------------
# CrossEntropy
# ---------------------------------------------------------------------------

class TestCrossEntropy:
    def test_perfect_hard_labels(self):
        """Predicting correct class with high logit → low loss."""
        loss = CrossEntropy(from_logits=True)
        # batch=2, n_steps=1, n_classes=3
        pred = torch.tensor([[[10.0, 0.0, 0.0]], [[0.0, 10.0, 0.0]]])
        target = torch.tensor([[[1, 0, 0]], [[0, 1, 0]]], dtype=torch.float32)
        val = loss(pred, target).item()
        assert val < 0.1

    def test_uniform_logits_high_loss(self):
        loss = CrossEntropy(from_logits=True)
        pred = torch.zeros(4, 1, 10)
        target = torch.zeros(4, 1, 10)
        target[:, :, 0] = 1.0  # class 0
        val = loss(pred, target).item()
        assert val > 1.0

    def test_scalar_output(self):
        loss = CrossEntropy()
        pred = torch.randn(4, 2, 5)
        target = torch.zeros(4, 2, 5)
        target[:, :, 0] = 1.0
        out = loss(pred, target)
        assert out.shape == ()


# ---------------------------------------------------------------------------
# SpikeRateLoss
# ---------------------------------------------------------------------------

class TestSpikeRateLoss:
    def test_at_target_rate_zero_loss(self):
        loss = SpikeRateLoss(target_rate=20.0)
        pred = _t(4, 10, 8, fill=20.0)  # mean rate = 20.0
        val = loss(pred).item()
        assert val == pytest.approx(0.0, abs=1e-5)

    def test_above_target_nonzero_loss(self):
        loss = SpikeRateLoss(target_rate=10.0)
        pred = _t(2, 5, 4, fill=50.0)
        val = loss(pred).item()
        assert val > 0

    def test_weight_scales_loss(self):
        loss_w1 = SpikeRateLoss(target_rate=0.0, weight=1.0)
        loss_w2 = SpikeRateLoss(target_rate=0.0, weight=2.0)
        pred = _t(2, 5, 4, fill=10.0)
        assert loss_w2(pred).item() == pytest.approx(2.0 * loss_w1(pred).item())

    def test_accepts_no_target(self):
        loss = SpikeRateLoss(target_rate=5.0)
        pred = _t(2, 10, 4, fill=5.0)
        val = loss(pred, None)  # target may be None
        assert np.isfinite(val.item())


# ---------------------------------------------------------------------------
# TargetFiringRate
# ---------------------------------------------------------------------------

class TestTargetFiringRate:
    def test_zero_loss_matching_rates(self):
        loss = TargetFiringRate()
        pred = _t(4, 10, 8, fill=30.0)
        target = _t(4, 10, 8, fill=30.0)
        val = loss(pred, target).item()
        assert val == pytest.approx(0.0, abs=1e-5)

    def test_positive_loss_on_mismatch(self):
        loss = TargetFiringRate()
        pred = _t(4, 10, 8, fill=0.0)
        target = _t(4, 10, 8, fill=50.0)
        assert loss(pred, target).item() > 0


# ---------------------------------------------------------------------------
# ProbeObjective
# ---------------------------------------------------------------------------

class TestProbeObjective:
    def test_wraps_callable(self):
        called = []

        def my_loss(pred, target):
            called.append(True)
            return (pred - target).pow(2).mean()

        obj = ProbeObjective(my_loss)
        pred = torch.ones(2, 3, 4)
        target = torch.zeros(2, 3, 4)
        val = obj(pred, target)
        assert called
        assert val.item() == pytest.approx(1.0)

    def test_is_nn_module(self):
        obj = ProbeObjective(lambda p, t: p.mean())
        assert isinstance(obj, torch.nn.Module)


# ---------------------------------------------------------------------------
# Loss functions in training loop
# ---------------------------------------------------------------------------

class TestLossesInTraining:
    @pytest.mark.parametrize("loss_name", ["mse", "mae"])
    def test_loss_string_in_compile(self, loss_name):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, neuron_type=nengo.RectifiedLinear(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            out = nengo.Node(size_in=1)
            nengo.Connection(ens, out, function=lambda x: x, synapse=None)
            p = nengo.Probe(out, synapse=None)

        x = np.zeros((8, 1, 1), dtype=np.float32)
        y = np.zeros((8, 1, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: loss_name})
            history = sim.fit(x={inp: x}, y={p: y}, n_steps=1, epochs=1)
        assert np.isfinite(history["loss"][0])

    def test_mse_module_in_compile(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, neuron_type=nengo.RectifiedLinear(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            out = nengo.Node(size_in=1)
            nengo.Connection(ens, out, function=lambda x: x, synapse=None)
            p = nengo.Probe(out, synapse=None)

        x = np.ones((8, 1, 1), dtype=np.float32)
        y = np.zeros((8, 1, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=8, seed=0) as sim:
            sim.compile(optimizer="adam", loss={p: MSE()})
            result = sim.evaluate(x={inp: x}, y={p: y}, n_steps=1)
        assert np.isfinite(result["loss"])
