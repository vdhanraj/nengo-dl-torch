"""Loss functions for training nengo-dl networks.

These follow PyTorch conventions: each loss function takes (prediction, target)
tensors of shape (batch, n_steps, *probe_shape) and returns a scalar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbeObjective(nn.Module):
    """Wrap a probe-specific loss to work with nengo-dl's training loop.

    Parameters
    ----------
    loss_fn : callable
        Loss function with signature ``loss_fn(pred, target) -> scalar``.
    """

    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target):
        return self.loss_fn(pred, target)


class MSE(nn.Module):
    """Mean-squared error loss.

    Accepts tensors of any shape (batch, n_steps, ...) and computes
    the mean squared error over all elements.
    """

    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class MAE(nn.Module):
    """Mean absolute error loss."""

    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class CrossEntropy(nn.Module):
    """Cross-entropy loss for classification.

    Expects ``pred`` of shape ``(batch, n_steps, n_classes)`` and
    ``target`` of shape ``(batch, n_steps)`` (class indices) or
    ``(batch, n_steps, n_classes)`` (one-hot).
    """

    def __init__(self, from_logits: bool = True):
        super().__init__()
        self.from_logits = from_logits

    def forward(self, pred, target):
        # Flatten batch and time dimensions
        b, t = pred.shape[:2]
        pred_flat = pred.reshape(b * t, -1)

        if target.dtype in (torch.float32, torch.float64):
            # One-hot → hard labels
            if target.dim() == pred.dim():
                # Broadcast target along time axis if shapes don't match
                # (e.g. target has 1 time step but pred has n_steps).
                if target.shape[1] != t:
                    target = target.expand(b, t, *target.shape[2:])
                target = target.reshape(b * t, -1)
                if self.from_logits:
                    return F.cross_entropy(
                        pred_flat, target.argmax(dim=-1)
                    )
                else:
                    return -(target * torch.log(pred_flat + 1e-8)).sum(dim=-1).mean()
            else:
                target_flat = target.reshape(b * t).long()
        else:
            target_flat = target.reshape(b * t).long()

        if self.from_logits:
            return F.cross_entropy(pred_flat, target_flat)
        else:
            return F.nll_loss(torch.log(pred_flat + 1e-8), target_flat)


class SpikeRateLoss(nn.Module):
    """Penalise the overall firing rate of a population.

    Can be used as a regulariser to encourage sparse spiking.

    Parameters
    ----------
    target_rate : float
        Target firing rate in Hz.
    dt : float
        Timestep in seconds.
    weight : float
        Relative weight for this loss term.
    """

    def __init__(self, target_rate: float = 20.0, dt: float = 0.001,
                 weight: float = 1.0):
        super().__init__()
        self.target_rate = target_rate
        self.dt = dt
        self.weight = weight

    def forward(self, pred, target=None):
        # pred: (batch, n_steps, n_neurons) – spike output (spikes/s or raw spikes)
        mean_rate = pred.mean()
        return self.weight * (mean_rate - self.target_rate) ** 2


class TargetFiringRate(nn.Module):
    """MSE loss on the mean firing rate of a population.

    Parameters
    ----------
    dt : float
        Timestep in seconds.
    """

    def __init__(self, dt: float = 0.001):
        super().__init__()
        self.dt = dt

    def forward(self, pred, target):
        # Average over time → mean firing rate
        mean_rate = pred.mean(dim=1)      # (batch, n_neurons)
        target_rate = target.mean(dim=1)  # same shape
        return F.mse_loss(mean_rate, target_rate)
