"""Tests for nengo_dl.dists weight-initialization distributions."""

import math

import numpy as np
import pytest

from nengo_dl import dists


# ---------------------------------------------------------------------------
# Reference statistics for a zero-mean truncated normal
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))


def _norm_pdf(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x ** 2)


def _tnorm_var(scale, limit):
    """Theoretical variance of TruncatedNormal(0, scale, limit)."""
    a = -limit / scale
    b = limit / scale
    pdf_a = _norm_pdf(a)
    pdf_b = _norm_pdf(b)
    z = _norm_cdf(b) - _norm_cdf(a)
    return scale ** 2 * (
        1 + (a * pdf_a - b * pdf_b) / z - ((pdf_a - pdf_b) / z) ** 2
    )


# ---------------------------------------------------------------------------
# VarianceScaling helpers
# ---------------------------------------------------------------------------

def _test_variance_scaling(dist, scale, mode, seed):
    shape = (1000, 2000)
    rng = np.random.RandomState(seed)

    if mode == "fan_in":
        scale /= shape[1]
    elif mode == "fan_out":
        scale /= shape[0]
    else:
        scale /= np.mean(shape)

    if dist.distribution == "uniform":
        scale *= 3  # for bound calculation

    std = np.sqrt(scale)

    samples = dist.sample(shape[0], shape[1], rng=rng)

    assert samples.shape == shape
    assert np.allclose(np.mean(samples), 0.0, atol=5e-4)

    if dist.distribution == "uniform":
        expected_var = 4 * std ** 2 / 12  # = std²/3
        assert np.allclose(np.var(samples), expected_var, rtol=5e-3)
    else:
        assert np.allclose(np.var(samples), _tnorm_var(std, 2 * std), rtol=5e-3)

    # calling without explicit rng should also work
    samples2 = dist.sample(shape[0], shape[1])
    assert samples2.shape == shape


# ---------------------------------------------------------------------------
# VarianceScaling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("mode", ["fan_in", "fan_out", "fan_avg"])
@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_variance_scaling(scale, mode, distribution, seed):
    dist = dists.VarianceScaling(scale=scale, mode=mode, distribution=distribution)
    _test_variance_scaling(dist, scale, mode, seed)


# ---------------------------------------------------------------------------
# Glorot
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_glorot(scale, distribution, seed):
    dist = dists.Glorot(scale=scale, distribution=distribution)
    _test_variance_scaling(dist, scale, "fan_avg", seed)


# ---------------------------------------------------------------------------
# He
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_he(scale, distribution, seed):
    dist = dists.He(scale=scale, distribution=distribution)
    _test_variance_scaling(dist, scale ** 2, "fan_in", seed)


# ---------------------------------------------------------------------------
# TruncatedNormal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("limit", [None, 0.5])
@pytest.mark.parametrize("stddev", [1, 0.2])
def test_truncated_normal(limit, stddev, seed):
    rng = np.random.RandomState(seed)
    dist = dists.TruncatedNormal(mean=0, stddev=stddev, limit=limit)
    eff_limit = limit if limit is not None else 2 * stddev

    samples = dist.sample(1000, 2000, rng=rng)

    assert samples.shape == (1000, 2000)
    assert np.allclose(np.mean(samples), 0.0, atol=5e-3)
    assert np.allclose(np.var(samples), _tnorm_var(stddev, eff_limit), rtol=5e-3)
    assert np.all(samples < eff_limit + 1e-6)
    assert np.all(samples > -eff_limit - 1e-6)

    # calling without explicit rng should also work
    samples2 = dist.sample(1000, 2000)
    assert samples2.shape == (1000, 2000)


# ---------------------------------------------------------------------------
# Seeding (reproducibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dist",
    [
        dists.TruncatedNormal(),
        dists.VarianceScaling(),
        dists.Glorot(),
        dists.He(),
    ],
)
def test_seeding(dist, seed):
    s1 = dist.sample(100, rng=np.random.RandomState(seed))
    s2 = dist.sample(100, rng=np.random.RandomState(seed))
    assert np.allclose(s1, s2)


# ---------------------------------------------------------------------------
# Nengo integration: distributions usable as connection transforms
# ---------------------------------------------------------------------------

def test_glorot_as_transform():
    """Glorot should be usable directly as a nengo Connection transform."""
    import nengo
    import nengo_dl

    with nengo.Network(seed=0) as net:
        inp = nengo.Node(np.zeros(4))
        ens = nengo.Ensemble(20, 4, seed=0)
        nengo.Connection(inp, ens, transform=dists.Glorot(), synapse=None)
        p = nengo.Probe(ens, synapse=None)

    with nengo_dl.Simulator(net, seed=0) as sim:
        sim.run_steps(1)
    assert sim.data[p].shape == (1, 4)
    assert not np.any(np.isnan(sim.data[p]))


def test_he_as_transform():
    """He should be usable as a nengo Connection transform."""
    import nengo
    import nengo_dl

    with nengo.Network(seed=0) as net:
        inp = nengo.Node(np.zeros(2))
        ens = nengo.Ensemble(10, 2, seed=0)
        nengo.Connection(inp, ens, transform=dists.He(), synapse=None)
        p = nengo.Probe(ens, synapse=None)

    with nengo_dl.Simulator(net, seed=0) as sim:
        sim.run_steps(1)
    assert not np.any(np.isnan(sim.data[p]))
