"""Tests for Nengo transform support in nengo-dl."""

import numpy as np
import pytest
import nengo
import nengo_dl


# ---------------------------------------------------------------------------
# Dense transforms
# ---------------------------------------------------------------------------

class TestDenseTransforms:
    def test_scalar_transform(self):
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([2.0]))
            dst = nengo.Node(size_in=1)
            nengo.Connection(src, dst, transform=5.0, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        assert sim.data[p][0, 0] == pytest.approx(10.0, abs=1e-4)

    def test_identity_transform(self):
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([3.0, -1.0]))
            dst = nengo.Node(size_in=2)
            nengo.Connection(src, dst, transform=np.eye(2), synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        np.testing.assert_allclose(sim.data[p][0], [3.0, -1.0], atol=1e-4)

    def test_matrix_transform_reduces_dim(self):
        """[1, 1] transform sums two inputs → 1 output."""
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([3.0, 4.0]))
            dst = nengo.Node(size_in=1)
            nengo.Connection(src, dst, transform=[[1.0, 1.0]], synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        assert sim.data[p][0, 0] == pytest.approx(7.0, abs=1e-4)

    def test_matrix_transform_expands_dim(self):
        """Column vector duplicates 1-D input → 2-D output."""
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([5.0]))
            dst = nengo.Node(size_in=2)
            nengo.Connection(src, dst, transform=[[2.0], [3.0]], synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        np.testing.assert_allclose(sim.data[p][0], [10.0, 15.0], atol=1e-4)

    def test_nengo_dense_transform_object(self):
        W = np.array([[1.0, 0.0], [0.0, -1.0]])
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([4.0, 6.0]))
            dst = nengo.Node(size_in=2)
            nengo.Connection(src, dst, transform=nengo.Dense(W.shape, init=W), synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        np.testing.assert_allclose(sim.data[p][0], [4.0, -6.0], atol=1e-4)

    def test_negative_transform(self):
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([3.0]))
            dst = nengo.Node(size_in=1)
            nengo.Connection(src, dst, transform=-1.0, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        assert sim.data[p][0, 0] == pytest.approx(-3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Connection function (decoder)
# ---------------------------------------------------------------------------

class TestConnectionFunction:
    def test_nef_function_connection(self):
        """Connection with function= uses NEF least-squares decoder."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(100, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            out = nengo.Node(size_in=1)
            # Identity function as baseline
            nengo.Connection(ens, out, function=lambda x: x, synapse=None)
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
        assert sim.data[p].shape == (5, 1)
        assert not np.any(np.isnan(sim.data[p]))

    def test_nef_sin_function(self):
        """NEF should approximate sin(x) across its input range."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(200, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            out = nengo.Node(size_in=1)
            nengo.Connection(ens, out, function=lambda x: np.sin(x[0]), synapse=None)
            p = nengo.Probe(out, synapse=None)

        n_pts = 100
        x_sweep = np.linspace(-1, 1, n_pts).reshape(1, n_pts, 1).astype(np.float32)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(n_pts, data={inp: x_sweep})
            y_pred = sim.data[p][:, 0]

        y_true = np.sin(np.linspace(-1, 1, n_pts))
        mse = np.mean((y_true - y_pred) ** 2)
        assert mse < 0.1, f"NEF sin approximation too poor: MSE={mse:.4f}"


# ---------------------------------------------------------------------------
# Multiple connections to same destination
# ---------------------------------------------------------------------------

class TestMultipleConnections:
    def test_two_sources_sum(self):
        """Two connections to the same destination node should sum."""
        with nengo.Network(seed=0) as net:
            src1 = nengo.Node(np.array([2.0]))
            src2 = nengo.Node(np.array([3.0]))
            dst = nengo.Node(size_in=1)
            nengo.Connection(src1, dst, synapse=None)
            nengo.Connection(src2, dst, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        # dst = src1 + src2 = 5.0
        assert sim.data[p][0, 0] == pytest.approx(5.0, abs=1e-4)

    def test_recurrent_connection(self):
        """A recurrent ensemble connection should not crash."""
        with nengo.Network(seed=0) as net:
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(ens, ens, transform=0.9, synapse=0.005)
            inp = nengo.Node(np.zeros(1))
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(10)
        assert sim.data[p].shape == (10, 1)


# ---------------------------------------------------------------------------
# Sliced connections
# ---------------------------------------------------------------------------

class TestSlicedConnections:
    def test_slice_source(self):
        """Connect a slice of the source to the destination."""
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([1.0, 2.0, 3.0]))
            dst = nengo.Node(size_in=1)
            # Only connect first element
            nengo.Connection(src[0], dst, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        assert sim.data[p][0, 0] == pytest.approx(1.0, abs=1e-4)

    def test_slice_destination(self):
        """Connect a 1-D source to a slice of the destination ensemble."""
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([5.0]))
            ens = nengo.Ensemble(10, 2, seed=0)
            inp = nengo.Node(np.zeros(2))
            nengo.Connection(inp, ens, synapse=None)
            # Drive first dimension only
            nengo.Connection(src, ens[0], synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(3)
        assert sim.data[p].shape == (3, 2)


# ---------------------------------------------------------------------------
# Batched transforms
# ---------------------------------------------------------------------------

class TestBatchedTransforms:
    def test_matrix_transform_batched(self):
        W = np.array([[2.0, 0.0], [0.0, 3.0]])
        bs = 4
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.zeros(2))
            dst = nengo.Node(size_in=2)
            nengo.Connection(src, dst, transform=W, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        x = np.ones((bs, 3, 2), dtype=np.float32)
        with nengo_dl.Simulator(net, minibatch_size=bs, seed=0) as sim:
            sim.run_steps(3, data={src: x})
        data = sim.data[p]
        assert data.shape == (bs, 3, 2)
        np.testing.assert_allclose(data[0, 0], [2.0, 3.0], atol=1e-4)
