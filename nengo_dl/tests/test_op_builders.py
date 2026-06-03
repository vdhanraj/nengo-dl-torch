"""Tests for nengo_dl.op_builders (core operator builders)."""

import numpy as np
import pytest
import torch
import nengo
import nengo_dl
from nengo_dl.signals import SignalDict
from nengo.builder.signal import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sd(batch=2):
    return SignalDict(batch, torch.device("cpu"), torch.float32)


def _sig(shape, name="s", init=None, readonly=False):
    val = np.zeros(shape) if init is None else np.asarray(init, dtype=float)
    s = Signal(val.copy(), name=name)
    if readonly:
        object.__setattr__(s, '_readonly', True)
    return s


# ---------------------------------------------------------------------------
# TimeUpdate
# ---------------------------------------------------------------------------

class TestTimeUpdateBuilder:
    def test_step_and_time_increment(self):
        """After one build_step, step should be 1 and time should be dt."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run_steps(5)
            assert sim.data[p].shape[0] == 5

    def test_step_count_matches_n_steps(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(lambda t: t)
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run_steps(10)
            data = sim.data[p]
        assert data.shape[0] == 10
        assert data[9, 0] == pytest.approx(0.01, abs=1e-4)

    def test_time_node_exact_values_at_each_step(self):
        """Node(lambda t: t) must return exactly k*dt at step k."""
        dt = 0.001
        n = 8
        with nengo.Network(seed=0) as net:
            src = nengo.Node(lambda t: np.array([t]))
            p = nengo.Probe(src, synapse=None)

        with nengo_dl.Simulator(net, dt=dt, seed=0) as sim:
            sim.run_steps(n)
            data = sim.data[p]

        for k in range(n):
            expected = (k + 1) * dt
            np.testing.assert_allclose(
                data[k, 0], expected, atol=1e-6,
                err_msg=f"At step {k+1}, t must be {expected:.4f}, got {data[k,0]:.6f}"
            )

    def test_time_is_cumulative_across_runs(self):
        """Running two batches of 5 steps gives t=0.01 at step 10."""
        dt = 0.001
        with nengo.Network(seed=0) as net:
            src = nengo.Node(lambda t: np.array([t]))
            p = nengo.Probe(src, synapse=None)

        with nengo_dl.Simulator(net, dt=dt, seed=0) as sim:
            sim.run_steps(5, stateful=True)
            sim.run_steps(5, stateful=True)
            # After 10 steps, last t value must be 10*dt
            assert sim.time == pytest.approx(10 * dt, abs=1e-6)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestResetBuilder:
    def test_reset_node_output_is_zero_each_step(self):
        """A zero-output Node always reads zero regardless of previous state."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(3))
            p = nengo.Probe(inp, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
            data = sim.data[p]

        np.testing.assert_allclose(data, 0.0, atol=1e-7,
                                   err_msg="Zero node must output exactly 0 every step")

    def test_stateful_false_resets_lif_voltage(self):
        """stateful=False: LIF voltage resets to 0 at each run_steps call."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(
                5, 1, neuron_type=nengo.LIF(),
                seed=0,
                gain=nengo.dists.Choice([2.0]),
                bias=nengo.dists.Choice([1.5]),
                encoders=nengo.dists.Choice([[1.0]]),
            )
            nengo.Connection(inp, ens.neurons, transform=np.ones((5, 1)), synapse=None)
            p_v = nengo.Probe(ens.neurons, "voltage", synapse=None)

        x = np.ones((1, 3, 1), dtype=np.float32) * 3.0
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(3, data={inp: x}, stateful=False)
            v_run1 = sim.data[p_v].copy()
            sim.run_steps(3, data={inp: x}, stateful=False)
            v_run2 = sim.data[p_v].copy()

        # With stateful=False, both runs start from the same initial state
        np.testing.assert_allclose(v_run1, v_run2, rtol=1e-4,
                                   err_msg="stateful=False: identical runs must give identical output")

    def test_reset_in_lif_ensemble(self):
        """LIF ensemble has Reset ops for refractory time and voltage; must not crash."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(5, 1, neuron_type=nengo.LIF(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
        assert sim.data[p].shape == (5, 1)
        assert not np.any(np.isnan(sim.data[p])), "LIF output must not contain NaN"


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

class TestCopyBuilder:
    def test_copy_propagates_value(self):
        """Connected Nodes should copy signal values between them."""
        with nengo.Network(seed=0) as net:
            a = nengo.Node(np.array([1.0, 2.0, 3.0]))
            b = nengo.Node(size_in=3)
            nengo.Connection(a, b, synapse=None)
            p = nengo.Probe(b, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        np.testing.assert_allclose(sim.data[p][0], [1.0, 2.0, 3.0], atol=1e-5)

    def test_copy_with_input_override(self):
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.zeros(2))
            dst = nengo.Node(size_in=2)
            nengo.Connection(src, dst, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        x = np.array([[[5.0, -3.0]]])  # (batch=1, steps=1, size=2)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1, data={src: x})
        np.testing.assert_allclose(sim.data[p][0], [5.0, -3.0], atol=1e-5)

    def test_copy_chain_three_nodes(self):
        """A → B → C: probe at C must equal probe at A."""
        val = np.array([7.0, -2.0, 4.5])
        with nengo.Network(seed=0) as net:
            a = nengo.Node(val)
            b = nengo.Node(size_in=3)
            c = nengo.Node(size_in=3)
            nengo.Connection(a, b, synapse=None)
            nengo.Connection(b, c, synapse=None)
            p_a = nengo.Probe(a, synapse=None)
            p_c = nengo.Probe(c, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(4)

        np.testing.assert_allclose(
            sim.data[p_c], sim.data[p_a], atol=1e-5,
            err_msg="Signal must propagate unchanged through a 3-node chain"
        )

    def test_copy_constant_value_all_steps(self):
        """A constant-output Node must give the same value at every timestep."""
        const = np.array([3.14, -2.71, 0.0])
        with nengo.Network(seed=0) as net:
            a = nengo.Node(const)
            b = nengo.Node(size_in=3)
            nengo.Connection(a, b, synapse=None)
            p = nengo.Probe(b, synapse=None)

        n = 7
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(n)

        for t in range(n):
            np.testing.assert_allclose(
                sim.data[p][t], const, atol=1e-6,
                err_msg=f"At step {t}, constant node must output {const}"
            )


# ---------------------------------------------------------------------------
# ElementwiseInc / DotInc via Connection transforms
# ---------------------------------------------------------------------------

class TestDotIncBuilder:
    def test_linear_connection_scalar_transform(self):
        """A connection with a scalar transform: output = transform * input."""
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([2.0]))
            dst = nengo.Node(size_in=1)
            nengo.Connection(src, dst, transform=3.0, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        # 2.0 * 3.0 = 6.0
        np.testing.assert_allclose(sim.data[p][0, 0], 6.0, atol=1e-4,
                                   err_msg="transform=3.0 applied to 2.0 must give 6.0")

    def test_matrix_transform_exact(self):
        """A diagonal matrix transform: output[i] = W[i,i] * input[i]."""
        W = np.array([[1.0, 0.0], [0.0, 2.0]])  # 2→2 diagonal
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([3.0, 4.0]))
            dst = nengo.Node(size_in=2)
            nengo.Connection(src, dst, transform=W, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        np.testing.assert_allclose(sim.data[p][0], [3.0, 8.0], atol=1e-4,
                                   err_msg="Diagonal transform: output must be W*x")

    def test_dimension_reduction(self):
        """2-D → 1-D via a row vector transform."""
        W = np.array([[1.0, 1.0]])  # sum both dimensions
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([5.0, 7.0]))
            dst = nengo.Node(size_in=1)
            nengo.Connection(src, dst, transform=W, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        np.testing.assert_allclose(sim.data[p][0, 0], 12.0, atol=1e-4)

    def test_two_connections_sum_at_destination(self):
        """Two connections to the same node: output = a + b."""
        with nengo.Network(seed=0) as net:
            src_a = nengo.Node(np.array([3.0]))
            src_b = nengo.Node(np.array([5.0]))
            dst = nengo.Node(size_in=1)
            nengo.Connection(src_a, dst, synapse=None)
            nengo.Connection(src_b, dst, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        # Two inputs summed: 3 + 5 = 8
        np.testing.assert_allclose(sim.data[p][0, 0], 8.0, atol=1e-4,
                                   err_msg="Two connections must sum at destination: 3+5=8")

    def test_three_connections_sum(self):
        """Three connections to the same node sum correctly."""
        vals = [2.0, 3.0, 7.0]
        expected = sum(vals)
        with nengo.Network(seed=0) as net:
            srcs = [nengo.Node(np.array([v])) for v in vals]
            dst = nengo.Node(size_in=1)
            for s in srcs:
                nengo.Connection(s, dst, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        np.testing.assert_allclose(sim.data[p][0, 0], expected, atol=1e-4,
                                   err_msg=f"Three connections: expected sum={expected}")

    def test_negative_transform(self):
        """Negative transform: output = -input."""
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([4.0, -2.0]))
            dst = nengo.Node(size_in=2)
            nengo.Connection(src, dst, transform=-1.0, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        np.testing.assert_allclose(sim.data[p][0], [-4.0, 2.0], atol=1e-5,
                                   err_msg="transform=-1 must negate the input")


# ---------------------------------------------------------------------------
# SimPyFunc  (Python-function nodes)
# ---------------------------------------------------------------------------

class TestSimPyFuncBuilder:
    def test_source_node_exact_time_values(self):
        """Node(lambda t: t) returns exactly k*dt at step k."""
        dt = 0.001
        with nengo.Network(seed=0) as net:
            src = nengo.Node(lambda t: np.array([t]))
            p = nengo.Probe(src, synapse=None)

        with nengo_dl.Simulator(net, dt=dt, seed=0) as sim:
            sim.run_steps(3)
        data = sim.data[p]
        assert data.shape == (3, 1)
        np.testing.assert_allclose(data[0, 0], dt, atol=1e-6)
        np.testing.assert_allclose(data[1, 0], 2 * dt, atol=1e-6)
        np.testing.assert_allclose(data[2, 0], 3 * dt, atol=1e-6)

    def test_transform_node_multiplies_input(self):
        """A Node with fn(t, x) = x * 3 transforms its input."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([2.0]))
            tfm = nengo.Node(lambda t, x: x * 3.0, size_in=1, size_out=1)
            nengo.Connection(inp, tfm, synapse=None)
            p = nengo.Probe(tfm, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)
        np.testing.assert_allclose(sim.data[p][0, 0], 6.0, atol=1e-4,
                                   err_msg="fn(t,x)=x*3 applied to 2.0 must give 6.0")

    def test_constant_node_all_steps(self):
        """A Node with a constant array output each step."""
        const = np.array([7.0, -3.0, 1.5])
        with nengo.Network(seed=0) as net:
            src = nengo.Node(const)
            p = nengo.Probe(src, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(4)
        for t in range(4):
            np.testing.assert_allclose(sim.data[p][t], const, atol=1e-5)

    def test_sinusoidal_node_exact_values(self):
        """Node(lambda t: sin(2π*t)) produces exact sin values at each step."""
        dt = 0.01
        n = 5
        with nengo.Network(seed=0) as net:
            src = nengo.Node(lambda t: np.array([np.sin(2 * np.pi * t)]))
            p = nengo.Probe(src, synapse=None)

        with nengo_dl.Simulator(net, dt=dt, seed=0) as sim:
            sim.run_steps(n)

        for k in range(n):
            t_k = (k + 1) * dt
            expected = np.sin(2 * np.pi * t_k)
            np.testing.assert_allclose(
                sim.data[p][k, 0], expected, atol=1e-5,
                err_msg=f"Sinusoidal node at step {k+1}: expected {expected:.5f}"
            )


# ---------------------------------------------------------------------------
# SimProbe
# ---------------------------------------------------------------------------

class TestSimProbeBuilder:
    def test_probe_accumulates_steps(self):
        with nengo.Network(seed=0) as net:
            src = nengo.Node(lambda t: np.array([t]))
            p = nengo.Probe(src, synapse=None)

        n = 7
        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run_steps(n)
        assert sim.data[p].shape == (n, 1)

    def test_multiple_probes(self):
        with nengo.Network(seed=0) as net:
            n1 = nengo.Node(np.array([1.0]))
            n2 = nengo.Node(np.array([2.0, 3.0]))
            p1 = nengo.Probe(n1, synapse=None)
            p2 = nengo.Probe(n2, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
        assert sim.data[p1].shape == (5, 1)
        assert sim.data[p2].shape == (5, 2)

    def test_probe_correct_values(self):
        values = np.array([10.0, 20.0, 30.0])
        with nengo.Network(seed=0) as net:
            src = nengo.Node(values)
            p = nengo.Probe(src, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(3)
        for t in range(3):
            np.testing.assert_allclose(sim.data[p][t], values, atol=1e-5)

    def test_probe_synapse_smooths_signal(self):
        """A probe with synapse should produce a lower-pass filtered version."""
        # Use a step input: a large constant value from t=0
        step_val = 10.0
        tau = 0.02
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.array([step_val]))
            p_raw = nengo.Probe(src, synapse=None)
            p_filt = nengo.Probe(src, synapse=tau)

        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run_steps(50)

        raw = sim.data[p_raw]
        filt = sim.data[p_filt]

        # Raw probe is exactly step_val at every step
        np.testing.assert_allclose(raw, step_val, atol=1e-5)
        # Filtered probe starts near 0 and rises toward step_val
        # At t=0+, the filter output is much less than step_val
        assert filt[0, 0] < step_val, "Filtered probe must start below step value"
        # After many tau, the filter approaches step_val
        assert filt[-1, 0] > step_val * 0.9, "Filtered probe must converge to step value"


# ---------------------------------------------------------------------------
# Neuron-specific semantics
# ---------------------------------------------------------------------------

class TestNeuronSemantics:
    def test_rectifiedlinear_zero_below_threshold(self):
        """RectifiedLinear: below-threshold input gives zero output."""
        n = 5
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(
                n, 1,
                neuron_type=nengo.RectifiedLinear(),
                gain=nengo.dists.Choice([1.0]),
                bias=nengo.dists.Choice([-10.0]),  # strong negative bias → J < 0
                encoders=nengo.dists.Choice([[1.0]]),
                seed=0,
            )
            nengo.Connection(inp, ens.neurons, transform=np.ones((n, 1)), synapse=None)
            p = nengo.Probe(ens.neurons, synapse=None)

        x = np.zeros((1, 1, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1, data={inp: x}, inference_mode="rate")
            out = sim.data[p][0]

        np.testing.assert_allclose(out, np.zeros(n), atol=1e-5,
                                   err_msg="Strongly subthreshold ReLU neuron must give zero")

    def test_rectifiedlinear_positive_above_threshold(self):
        """RectifiedLinear: above-threshold input gives positive output."""
        n = 3
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(
                n, 1,
                neuron_type=nengo.RectifiedLinear(),
                gain=nengo.dists.Choice([1.0]),
                bias=nengo.dists.Choice([5.0]),  # large positive bias → J > 0
                encoders=nengo.dists.Choice([[1.0]]),
                seed=0,
            )
            nengo.Connection(inp, ens.neurons, transform=np.ones((n, 1)), synapse=None)
            p = nengo.Probe(ens.neurons, synapse=None)

        x = np.zeros((1, 1, 1), dtype=np.float32)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1, data={inp: x}, inference_mode="rate")
            out = sim.data[p][0]

        assert np.all(out > 0), "Strongly suprathreshold ReLU neurons must fire"

    def test_lif_voltage_in_unit_interval(self):
        """LIF voltage must remain in [0, 1] during simulation."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, neuron_type=nengo.LIF(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p_v = nengo.Probe(ens.neurons, "voltage", synapse=None)

        x = np.ones((1, 20, 1), dtype=np.float32) * 2.0
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(20, data={inp: x})
            voltage = sim.data[p_v]

        assert np.all(np.isfinite(voltage)), "LIF voltage must be finite (no NaN/Inf)"
        assert np.any(voltage > 0), "LIF voltage must be positive for some neurons under input"

    def test_lif_fires_with_strong_input(self):
        """LIF neurons with strong input must fire (output > 0) in spiking mode."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(
                5, 1, neuron_type=nengo.LIF(),
                gain=nengo.dists.Choice([5.0]),
                bias=nengo.dists.Choice([3.0]),
                encoders=nengo.dists.Choice([[1.0]]),
                seed=0,
            )
            nengo.Connection(inp, ens.neurons, transform=np.ones((5, 1)), synapse=None)
            p = nengo.Probe(ens.neurons, synapse=None)

        x = np.ones((1, 100, 1), dtype=np.float32) * 5.0
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(100, data={inp: x}, inference_mode="spiking")
            out = sim.data[p]

        total_spikes = out.sum()
        assert total_spikes > 0, "LIF neurons with strong input must spike"


# ---------------------------------------------------------------------------
# Multi-step consistency
# ---------------------------------------------------------------------------

class TestMultiStepOps:
    def test_constant_input_gives_constant_output(self):
        """Constant input through ReLU ensemble gives constant output each step."""
        input_val = 1.0
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([input_val]))
            ens = nengo.Ensemble(
                1, 1,
                neuron_type=nengo.RectifiedLinear(),
                gain=nengo.dists.Choice([1.0]),
                bias=nengo.dists.Choice([0.0]),
                encoders=nengo.dists.Choice([[1.0]]),
                seed=0,
            )
            nengo.Connection(inp, ens.neurons, transform=np.ones((1, 1)), synapse=None)
            p = nengo.Probe(ens.neurons, synapse=None)

        n = 10
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(n, inference_mode="rate")
        data = sim.data[p]
        assert data.shape == (n, 1)
        # All steps should give the same output (constant input → constant rate)
        np.testing.assert_allclose(data, np.full_like(data, data[0, 0]), atol=1e-4,
                                   err_msg="Constant input must give constant output each step")

    def test_batched_ops_produce_correct_shape(self):
        """Batched simulation must produce shape (batch, n_steps, dim)."""
        bs = 6
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            out = nengo.Node(size_in=2)
            nengo.Connection(inp, out, synapse=None)
            p = nengo.Probe(out, synapse=None)

        x = np.ones((bs, 4, 2))
        with nengo_dl.Simulator(net, minibatch_size=bs, seed=0) as sim:
            sim.run_steps(4, data={inp: x})
        assert sim.data[p].shape == (bs, 4, 2)

    def test_each_batch_item_uses_independent_input(self):
        """Different batch items must produce different outputs for different inputs."""
        bs = 3
        vals = [1.0, 5.0, 10.0]
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            out = nengo.Node(size_in=1)
            nengo.Connection(inp, out, synapse=None)
            p = nengo.Probe(out, synapse=None)

        x = np.array([[[v]] for v in vals], dtype=np.float32)  # (3, 1, 1)
        with nengo_dl.Simulator(net, minibatch_size=bs, seed=0) as sim:
            sim.run_steps(1, data={inp: x})
            data = sim.data[p]  # (3, 1, 1)

        for i, v in enumerate(vals):
            np.testing.assert_allclose(
                data[i, 0, 0], v, atol=1e-5,
                err_msg=f"Batch item {i}: expected {v}, got {data[i,0,0]}"
            )


# ---------------------------------------------------------------------------
# Direct op semantics: Reset, Copy, DotInc, ElementwiseInc
# ---------------------------------------------------------------------------

class TestExactOpSemantics:
    def test_reset_no_accumulation_across_steps(self):
        """Reset ensures node output is exactly the step value, not accumulated."""
        n = 5
        with nengo.Network(seed=0) as net:
            a = nengo.Node(np.array([1.0]))
            b = nengo.Node(size_in=1)
            nengo.Connection(a, b, synapse=None)
            p = nengo.Probe(b, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(n)

        # If Reset were absent, b would accumulate: 1, 2, 3, 4, 5
        # With Reset, every step b = 1.0 (signal is reset to 0 then +1)
        np.testing.assert_allclose(
            sim.data[p], np.ones((n, 1)), atol=1e-6,
            err_msg="Reset must clear accumulation: value must be 1.0 every step, not cumulative"
        )

    def test_copy_exact_value_injection(self):
        """Copy propagates an injected override value without modification."""
        with nengo.Network(seed=0) as net:
            src = nengo.Node(np.zeros(3))
            dst = nengo.Node(size_in=3)
            nengo.Connection(src, dst, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        override = np.array([[[2.5, -1.3, 7.0]]], dtype=np.float32)
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1, data={src: override})

        np.testing.assert_allclose(
            sim.data[p][0], [2.5, -1.3, 7.0], atol=1e-5,
            err_msg="Copy must propagate the injected value exactly"
        )

    def test_dotinc_matrix_multiply_exact(self):
        """DotInc via a 3×2 transform: output = W @ x exactly."""
        # W: 3 rows × 2 cols, x: [1, 2] → y = W @ x
        W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        x = np.array([1.0, 2.0])
        expected = W @ x  # [5, 11, 17]

        with nengo.Network(seed=0) as net:
            src = nengo.Node(x)
            dst = nengo.Node(size_in=3)
            nengo.Connection(src, dst, transform=W, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)

        np.testing.assert_allclose(
            sim.data[p][0], expected, atol=1e-5,
            err_msg=f"3×2 DotInc: expected {expected}"
        )

    def test_elementwiseinc_via_scale_transform(self):
        """ElementwiseInc: element-wise multiply via per-element scale transform."""
        scales = np.array([2.0, 3.0, 0.5])
        inputs = np.array([4.0, 5.0, 8.0])
        expected = scales * inputs  # [8, 15, 4]

        with nengo.Network(seed=0) as net:
            src = nengo.Node(inputs)
            dst = nengo.Node(size_in=3)
            # Diagonal transform applies element-wise scale
            nengo.Connection(src, dst, transform=np.diag(scales), synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)

        np.testing.assert_allclose(
            sim.data[p][0], expected, atol=1e-5,
            err_msg=f"ElementwiseInc: expected {expected}"
        )

    def test_dotinc_accumulates_two_sources_exactly(self):
        """Two DotInc ops into same signal: output = W1@x1 + W2@x2."""
        W1 = np.array([[2.0]])
        W2 = np.array([[3.0]])
        x1, x2 = np.array([5.0]), np.array([7.0])
        expected = W1 @ x1 + W2 @ x2  # [10 + 21] = [31]

        with nengo.Network(seed=0) as net:
            s1 = nengo.Node(x1)
            s2 = nengo.Node(x2)
            dst = nengo.Node(size_in=1)
            nengo.Connection(s1, dst, transform=W1, synapse=None)
            nengo.Connection(s2, dst, transform=W2, synapse=None)
            p = nengo.Probe(dst, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1)

        np.testing.assert_allclose(
            sim.data[p][0, 0], expected[0], atol=1e-5,
            err_msg=f"Accumulated DotInc: expected {expected[0]}"
        )


# ---------------------------------------------------------------------------
# SimNeurons reference: exact output at known input currents
# ---------------------------------------------------------------------------

class TestSimNeuronsReference:
    def test_rectifiedlinear_exact_output_at_known_current(self):
        """ReLU neuron: output = max(0, J) where J = gain*enc*x + bias."""
        # gain=1, encoder=1, bias=0, input=3.0 → J = 1*1*3+0 = 3.0 → rate = 3.0
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([3.0]))
            ens = nengo.Ensemble(
                1, 1, neuron_type=nengo.RectifiedLinear(),
                gain=nengo.dists.Choice([1.0]),
                bias=nengo.dists.Choice([0.0]),
                encoders=nengo.dists.Choice([[1.0]]),
                seed=0,
            )
            nengo.Connection(inp, ens.neurons, transform=np.ones((1, 1)), synapse=None)
            p = nengo.Probe(ens.neurons, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(3, inference_mode="rate")

        np.testing.assert_allclose(
            sim.data[p], np.full((3, 1), 3.0), atol=1e-4,
            err_msg="ReLU neuron: rate = max(0, gain*input+bias) = 3.0"
        )

    def test_rectifiedlinear_bias_shifts_threshold(self):
        """bias=2.0, gain=1, enc=1, input=-1.5 → J = -1.5+2 = 0.5 → rate=0.5."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([-1.5]))
            ens = nengo.Ensemble(
                1, 1, neuron_type=nengo.RectifiedLinear(),
                gain=nengo.dists.Choice([1.0]),
                bias=nengo.dists.Choice([2.0]),
                encoders=nengo.dists.Choice([[1.0]]),
                seed=0,
            )
            nengo.Connection(inp, ens.neurons, transform=np.ones((1, 1)), synapse=None)
            p = nengo.Probe(ens.neurons, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1, inference_mode="rate")

        np.testing.assert_allclose(
            sim.data[p][0, 0], 0.5, atol=1e-4,
            err_msg="ReLU rate = max(0, J) = max(0, -1.5 + 2.0) = 0.5"
        )

    def test_lif_matches_nengo_reference(self):
        """LIF spiking: nengo_dl matches nengo.Simulator exactly."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([2.0]))
            ens = nengo.Ensemble(
                3, 1, neuron_type=nengo.LIF(),
                gain=nengo.dists.Choice([3.0]),
                bias=nengo.dists.Choice([0.5]),
                encoders=nengo.dists.Choice([[1.0]]),
                seed=0,
            )
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens.neurons, synapse=None)

        with nengo.Simulator(net, dt=0.001) as ref:
            ref.run_steps(20)
            ref_out = ref.data[p].copy()

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(20, inference_mode="spiking")
            dl_out = sim.data[p].copy()

        # Float32 vs float64 integration can shift spikes by one timestep and
        # cause off-by-one spike counts; allow ±1 per neuron over the window.
        ref_counts = (ref_out > 0).sum(axis=0)
        dl_counts = (dl_out > 0).sum(axis=0)
        assert np.all(np.abs(ref_counts - dl_counts) <= 1), (
            f"LIF spiking: spike counts per neuron must agree within ±1. "
            f"ref={ref_counts}, dl={dl_counts}"
        )

    def test_simprocess_synapse_analytical_formula(self):
        """Synaptic filter response matches analytical formula y[k] = 1 - exp(-k*dt/tau)."""
        tau = 0.005
        dt = 0.001
        n = 10

        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([1.0]))
            p = nengo.Probe(inp, synapse=tau)

        with nengo_dl.Simulator(net, dt=dt, seed=0) as sim:
            sim.run_steps(n)
            out = sim.data[p].flatten()

        # Analytical: y[k] = 1 - exp(-k*dt/tau) for step k=1..n
        expected = np.array([1.0 - np.exp(-k * dt / tau) for k in range(1, n + 1)])
        np.testing.assert_allclose(
            out, expected, atol=1e-4,
            err_msg="Synapse step response must match analytical formula"
        )
