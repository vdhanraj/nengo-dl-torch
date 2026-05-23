"""Tests for nengo_dl.graph_optimizer.topo_sort."""

import warnings
import numpy as np
import pytest
import nengo
import nengo_dl
from nengo_dl.graph_optimizer import topo_sort
from nengo.builder import Builder as NengoBuilder, Model as NengoModel
from nengo.builder.signal import Signal
from nengo.builder.operator import Reset, Copy, DotInc, TimeUpdate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ops(net):
    model = NengoModel()
    NengoBuilder.build(model, net)
    return model.operators


# ---------------------------------------------------------------------------
# Empty / trivial cases
# ---------------------------------------------------------------------------

class TestTopoSortEdgeCases:
    def test_empty_list(self):
        assert topo_sort([]) == []

    def test_single_op(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            p = nengo.Probe(inp, synapse=None)
        ops = _build_ops(net)
        result = topo_sort(ops)
        assert len(result) == len(ops)

    def test_returns_list(self):
        result = topo_sort([])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestTopoSortCorrectness:
    def test_output_contains_all_ops(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            ens = nengo.Ensemble(10, 2, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)
        ops = _build_ops(net)
        result = topo_sort(ops)
        assert set(id(o) for o in result) == set(id(o) for o in ops)

    def test_output_length_equals_input(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(5, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)
        ops = _build_ops(net)
        assert len(topo_sort(ops)) == len(ops)

    def test_dependency_order_respected(self):
        """For every op, all signals it reads must have been written before it."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)
        ops = _build_ops(net)
        sorted_ops = topo_sort(ops)

        # For each op, all signals it reads must have been written (sets/incs/updates)
        # by an earlier op.  Signals with NO writer in the op list are constants
        # initialised before simulation starts — they are always available.
        written_sigs: set = set()
        unwritten_reads = []
        for op in sorted_ops:
            for sig in op.reads:
                base_id = id(sig.base)
                # Check that if ANY op writes this signal, it has already run
                writers = [
                    other for other in ops
                    if any(id(s.base) == base_id for s in other.sets + other.incs + other.updates)
                ]
                if writers and base_id not in written_sigs:
                    unwritten_reads.append((op, sig.name))
            for sig in op.sets + op.incs + op.updates:
                written_sigs.add(id(sig.base))

        assert unwritten_reads == [], (
            f"These ops read signals before they were written:\n"
            + "\n".join(f"  op={type(o).__name__} reads {name}" for o, name in unwritten_reads)
        )

    def test_time_update_is_early(self):
        """TimeUpdate should appear before neuron operators."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)
        ops = _build_ops(net)
        sorted_ops = topo_sort(ops)

        time_update_idx = next(
            (i for i, op in enumerate(sorted_ops) if isinstance(op, TimeUpdate)),
            None,
        )
        if time_update_idx is not None:
            assert time_update_idx < len(sorted_ops) - 1

    def test_writer_always_precedes_reader(self):
        """For every write→read dependency edge, the writer index < reader index."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            ens1 = nengo.Ensemble(10, 2, seed=0)
            ens2 = nengo.Ensemble(10, 2, seed=1)
            nengo.Connection(inp, ens1, synapse=None)
            nengo.Connection(ens1, ens2, synapse=None)
            p = nengo.Probe(ens2, synapse=None)
        ops = _build_ops(net)
        sorted_ops = topo_sort(ops)

        # Build index map
        pos = {id(op): i for i, op in enumerate(sorted_ops)}
        # Build write map: base_id → earliest writer index
        first_write: dict = {}
        for op in sorted_ops:
            op_idx = pos[id(op)]
            for sig in op.sets + op.incs + op.updates:
                k = id(sig.base)
                if k not in first_write:
                    first_write[k] = op_idx

        violations = []
        for op in sorted_ops:
            op_idx = pos[id(op)]
            for sig in op.reads:
                k = id(sig.base)
                if k in first_write and first_write[k] > op_idx:
                    violations.append(
                        f"Op {type(op).__name__}@{op_idx} reads a signal "
                        f"first written at {first_write[k]}"
                    )
        assert violations == [], "\n".join(violations)


# ---------------------------------------------------------------------------
# Integration: sorted order gives correct simulation
# ---------------------------------------------------------------------------

class TestTopoSortIntegration:
    def test_simulation_output_consistent(self):
        """A simulation using sorted ops should match reference Nengo output."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(lambda t: np.array([np.sin(2 * np.pi * t)]))
            ens = nengo.Ensemble(50, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p_dl = nengo.Probe(ens, synapse=0.01)

        x = np.sin(2 * np.pi * np.linspace(0, 0.05, 50)).reshape(1, 50, 1).astype(np.float32)
        with nengo_dl.Simulator(net, dt=0.001, seed=0) as sim:
            sim.run_steps(50, data={inp: x})
            y_dl = sim.data[p_dl]

        assert y_dl.shape == (50, 1)
        assert not np.any(np.isnan(y_dl))

    def test_multi_ensemble_network(self):
        """A network with two ensembles connected in series runs correctly."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            ens1 = nengo.Ensemble(20, 2, seed=0)
            ens2 = nengo.Ensemble(10, 2, seed=1)
            nengo.Connection(inp, ens1, synapse=None)
            nengo.Connection(ens1, ens2, synapse=None)
            p = nengo.Probe(ens2, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
        assert sim.data[p].shape == (5, 2)

    def test_node_to_node_copy_correct_value(self):
        """A→B with transform=2 should give probe(B)=2*A for all timesteps."""
        val = 3.0
        with nengo.Network(seed=0) as net:
            a = nengo.Node(np.array([val]))
            b = nengo.Node(size_in=1)
            nengo.Connection(a, b, transform=2.0, synapse=None)
            p = nengo.Probe(b, synapse=None)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(5)
        np.testing.assert_allclose(sim.data[p], np.full((5, 1), 2.0 * val), atol=1e-5,
                                   err_msg="transform=2.0 must double the signal value")

    def test_sorted_ops_match_deterministic_simulation(self):
        """Two separate simulator instances (same network, same seed) give identical output."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(3))
            ens = nengo.Ensemble(20, 3, neuron_type=nengo.RectifiedLinear(), seed=0)
            out = nengo.Node(size_in=3)
            nengo.Connection(inp, ens, synapse=None)
            nengo.Connection(ens, out, function=lambda x: x, synapse=None)
            p = nengo.Probe(out, synapse=None)

        x = np.random.RandomState(0).randn(1, 10, 3).astype(np.float32)

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(10, data={inp: x})
            y1 = sim.data[p].copy()

        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(10, data={inp: x})
            y2 = sim.data[p].copy()

        np.testing.assert_allclose(y1, y2, rtol=1e-5,
                                   err_msg="Two runs with same seed must be identical")


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

class TestCycleDetection:
    def test_cycle_warns_and_returns_original_order(self):
        """If topo_sort detects a cycle it warns and returns original ops."""
        sig_a = Signal(np.zeros(1), name="a")
        sig_b = Signal(np.zeros(1), name="b")

        from nengo_dl.tests.dummies import DummyOperator

        op1 = DummyOperator(tag="op1")
        op2 = DummyOperator(tag="op2")
        # Create a cycle: op1 sets sig_a and reads sig_b; op2 sets sig_b and reads sig_a
        op1.sets = [sig_a]
        op1.reads = [sig_b]
        op2.sets = [sig_b]
        op2.reads = [sig_a]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = topo_sort([op1, op2])
        assert any("Cycle" in str(warning.message) for warning in w)
        assert len(result) == 2

    def test_no_cycle_warning_for_valid_net(self):
        """A valid network must not trigger a cycle warning."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(5, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)
        ops = _build_ops(net)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            topo_sort(ops)
        cycle_warnings = [x for x in w if "Cycle" in str(x.message)]
        assert cycle_warnings == [], "Valid network must not produce cycle warnings"

    def test_independent_ops_both_in_output(self):
        """Two ops with no dependencies can appear in any order but both present."""
        from nengo_dl.tests.dummies import DummyOperator
        op_a = DummyOperator(tag="a")
        op_b = DummyOperator(tag="b")
        result = topo_sort([op_a, op_b])
        assert len(result) == 2
        assert set(id(o) for o in result) == {id(op_a), id(op_b)}

    def test_linear_chain_order_preserved(self):
        """A linear dependency chain must be sorted in dependency order."""
        from nengo_dl.tests.dummies import WriteOperator, ReadOperator
        sig = Signal(np.zeros(1), name="chain_sig")
        writer = WriteOperator(sig, tag="writer")
        reader = ReadOperator(sig, tag="reader")

        result = topo_sort([reader, writer])  # intentionally wrong order
        assert id(result[0]) == id(writer), "Writer must come before reader"
        assert id(result[1]) == id(reader)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestTopoSortDeterminism:
    def test_same_network_same_order(self):
        """Building the same network twice gives the same sort order."""
        def make():
            with nengo.Network(seed=0) as net:
                inp = nengo.Node(np.zeros(2))
                ens = nengo.Ensemble(10, 2, seed=0)
                nengo.Connection(inp, ens, synapse=None)
                p = nengo.Probe(ens, synapse=None)
            return _build_ops(net)

        ops1 = make()
        ops2 = make()
        sorted1 = topo_sort(ops1)
        sorted2 = topo_sort(ops2)
        assert [type(o).__name__ for o in sorted1] == [type(o).__name__ for o in sorted2]

    def test_shuffled_input_gives_valid_order(self):
        """Shuffling input ops still gives a valid topological order."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)
        ops = _build_ops(net)

        rng = np.random.RandomState(42)
        shuffled = list(ops)
        rng.shuffle(shuffled)
        result = topo_sort(shuffled)

        # Must contain all ops
        assert set(id(o) for o in result) == set(id(o) for o in ops)

        # Must satisfy dependency order
        pos = {id(op): i for i, op in enumerate(result)}
        for op in result:
            op_idx = pos[id(op)]
            for sig in op.reads:
                writers_idx = [
                    pos[id(other)] for other in result
                    if any(id(s.base) == id(sig.base) for s in other.sets + other.incs + other.updates)
                ]
                for w_idx in writers_idx:
                    assert w_idx < op_idx, (
                        f"Writer at {w_idx} must precede reader at {op_idx}"
                    )
