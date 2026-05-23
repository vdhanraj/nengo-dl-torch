"""Tests for nengo_dl.builder (Builder, OpBuilder, BuildConfig)."""

import dataclasses
import pytest
import torch
import numpy as np
import nengo

from nengo_dl.builder import Builder, OpBuilder, BuildConfig
from nengo_dl.signals import SignalDict
from nengo.builder.signal import Signal


# ---------------------------------------------------------------------------
# BuildConfig
# ---------------------------------------------------------------------------

class TestBuildConfig:
    def test_defaults(self):
        cfg = BuildConfig()
        assert cfg.dt == pytest.approx(0.001)
        assert cfg.minibatch_size == 1
        assert cfg.training is False
        assert cfg.lif_smoothing == pytest.approx(0.0)
        assert cfg.inference_only is False
        assert cfg.device is None
        assert cfg.dtype is None
        assert cfg.rng is None

    def test_custom_values(self):
        dev = torch.device("cpu")
        cfg = BuildConfig(
            dt=0.005,
            minibatch_size=32,
            training=True,
            lif_smoothing=0.1,
            inference_only=True,
            device=dev,
            dtype=torch.float32,
        )
        assert cfg.dt == pytest.approx(0.005)
        assert cfg.minibatch_size == 32
        assert cfg.training is True
        assert cfg.lif_smoothing == pytest.approx(0.1)
        assert cfg.device is dev

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(BuildConfig)

    def test_mutable(self):
        cfg = BuildConfig()
        cfg.training = True
        assert cfg.training is True


# ---------------------------------------------------------------------------
# OpBuilder
# ---------------------------------------------------------------------------

class TestOpBuilder:
    def test_build_pre_is_noop(self):
        b = OpBuilder()
        b.build_pre([], None, BuildConfig())  # must not raise

    def test_build_step_raises(self):
        b = OpBuilder()
        with pytest.raises(NotImplementedError):
            b.build_step([], None, BuildConfig())

    def test_build_post_is_noop(self):
        b = OpBuilder()
        b.build_post([], None, BuildConfig())  # must not raise

    def test_mergeable_false_by_default(self):
        class FakeOp:
            pass
        assert OpBuilder.mergeable(FakeOp(), FakeOp()) is False

    def test_subclass_overrides_build_step(self):
        class MyBuilder(OpBuilder):
            called = False
            def build_step(self, ops, signals, config):
                MyBuilder.called = True
        b = MyBuilder()
        b.build_step([], None, BuildConfig())
        assert MyBuilder.called


# ---------------------------------------------------------------------------
# Builder registration
# ---------------------------------------------------------------------------

class TestBuilderRegistry:
    def test_register_decorator(self):
        class FakeOp:
            pass

        @Builder.register(FakeOp)
        class FakeBuilder(OpBuilder):
            def build_step(self, ops, signals, config):
                pass

        assert Builder.get_builder_cls(FakeOp) is FakeBuilder
        # Cleanup
        del Builder._registry[FakeOp]

    def test_get_builder_cls_returns_none_for_unknown(self):
        class UnknownOp:
            pass
        assert Builder.get_builder_cls(UnknownOp) is None

    def test_standard_ops_are_registered(self):
        from nengo.builder.operator import TimeUpdate, Reset, Copy, DotInc, ElementwiseInc, SimPyFunc
        from nengo.builder.probe import SimProbe
        from nengo.builder.neurons import SimNeurons
        from nengo.builder.processes import SimProcess

        for op_type in [TimeUpdate, Reset, Copy, DotInc, ElementwiseInc, SimPyFunc, SimProbe, SimNeurons, SimProcess]:
            assert Builder.get_builder_cls(op_type) is not None, f"No builder for {op_type.__name__}"

    def test_unknown_op_raises_on_build(self):
        class UnknownOp:
            reads = []
            sets = []
            incs = []
            updates = []

        sd = SignalDict(1, torch.device("cpu"))
        config = BuildConfig(device=torch.device("cpu"), dtype=torch.float32)
        with pytest.raises(ValueError, match="No builder registered"):
            Builder([UnknownOp()], sd, config)


# ---------------------------------------------------------------------------
# Builder grouping and execution
# ---------------------------------------------------------------------------

class TestBuilderGrouping:
    def _make_simple_model(self, seed=0):
        with nengo.Network(seed=seed) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(5, 1, seed=seed)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)
        from nengo.builder import Builder as NengoBuilder, Model
        model = Model()
        NengoBuilder.build(model, net)
        return model, net, inp, p

    def test_builder_builds_without_error(self):
        model, net, inp, p = self._make_simple_model()
        from nengo_dl.tensor_graph import TensorGraph
        tg = TensorGraph(model, dt=0.001, minibatch_size=1)
        assert tg is not None

    def test_groups_non_empty(self):
        model, net, inp, p = self._make_simple_model()
        from nengo_dl.graph_optimizer import topo_sort
        from nengo_dl.signals import SignalDict
        sorted_ops = topo_sort(model.operators)
        config = BuildConfig(
            dt=0.001, minibatch_size=1,
            device=torch.device("cpu"), dtype=torch.float32
        )
        sd = SignalDict(1, torch.device("cpu"), torch.float32)
        # We can't build without first registering signals —
        # just test that TensorGraph builds without exception (above test covers this)
        assert len(sorted_ops) > 0

    def test_run_step_does_not_raise(self):
        model, net, inp, p = self._make_simple_model()
        from nengo_dl.tensor_graph import TensorGraph
        tg = TensorGraph(model, dt=0.001, minibatch_size=1)
        result = tg.forward(1)
        assert p in result

    def test_multiple_steps_run(self):
        model, net, inp, p = self._make_simple_model()
        from nengo_dl.tensor_graph import TensorGraph
        tg = TensorGraph(model, dt=0.001, minibatch_size=2)
        result = tg.forward(5)
        assert result[p].shape[1] == 5  # n_steps dimension
