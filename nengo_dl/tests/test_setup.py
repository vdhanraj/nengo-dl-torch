"""Tests for package structure, imports, and public API surface."""

import importlib
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------

class TestTopLevelImports:
    def test_import_nengo_dl(self):
        import nengo_dl
        assert isinstance(nengo_dl, types.ModuleType)

    def test_version_attribute(self):
        import nengo_dl
        assert hasattr(nengo_dl, "__version__")
        assert isinstance(nengo_dl.__version__, str)
        assert len(nengo_dl.__version__) > 0

    def test_simulator_importable(self):
        from nengo_dl import Simulator
        assert Simulator is not None

    def test_torch_node_importable(self):
        from nengo_dl import TorchNode
        assert TorchNode is not None

    def test_configure_settings_importable(self):
        from nengo_dl import configure_settings
        assert callable(configure_settings)


# ---------------------------------------------------------------------------
# Sub-module imports
# ---------------------------------------------------------------------------

class TestSubModuleImports:
    def test_simulator_module(self):
        import nengo_dl.simulator
        assert hasattr(nengo_dl.simulator, "Simulator")

    def test_tensor_graph_module(self):
        import nengo_dl.tensor_graph
        assert hasattr(nengo_dl.tensor_graph, "TensorGraph")

    def test_op_builders_module(self):
        import nengo_dl.op_builders
        assert hasattr(nengo_dl.op_builders, "ResetBuilder")

    def test_neurons_module(self):
        import nengo_dl.neurons
        assert hasattr(nengo_dl.neurons, "SoftLIFRate")

    def test_losses_module(self):
        import nengo_dl.losses
        assert hasattr(nengo_dl.losses, "MSELoss")

    def test_utils_module(self):
        import nengo_dl.utils
        assert hasattr(nengo_dl.utils, "to_numpy")

    def test_builder_module(self):
        import nengo_dl.builder
        assert hasattr(nengo_dl.builder, "Builder")

    def test_config_module(self):
        import nengo_dl.config
        assert hasattr(nengo_dl.config, "configure_settings")

    def test_converter_module(self):
        import nengo_dl.converter
        assert hasattr(nengo_dl.converter, "Converter")

    def test_graph_optimizer_module(self):
        import nengo_dl.graph_optimizer
        assert hasattr(nengo_dl.graph_optimizer, "topo_sort")


# ---------------------------------------------------------------------------
# Public API completeness
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_simulator_class_present(self):
        import nengo_dl
        assert hasattr(nengo_dl, "Simulator")

    def test_torch_node_class_present(self):
        import nengo_dl
        assert hasattr(nengo_dl, "TorchNode")

    def test_configure_settings_present(self):
        import nengo_dl
        assert hasattr(nengo_dl, "configure_settings")

    def test_layer_present(self):
        from nengo_dl import Layer
        assert Layer is not None

    def test_no_import_errors_on_fresh_import(self):
        """Re-importing nengo_dl should not raise."""
        importlib.reload(importlib.import_module("nengo_dl"))


# ---------------------------------------------------------------------------
# Dependencies reachable
# ---------------------------------------------------------------------------

class TestDependencies:
    def test_torch_available(self):
        import torch
        assert torch.__version__

    def test_nengo_available(self):
        import nengo
        assert nengo.__version__

    def test_numpy_available(self):
        import numpy as np
        assert np.__version__

    def test_torch_version_string(self):
        import torch
        parts = torch.__version__.split(".")
        assert int(parts[0]) >= 1, "PyTorch >= 1.x required"
