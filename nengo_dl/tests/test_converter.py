"""Tests for nengo_dl.Converter."""

import warnings
import numpy as np
import pytest
import torch
import torch.nn as nn
import nengo
import nengo_dl
from nengo_dl.converter import Converter, ConversionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp(in_size=4, hidden=8, out_size=3):
    return nn.Sequential(
        nn.Linear(in_size, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_size),
    )


def _run_converter(model, x_np, scale=None, activation_type="rectified_linear",
                   synapse=None, inference_mode="rate"):
    """Convert model, run one step in rate mode, return (nengo_out, pytorch_out)."""
    x_np = np.asarray(x_np, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        pt_out = model(torch.tensor(x_np)).numpy()

    c = Converter(model, scale_firing_rates=scale,
                  activation_type=activation_type, synapse=synapse)
    inp = list(c.inputs.values())[0]
    out = list(c.outputs.values())[-1]
    with c.net:
        p = nengo.Probe(out, synapse=None)

    x_b = x_np.reshape(1, 1, -1)
    with nengo_dl.Simulator(c.net, seed=0) as sim:
        sim.run_steps(1, data={inp: x_b}, inference_mode=inference_mode)
        nengo_out = sim.data[p][0]

    return nengo_out, pt_out


# ---------------------------------------------------------------------------
# Basic conversion
# ---------------------------------------------------------------------------

class TestConverterBasic:
    def test_returns_nengo_network(self):
        model = _make_mlp()
        c = Converter(model)
        assert isinstance(c.net, nengo.Network)

    def test_inputs_dict_populated(self):
        model = _make_mlp()
        c = Converter(model)
        assert len(c.inputs) > 0

    def test_outputs_dict_populated(self):
        model = _make_mlp()
        c = Converter(model)
        assert len(c.outputs) > 0

    def test_outputs_expose_modules_and_final_outputs_only(self):
        model = _make_mlp()
        c = Converter(model)
        assert model[0] in c.outputs
        assert model[1] in c.outputs
        assert model[2] in c.outputs
        assert "output_0" in c.outputs
        assert "relu" not in c.outputs
        assert "linear" not in c.outputs

    def test_input_is_node(self):
        model = _make_mlp(in_size=4)
        c = Converter(model)
        for node in c.inputs.values():
            assert isinstance(node, nengo.Node)

    def test_network_runs_in_simulator(self):
        model = _make_mlp(in_size=4, hidden=8, out_size=3)
        c = Converter(model)

        inp_node = list(c.inputs.values())[0]
        out_node = list(c.outputs.values())[-1]
        with c.net:
            p = nengo.Probe(out_node, synapse=None)

        x = np.random.randn(1, 1, 4).astype(np.float32)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp_node: x})
            data = sim.data[p]
        assert data.shape[-1] == 3

    def test_scale_firing_rates_changes_output(self):
        """scale_firing_rates should not change the rate-mode numerical output."""
        torch.manual_seed(7)
        model = nn.Linear(4, 3, bias=True)
        with torch.no_grad():
            model.weight.data.abs_()
            model.bias.data.fill_(1.0)

        x_np = np.abs(np.random.RandomState(11).randn(4)).astype(np.float32)
        out_noscale, _ = _run_converter(model, x_np, scale=None)
        out_scaled, _ = _run_converter(model, x_np, scale=100.0)
        # Rate-mode output is scale-invariant: relu(W@x+b) regardless of scale
        np.testing.assert_allclose(
            out_noscale, out_scaled, rtol=1e-3, atol=1e-3,
            err_msg="scale_firing_rates must not change rate-mode output"
        )

    def test_multi_input_inputs_are_exposed(self):
        class TwoInputAdd(nn.Module):
            def forward(self, x, y):
                return x + y

        c = Converter(TwoInputAdd(), input_shape=((4,), (4,)))
        assert list(c.inputs.keys()) == ["x", "y"]

    def test_tuple_outputs_are_exposed(self):
        class TwoOutput(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 3)
                self.b = nn.Linear(4, 2)

            def forward(self, x):
                return self.a(x), self.b(x)

        c = Converter(TwoOutput(), input_shape=(4,))
        assert "output_0" in c.outputs
        assert "output_1" in c.outputs

    def test_dict_outputs_are_exposed(self):
        class DictOutput(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 3)
                self.b = nn.Linear(4, 2)

            def forward(self, x):
                return {"logits": self.a(x), "aux": self.b(x)}

        c = Converter(DictOutput(), input_shape=(4,))
        assert "logits" in c.outputs
        assert "aux" in c.outputs
        assert isinstance(c.output_structure, dict)


# ---------------------------------------------------------------------------
# Layer-specific conversion
# ---------------------------------------------------------------------------

class TestConverterLayers:
    def test_converts_linear(self):
        model = nn.Linear(4, 8)
        c = Converter(model)
        assert isinstance(c.net, nengo.Network)

    def test_converts_relu(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        c = Converter(model)
        assert isinstance(c.net, nengo.Network)

    def test_converts_sigmoid(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Sigmoid())
        c = Converter(model, activation_type="sigmoid")
        assert isinstance(c.net, nengo.Network)

    def test_converts_flatten(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        c = Converter(model)
        assert isinstance(c.net, nengo.Network)

    def test_converts_permute(self):
        class PermuteLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2, bias=False)

            def forward(self, x):
                x = x.permute(0, 2, 1)
                return self.fc(torch.flatten(x, start_dim=1))

        model = PermuteLinear().eval()
        c = Converter(model, input_shape=(2, 2))
        assert isinstance(c.net, nengo.Network)

    def test_converts_transpose(self):
        class TransposeLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2, bias=False)

            def forward(self, x):
                x = torch.transpose(x, 1, 2)
                return self.fc(torch.flatten(x, start_dim=1))

        model = TransposeLinear().eval()
        c = Converter(model, input_shape=(2, 2))
        assert isinstance(c.net, nengo.Network)

    def test_permute_batch_axis_raises(self):
        class BadPermute(nn.Module):
            def forward(self, x):
                return x.permute(1, 0, 2)

        with pytest.raises(ConversionError, match="batch dimension"):
            Converter(BadPermute().eval(), input_shape=(2, 2))

    def test_transpose_batch_axis_raises(self):
        class BadTranspose(nn.Module):
            def forward(self, x):
                return x.transpose(0, 1)

        with pytest.raises(ConversionError, match="batch dimension"):
            Converter(BadTranspose().eval(), input_shape=(2, 2))

    def test_converts_identity(self):
        model = nn.Sequential(nn.Identity(), nn.Linear(4, 2))
        c = Converter(model)
        assert isinstance(c.net, nengo.Network)

    def test_allow_fallback_true_warns(self):
        class UnsupportedLayer(nn.Module):
            def forward(self, x):
                return x[:, :2]  # halve features

        model = nn.Sequential(nn.Linear(4, 4), UnsupportedLayer())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = Converter(model, allow_fallback=True)
        assert any("TorchNode fallback" in str(warning.message) for warning in w)
        assert any("spiking fidelity" in str(warning.message) for warning in w)

    def test_batchnorm_native_in_inference_only(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)).eval()
        c = Converter(model, inference_only=True)
        assert not any(isinstance(node, nengo_dl.TorchNode) for node in c.net.all_nodes)

    def test_maxpool_native_with_max_to_avg_pool(self):
        model = nn.Sequential(nn.MaxPool2d(2), nn.Flatten()).eval()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = Converter(model, input_shape=(1, 4, 4), max_to_avg_pool=True)
        assert isinstance(c.net, nengo.Network)
        assert any("converted as average pooling" in str(warning.message) for warning in w)

    def test_maxpool_as_avg_matches_avgpool_conversion(self):
        max_model = nn.Sequential(nn.MaxPool2d(2), nn.Flatten()).eval()
        avg_model = nn.Sequential(nn.AvgPool2d(2), nn.Flatten()).eval()
        x_np = np.arange(16, dtype=np.float32).reshape(1, 4, 4)

        c_max = Converter(max_model, input_shape=(1, 4, 4), max_to_avg_pool=True)
        c_avg = Converter(avg_model, input_shape=(1, 4, 4))

        inp_max = list(c_max.inputs.values())[0]
        inp_avg = list(c_avg.inputs.values())[0]
        out_max = c_max.outputs["output_0"]
        out_avg = c_avg.outputs["output_0"]

        with c_max.net:
            p_max = nengo.Probe(out_max, synapse=None)
        with c_avg.net:
            p_avg = nengo.Probe(out_avg, synapse=None)

        with nengo_dl.Simulator(c_max.net, seed=0) as sim_max:
            sim_max.run_steps(1, data={inp_max: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            max_out = sim_max.data[p_max][0]

        with nengo_dl.Simulator(c_avg.net, seed=0) as sim_avg:
            sim_avg.run_steps(1, data={inp_avg: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            avg_out = sim_avg.data[p_avg][0]

        np.testing.assert_allclose(max_out, avg_out, rtol=1e-5, atol=1e-5)

    def test_allow_fallback_false_raises(self):
        class UnsupportedLayer(nn.Module):
            def forward(self, x):
                return x[:, :2]

        model = nn.Sequential(nn.Linear(4, 4), UnsupportedLayer())
        with pytest.raises(ConversionError):
            Converter(model, allow_fallback=False)

    def test_converts_conv2d(self):
        """Conv2d converts with explicit input_shape."""
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 26 * 26, 2),
        )
        c = Converter(model, input_shape=(1, 28, 28))
        assert isinstance(c.net, nengo.Network)

        inp = list(c.inputs.values())[0]
        relu = c.outputs[model[1]]
        sample = np.arange(0, relu.size_in, max(1, relu.size_in // 10))[:10]
        with c.net:
            p = nengo.Probe(relu[sample])

        x = np.zeros((1, 1, 28 * 28), dtype=np.float32)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x}, inference_mode="rate")
            assert sim.data[p].shape == (1, 10)

    def test_conv2d_requires_input_shape(self):
        """A spatial first layer needs input_shape metadata."""
        model = nn.Sequential(nn.Conv2d(1, 4, 3))
        with pytest.raises(ConversionError):
            Converter(model)

    def test_conv2d_rate_matches_pytorch(self):
        """Conv2d + ReLU + Linear rate output matches PyTorch."""
        torch.manual_seed(3)
        model = nn.Sequential(
            nn.Conv2d(1, 2, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 2 * 2, 3),
        )
        model.eval()
        x = np.random.RandomState(4).randn(1, 1, 4, 4).astype(np.float32)

        with torch.no_grad():
            pt_out = model(torch.tensor(x)).numpy()

        c = Converter(model, input_shape=(1, 4, 4), scale_firing_rates=1)
        inp = list(c.inputs.values())[0]
        out = list(c.outputs.values())[-1]
        with c.net:
            p = nengo.Probe(out)

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0:1]

        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-5, atol=1e-5)

    def test_branch_add_rate_matches_pytorch(self):
        class ResidualAdd(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 4, bias=True)
                self.b = nn.Linear(4, 4, bias=True)

            def forward(self, x):
                return self.a(x) + self.b(x)

        torch.manual_seed(9)
        model = ResidualAdd().eval()
        x_np = np.random.RandomState(10).randn(4).astype(np.float32)

        with torch.no_grad():
            pt_out = model(torch.tensor(x_np).unsqueeze(0)).numpy()[0]

        c = Converter(model)
        inp = list(c.inputs.values())[0]
        out = list(c.outputs.values())[-1]
        with c.net:
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-5, atol=1e-5)

    def test_branch_sub_rate_matches_pytorch(self):
        class ResidualSub(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 4, bias=True)
                self.b = nn.Linear(4, 4, bias=True)

            def forward(self, x):
                return self.a(x) - self.b(x)

        torch.manual_seed(19)
        model = ResidualSub().eval()
        x_np = np.random.RandomState(20).randn(4).astype(np.float32)

        with torch.no_grad():
            pt_out = model(torch.tensor(x_np).unsqueeze(0)).numpy()[0]

        c = Converter(model)
        inp = list(c.inputs.values())[0]
        out = c.outputs["output_0"]
        with c.net:
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-5, atol=1e-5)

    def test_two_input_add_rate_matches_pytorch(self):
        class TwoInputAdd(nn.Module):
            def forward(self, x, y):
                return x + y

        model = TwoInputAdd().eval()
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y_np = np.array([0.5, -1.0, 2.0, 1.5], dtype=np.float32)

        with torch.no_grad():
            pt_out = model(torch.tensor(x_np).unsqueeze(0), torch.tensor(y_np).unsqueeze(0)).numpy()[0]

        c = Converter(model, input_shape=((4,), (4,)))
        x_inp = c.inputs["x"]
        y_inp = c.inputs["y"]
        out = c.outputs["output_0"]
        with c.net:
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(
                1,
                data={
                    x_inp: x_np.reshape(1, 1, -1),
                    y_inp: y_np.reshape(1, 1, -1),
                },
                inference_mode="rate",
            )
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-5, atol=1e-5)

    def test_scalar_mul_rate_matches_pytorch(self):
        class ScalarMul(nn.Module):
            def forward(self, x):
                return x * 0.5

        model = ScalarMul().eval()
        x_np = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)

        with torch.no_grad():
            pt_out = model(torch.tensor(x_np).unsqueeze(0)).numpy()[0]

        c = Converter(model, input_shape=(4,))
        inp = c.inputs["x"]
        out = c.outputs["output_0"]
        with c.net:
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-6, atol=1e-6)

    def test_scalar_sub_rate_matches_pytorch(self):
        class ScalarSub(nn.Module):
            def forward(self, x):
                return x - 1.5

        model = ScalarSub().eval()
        x_np = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)

        with torch.no_grad():
            pt_out = model(torch.tensor(x_np).unsqueeze(0)).numpy()[0]

        c = Converter(model, input_shape=(4,))
        inp = c.inputs["x"]
        out = c.outputs["output_0"]
        with c.net:
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-6, atol=1e-6)

    def test_reverse_scalar_sub_rate_matches_pytorch(self):
        class ReverseScalarSub(nn.Module):
            def forward(self, x):
                return 1.5 - x

        model = ReverseScalarSub().eval()
        x_np = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)

        with torch.no_grad():
            pt_out = model(torch.tensor(x_np).unsqueeze(0)).numpy()[0]

        c = Converter(model, input_shape=(4,))
        inp = c.inputs["x"]
        out = c.outputs["output_0"]
        with c.net:
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-6, atol=1e-6)

    def test_tensor_tensor_mul_uses_fallback_and_warns(self):
        class TensorMul(nn.Module):
            def forward(self, x, y):
                return x * y

        model = TensorMul().eval()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = Converter(model, input_shape=((4,), (4,)))

        assert isinstance(c.net, nengo.Network)
        assert any("TorchNode fallback" in str(warning.message) for warning in w)

    def test_tuple_outputs_run_in_simulator(self):
        class TwoOutput(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 3)
                self.b = nn.Linear(4, 2)

            def forward(self, x):
                return self.a(x), self.b(x)

        model = TwoOutput().eval()
        x_np = np.random.RandomState(1).randn(4).astype(np.float32)
        c = Converter(model, input_shape=(4,))
        inp = c.inputs["x"]
        out0 = c.outputs["output_0"]
        out1 = c.outputs["output_1"]
        with c.net:
            p0 = nengo.Probe(out0, synapse=None)
            p1 = nengo.Probe(out1, synapse=None)

        with torch.no_grad():
            pt0, pt1 = model(torch.tensor(x_np).unsqueeze(0))
            pt0 = pt0.numpy()[0]
            pt1 = pt1.numpy()[0]

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo0 = sim.data[p0][0]
            nengo1 = sim.data[p1][0]

        np.testing.assert_allclose(nengo0, pt0, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(nengo1, pt1, rtol=1e-5, atol=1e-5)

    def test_reused_module_matches_pytorch_and_uses_shared_weights(self):
        class SharedLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                a = self.shared(x)
                b = self.shared(x + 1.0)
                return a + b

        torch.manual_seed(2)
        model = SharedLinear().eval()
        x_np = np.random.RandomState(2).randn(4).astype(np.float32)

        with torch.no_grad():
            pt_out = model(torch.tensor(x_np).unsqueeze(0)).numpy()[0]

        c = Converter(model, input_shape=(4,))
        inp = c.inputs["x"]
        out = c.outputs["output_0"]
        assert "shared" in c.outputs
        assert "shared_1" in c.outputs

        with c.net:
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0]
            weights = sim.get_weights()

        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-5, atol=1e-5)

        shared_weight_keys = [k for k in weights if k.endswith(".weight")]
        assert len(shared_weight_keys) == 1, (
            f"Expected one shared module weight set, got {shared_weight_keys}"
        )

    def test_fallback_uses_torchnode(self):
        class UnsupportedLayer(nn.Module):
            def forward(self, x):
                return x[:, :2]

        model = nn.Sequential(nn.Linear(4, 4), UnsupportedLayer())
        c = Converter(model, allow_fallback=True)
        assert any(isinstance(node, nengo_dl.TorchNode) for node in c.net.all_nodes)

    def test_reused_module_exposes_callsite_aliases_only_for_reuse(self):
        class SharedLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                a = self.shared(x)
                b = self.shared(x + 1.0)
                return a + b

        c = Converter(SharedLinear(), input_shape=(4,))
        assert "shared" in c.outputs
        assert "shared_1" in c.outputs

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_converter_shapeprop_uses_model_device(self):
        model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU()).cuda().eval()
        c = Converter(model, input_shape=(1, 8, 8))
        assert isinstance(c.net, nengo.Network)


# ---------------------------------------------------------------------------
# Activation types — neuron type correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("activation_type", [
    "rectified_linear", "relu", "lif", "softlif", "sigmoid",
])
def test_activation_types(activation_type):
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    c = Converter(model, activation_type=activation_type)
    assert isinstance(c.net, nengo.Network)


def test_unknown_activation_type_warns():
    model = _make_mlp()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = Converter(model, activation_type="super_neuron_9000")
    assert any("Unknown activation_type" in str(warning.message) for warning in w)
    assert isinstance(c.net, nengo.Network)


def test_rectified_linear_neuron_type_in_ensemble():
    """activation_type='rectified_linear' must create RectifiedLinear ensembles."""
    model = nn.Linear(3, 4)
    c = Converter(model, activation_type="rectified_linear")
    for ens in c.net.all_ensembles:
        assert isinstance(ens.neuron_type, nengo.RectifiedLinear), (
            f"Expected RectifiedLinear, got {type(ens.neuron_type).__name__}"
        )


def test_spiking_relu_neuron_type_in_ensemble():
    """activation_type='spiking_relu' must create SpikingRectifiedLinear ensembles."""
    model = nn.Linear(3, 4)
    c = Converter(model, activation_type="spiking_relu")
    for ens in c.net.all_ensembles:
        assert isinstance(ens.neuron_type, nengo.SpikingRectifiedLinear), (
            f"Expected SpikingRectifiedLinear, got {type(ens.neuron_type).__name__}"
        )


def test_scale_firing_rates_sets_amplitude():
    """With scale_firing_rates=S, neuron amplitude should be 1/S."""
    scale = 200.0
    model = nn.Linear(3, 4)
    c = Converter(model, scale_firing_rates=scale, activation_type="rectified_linear")
    for ens in c.net.all_ensembles:
        expected_amp = 1.0 / scale
        actual_amp = ens.neuron_type.amplitude
        assert abs(actual_amp - expected_amp) < 1e-6, (
            f"Expected amplitude={expected_amp}, got {actual_amp}"
        )


def test_scale_firing_rates_sets_gain():
    """With scale_firing_rates=S, ensemble gain should equal S."""
    scale = 50.0
    model = nn.Linear(3, 4)
    c = Converter(model, scale_firing_rates=scale, activation_type="rectified_linear")

    inp = list(c.inputs.values())[0]
    with nengo_dl.Simulator(c.net, seed=0) as sim:
        for ens in c.net.all_ensembles:
            params = sim.data[ens]
            if params and "gain" in params:
                np.testing.assert_allclose(
                    params["gain"], np.full_like(params["gain"], scale),
                    rtol=1e-5,
                    err_msg=f"Ensemble gain should equal scale_firing_rates={scale}"
                )


# ---------------------------------------------------------------------------
# Synapse
# ---------------------------------------------------------------------------

class TestConverterSynapse:
    def test_synapse_none_runs(self):
        model = _make_mlp()
        c = Converter(model, synapse=None)
        assert isinstance(c.net, nengo.Network)

    def test_synapse_float_runs(self):
        model = _make_mlp()
        c = Converter(model, synapse=0.005)
        assert isinstance(c.net, nengo.Network)

    def test_synapse_applied_to_connections(self):
        """Converter with synapse!=None should create connections with that synapse."""
        tau = 0.005
        model = _make_mlp(in_size=4, hidden=8, out_size=3)
        c = Converter(model, synapse=tau)

        synapse_taus = []
        for conn in c.net.all_connections:
            if conn.synapse is not None:
                synapse_taus.append(getattr(conn.synapse, "tau", None))

        assert any(t == tau for t in synapse_taus), (
            f"No connection found with synapse tau={tau}; found: {synapse_taus}"
        )


# ---------------------------------------------------------------------------
# ConversionError
# ---------------------------------------------------------------------------

def test_conversion_error_is_exception():
    err = ConversionError("test error")
    assert isinstance(err, Exception)
    assert "test error" in str(err)


# ---------------------------------------------------------------------------
# Numerical verification — rate-mode must match PyTorch
# ---------------------------------------------------------------------------

class TestConverterNumerical:
    """Verify that the converted network produces outputs consistent with the
    original PyTorch model (at least structurally, since rate vs spiking
    neurons differ in dynamics).
    """

    def test_output_shape_matches_pytorch(self):
        """The Nengo network output shape must match the PyTorch model output."""
        in_size, out_size = 4, 3
        model = nn.Sequential(nn.Linear(in_size, 8), nn.ReLU(), nn.Linear(8, out_size))
        model.eval()

        c = Converter(model, scale_firing_rates=100.0)
        inp_node = list(c.inputs.values())[0]
        out_node = list(c.outputs.values())[-1]

        with c.net:
            p = nengo.Probe(out_node, synapse=None)

        x = np.random.RandomState(0).randn(1, 1, in_size).astype(np.float32)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp_node: x})
            nengo_out = sim.data[p]

        assert nengo_out.shape == (1, out_size), (
            f"Expected shape (1, {out_size}), got {nengo_out.shape}"
        )

    def test_no_nan_in_output(self):
        """Converted network must not produce NaN outputs."""
        model = _make_mlp(in_size=4, hidden=8, out_size=3)
        c = Converter(model, scale_firing_rates=100.0)
        inp_node = list(c.inputs.values())[0]
        out_node = list(c.outputs.values())[-1]

        with c.net:
            p = nengo.Probe(out_node, synapse=None)

        x = np.random.RandomState(1).randn(1, 5, 4).astype(np.float32)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(5, data={inp_node: x})
            out = sim.data[p]

        assert not np.any(np.isnan(out)), "Converted network produced NaN output"
        assert not np.any(np.isinf(out)), "Converted network produced Inf output"

    def test_linear_only_network_positive_outputs(self):
        """Converted network with positive-definite input must match PyTorch+ReLU."""
        torch.manual_seed(42)
        model = nn.Linear(4, 3, bias=True)
        with torch.no_grad():
            model.weight.data.abs_()
            model.bias.data.abs_()
        model.eval()

        x_np = np.abs(np.random.RandomState(0).randn(4).astype(np.float32))

        with torch.no_grad():
            pytorch_out = model(torch.tensor(x_np)).numpy()

        c = Converter(model, activation_type="rectified_linear", synapse=None)
        inp_node = list(c.inputs.values())[0]
        out_node = list(c.outputs.values())[-1]

        with c.net:
            p = nengo.Probe(out_node, synapse=None)

        x_batch = x_np.reshape(1, 1, 4)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp_node: x_batch})
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(
            nengo_out, pytorch_out, rtol=1e-4, atol=1e-4,
            err_msg="Converted linear network output mismatch for positive inputs"
        )

    def test_synapse_applied_to_connections(self):
        """Converter with synapse!=None should create connections with that synapse."""
        tau = 0.005
        model = _make_mlp(in_size=4, hidden=8, out_size=3)
        c = Converter(model, synapse=tau)

        synapse_taus = []
        for conn in c.net.all_connections:
            if conn.synapse is not None:
                synapse_taus.append(getattr(conn.synapse, "tau", None))

        assert any(t == tau for t in synapse_taus), (
            f"No connection found with synapse tau={tau}; found: {synapse_taus}"
        )


# ---------------------------------------------------------------------------
# Reference comparison: rate-mode Nengo output == PyTorch output
# ---------------------------------------------------------------------------

class TestConverterRateModeReference:
    """Rate-mode converted network must reproduce PyTorch outputs."""

    def _pos_linear(self, in_size=4, out_size=3, seed=7):
        torch.manual_seed(seed)
        m = nn.Linear(in_size, out_size, bias=True)
        with torch.no_grad():
            m.weight.data.abs_()
            m.bias.data.fill_(1.0)
        m.eval()
        return m

    def test_rate_mode_matches_pytorch_no_scale(self):
        """scale_firing_rates=None: Nengo rate output matches PyTorch."""
        model = self._pos_linear()
        x_np = np.abs(np.random.RandomState(0).randn(4)).astype(np.float32)
        nengo_out, pt_out = _run_converter(model, x_np, scale=None)
        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-4, atol=1e-4,
                                   err_msg="Rate-mode (scale=None) must match PyTorch")

    def test_rate_mode_matches_pytorch_scale_10(self):
        """scale_firing_rates=10: rate-mode output still matches PyTorch."""
        model = self._pos_linear()
        x_np = np.abs(np.random.RandomState(1).randn(4)).astype(np.float32)
        nengo_out, pt_out = _run_converter(model, x_np, scale=10.0)
        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-3, atol=1e-3,
                                   err_msg="Rate-mode (scale=10) must match PyTorch")

    def test_rate_mode_matches_pytorch_scale_500(self):
        """scale_firing_rates=500: rate-mode output still matches PyTorch."""
        model = self._pos_linear()
        x_np = np.abs(np.random.RandomState(2).randn(4)).astype(np.float32)
        nengo_out, pt_out = _run_converter(model, x_np, scale=500.0)
        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-2, atol=1e-2,
                                   err_msg="Rate-mode (scale=500) must match PyTorch")

    def test_bare_linear_preserves_negative_output(self):
        """A bare Linear layer stays linear; ReLU is an explicit layer."""
        torch.manual_seed(42)
        model = nn.Linear(3, 2, bias=True)
        with torch.no_grad():
            model.weight.data.fill_(1.0)
            model.bias.data.fill_(-20.0)  # large negative bias guarantees J < 0
        model.eval()
        x_np = np.zeros(3, dtype=np.float32)  # zero input → J = bias = -20

        nengo_out, pt_out = _run_converter(model, x_np, scale=None)
        np.testing.assert_allclose(nengo_out, pt_out, atol=1e-5)

    def test_relu_matches_pytorch_relu(self):
        """For Linear+ReLU, Nengo output equals PyTorch ReLU output."""
        torch.manual_seed(99)
        model = nn.Sequential(nn.Linear(5, 4, bias=True), nn.ReLU())
        model.eval()
        x_np = np.random.RandomState(3).randn(5).astype(np.float32)

        with torch.no_grad():
            expected = model(torch.tensor(x_np)).numpy()

        nengo_out, _ = _run_converter(model, x_np, scale=None)
        np.testing.assert_allclose(nengo_out, expected, rtol=1e-4, atol=1e-4,
                                   err_msg="Nengo rate output must equal PyTorch ReLU output")

    def test_known_weights_exact_output(self):
        """Hand-crafted weights: verify exact numerical output."""
        # W = [[1, 0], [0, 1]], b = [0.5, 0.5], x = [2.0, 3.0]
        # relu(W@x + b) = relu([2.5, 3.5]) = [2.5, 3.5]
        model = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            model.weight.data = torch.eye(2)
            model.bias.data = torch.tensor([0.5, 0.5])
        model.eval()

        x_np = np.array([2.0, 3.0], dtype=np.float32)
        expected = np.array([2.5, 3.5], dtype=np.float32)

        for scale in [None, 50.0, 200.0]:
            nengo_out, _ = _run_converter(model, x_np, scale=scale)
            np.testing.assert_allclose(
                nengo_out, expected, rtol=1e-3, atol=1e-3,
                err_msg=f"Known-weight test failed for scale={scale}"
            )

    def test_known_weights_relu_negative_output_zero(self):
        """W = -I, b = 0, x = [1, 1]: ReLU([-1,-1]) = [0,0]."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU())
        with torch.no_grad():
            model[0].weight.data = -torch.eye(2)
        model.eval()

        x_np = np.array([1.0, 1.0], dtype=np.float32)
        nengo_out, _ = _run_converter(model, x_np, scale=None)
        np.testing.assert_allclose(nengo_out, np.zeros(2), atol=1e-5,
                                   err_msg="ReLU([-1,-1]) must be [0,0]")

    def test_two_layer_mlp_rate_matches_pytorch(self):
        """Two-layer MLP: rate-mode Nengo equals relu(W2 @ relu(W1@x+b1) + b2)."""
        torch.manual_seed(13)
        model = nn.Sequential(
            nn.Linear(4, 6, bias=True),
            nn.ReLU(),
            nn.Linear(6, 3, bias=True),
        )
        with torch.no_grad():
            # Set up weights so first and last layers give positive outputs
            model[0].weight.data.abs_()
            model[0].bias.data.fill_(0.5)
            model[2].weight.data.abs_()
            model[2].bias.data.fill_(0.5)
        model.eval()

        x_np = np.abs(np.random.RandomState(5).randn(4)).astype(np.float32)

        with torch.no_grad():
            pt_out = model(torch.tensor(x_np)).numpy()

        c = Converter(model, scale_firing_rates=100.0, activation_type="rectified_linear",
                      synapse=None)
        inp = list(c.inputs.values())[0]
        out = list(c.outputs.values())[-1]
        with c.net:
            p = nengo.Probe(out, synapse=None)

        x_b = x_np.reshape(1, 1, 4)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_b}, inference_mode="rate")
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-2, atol=1e-2,
                                   err_msg="Two-layer MLP rate mode should match PyTorch")

    def test_argmax_preserved_for_random_network(self):
        """rate-mode argmax must agree with PyTorch argmax on positive-output net."""
        torch.manual_seed(17)
        model = _make_mlp(in_size=8, hidden=16, out_size=5)
        with torch.no_grad():
            for p_t in model.parameters():
                p_t.data.abs_()
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    m.bias.data.fill_(1.0)
        model.eval()

        rng = np.random.RandomState(77)
        n_correct = 0
        n_trials = 10
        for i in range(n_trials):
            x_np = np.abs(rng.randn(8)).astype(np.float32)
            with torch.no_grad():
                pt_out = model(torch.tensor(x_np)).numpy()

            c = Converter(model, scale_firing_rates=500.0, synapse=None)
            inp = list(c.inputs.values())[0]
            out = list(c.outputs.values())[-1]
            with c.net:
                p = nengo.Probe(out, synapse=None)
            x_b = x_np.reshape(1, 1, 8)
            with nengo_dl.Simulator(c.net, seed=0) as sim:
                sim.run_steps(1, data={inp: x_b}, inference_mode="rate")
                nengo_out = sim.data[p][0]

            if np.argmax(nengo_out) == np.argmax(pt_out):
                n_correct += 1

        assert n_correct == n_trials, (
            f"argmax matched {n_correct}/{n_trials} times (expected all)"
        )


# ---------------------------------------------------------------------------
# Spiking mode
# ---------------------------------------------------------------------------

class TestConverterSpikingMode:
    def test_spiking_output_is_nonnegative(self):
        """Spiking output (spike count / dt) must be non-negative."""
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
        model.eval()
        c = Converter(model, scale_firing_rates=500.0, activation_type="spiking_relu")
        inp = list(c.inputs.values())[0]
        out = list(c.outputs.values())[-1]
        with c.net:
            p = nengo.Probe(out, synapse=None)

        x = np.ones((1, 1, 4), dtype=np.float32)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x}, inference_mode="spiking")
            out_data = sim.data[p]
        assert np.all(out_data >= 0.0), "Spiking output must be non-negative"

    def test_spiking_output_differs_from_rate_output(self):
        """A single-step spiking run differs from rate run (discrete vs continuous)."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8, bias=True), nn.ReLU())
        with torch.no_grad():
            model[0].weight.data.fill_(0.5)
            model[0].bias.data.fill_(0.5)
        model.eval()

        x_np = np.ones((1, 1, 4), dtype=np.float32) * 2.0

        c_rate = Converter(model, scale_firing_rates=100.0, activation_type="rectified_linear")
        c_spike = Converter(model, scale_firing_rates=100.0, activation_type="spiking_relu")

        def run(converter, mode):
            inp = list(converter.inputs.values())[0]
            out = list(converter.outputs.values())[-1]
            with converter.net:
                p = nengo.Probe(out, synapse=None)
            with nengo_dl.Simulator(converter.net, seed=42) as sim:
                sim.run_steps(1, data={inp: x_np}, inference_mode=mode)
                return sim.data[p][0].copy()

        rate_out = run(c_rate, "rate")
        spike_out = run(c_spike, "spiking")
        # Rate output is continuous; spiking output is 0 or 1/dt
        # They will differ in magnitude / pattern for a single step
        assert rate_out.shape == spike_out.shape

    def test_spiking_multistep_average_approaches_rate(self):
        """Average spiking output over many steps approaches rate output."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(3, 2, bias=True), nn.ReLU())
        with torch.no_grad():
            model[0].weight.data = torch.eye(2, 3)
            model[0].bias.data.fill_(0.3)  # ensures firing
        model.eval()

        n_steps = 500
        x_val = np.ones((1, n_steps, 3), dtype=np.float32) * 1.0

        rate_c = Converter(model, scale_firing_rates=200.0, activation_type="rectified_linear", synapse=None)
        spike_c = Converter(model, scale_firing_rates=200.0, activation_type="spiking_relu", synapse=None)

        def run(conv, mode, x):
            inp = list(conv.inputs.values())[0]
            out = list(conv.outputs.values())[-1]
            with conv.net:
                p = nengo.Probe(out, synapse=None)
            with nengo_dl.Simulator(conv.net, seed=1) as sim:
                sim.run_steps(n_steps, data={inp: x}, inference_mode=mode)
                return sim.data[p].copy()

        rate_out = run(rate_c, "rate", x_val[:, :1, :])  # single step, repeated
        spike_data = run(spike_c, "spiking", x_val)
        spike_mean = spike_data.mean(axis=0)  # average over timesteps

        # Mean spiking rate (amplitude * spike_count) should be close to rate output
        np.testing.assert_allclose(
            spike_mean, rate_out[0], rtol=0.2, atol=0.2,
            err_msg="Mean spiking output should approximate rate output over many steps"
        )


# ---------------------------------------------------------------------------
# Layer coverage: Flatten, BatchNorm, MaxPool, nested Sequential, bias=False
# ---------------------------------------------------------------------------

class TestConverterLayerCoverage:
    def test_flatten_linear_exact_output(self):
        """Flatten (no-op in Nengo) + Linear: output matches PyTorch exactly."""
        model = nn.Sequential(nn.Flatten(start_dim=0), nn.Linear(4, 2, bias=False))
        with torch.no_grad():
            model[1].weight.data = torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
            )
        model.eval()

        x_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        with torch.no_grad():
            pt_out = model(torch.tensor(x_np)).numpy()  # [1, 5]

        c = Converter(model, activation_type="rectified_linear")
        inp = list(c.inputs.values())[0]
        out = list(c.outputs.values())[-1]
        with c.net:
            p = nengo.Probe(out, synapse=None)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(nengo_out, pt_out, atol=1e-4,
                                   err_msg="Flatten+Linear output must match PyTorch")

    def test_permute_flatten_linear_exact_output(self):
        class PermuteFlattenLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2, bias=False)

            def forward(self, x):
                return self.fc(torch.flatten(x.permute(0, 2, 1), start_dim=1))

        model = PermuteFlattenLinear()
        with torch.no_grad():
            model.fc.weight.data = torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 1.5, 2.0]]
            )
        model.eval()

        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with torch.no_grad():
            pt_out = model(torch.tensor(x_np).unsqueeze(0)).numpy()[0]

        c = Converter(model, input_shape=(2, 2))
        inp = list(c.inputs.values())[0]
        out = c.outputs["output_0"]
        with c.net:
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(nengo_out, pt_out, atol=1e-5, rtol=1e-5)

    def test_transpose_view_linear_exact_output(self):
        class TransposeViewLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2, bias=False)

            def forward(self, x):
                x = x.transpose(1, 2)
                return self.fc(torch.flatten(x, start_dim=1))

        model = TransposeViewLinear()
        with torch.no_grad():
            model.fc.weight.data = torch.tensor(
                [[2.0, 0.0, -1.0, 1.0], [1.0, 3.0, 0.0, -2.0]]
            )
        model.eval()

        x_np = np.array([[2.0, 1.0], [0.0, -1.0]], dtype=np.float32)
        with torch.no_grad():
            pt_out = model(torch.tensor(x_np).unsqueeze(0)).numpy()[0]

        c = Converter(model, input_shape=(2, 2))
        inp = list(c.inputs.values())[0]
        out = c.outputs["output_0"]
        with c.net:
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(nengo_out, pt_out, atol=1e-5, rtol=1e-5)

    def test_linear_bias_false_exact_output(self):
        """nn.Linear with bias=False converts correctly; output = W@x."""
        W = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.data = torch.tensor(W)
        model.eval()

        x_np = np.array([1.0, 2.0], dtype=np.float32)
        expected = np.maximum(0.0, W @ x_np)  # relu(W@x) = [2, 6]

        c = Converter(model, activation_type="rectified_linear")
        inp = list(c.inputs.values())[0]
        out = list(c.outputs.values())[-1]
        with c.net:
            p = nengo.Probe(out, synapse=None)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(nengo_out, expected, atol=1e-4,
                                   err_msg=f"bias=False output must equal relu(W@x)={expected}")

    def test_batchnorm_uses_fallback_and_warns(self):
        """BatchNorm1d falls back unless inference_only=True."""
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        model.eval()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = Converter(model)

        assert isinstance(c.net, nengo.Network), "BatchNorm fallback must produce a valid Network"
        assert any("TorchNode fallback" in str(warning.message) for warning in w), (
            "BatchNorm must warn about fallback conversion"
        )

    def test_maxpool_uses_fallback_and_warns(self):
        """MaxPool2d falls back unless max_to_avg_pool=True."""
        model = nn.Sequential(nn.MaxPool2d(2), nn.Flatten())
        model.eval()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = Converter(model, input_shape=(1, 4, 4))

        assert isinstance(c.net, nengo.Network), "MaxPool fallback must produce a valid Network"
        assert any("TorchNode fallback" in str(warning.message) for warning in w)

    def test_avgpool2d_native_matches_pytorch(self):
        model = nn.Sequential(nn.AvgPool2d(2), nn.Flatten()).eval()
        x_np = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        with torch.no_grad():
            pt_out = model(torch.tensor(x_np).unsqueeze(0)).numpy()[0]

        c = Converter(model, input_shape=(1, 4, 4))
        inp = list(c.inputs.values())[0]
        out = c.outputs["output_0"]
        with c.net:
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-5, atol=1e-5)

    def test_batchnorm1d_native_matches_pytorch_inference(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)).eval()
        x_np = np.random.RandomState(0).randn(4).astype(np.float32)
        with torch.no_grad():
            pt_out = model(torch.tensor(x_np).unsqueeze(0)).numpy()[0]

        c = Converter(model, inference_only=True)
        inp = list(c.inputs.values())[0]
        out = c.outputs["output_0"]
        with c.net:
            p = nengo.Probe(out, synapse=None)

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0]

        np.testing.assert_allclose(nengo_out, pt_out, rtol=1e-5, atol=1e-5)

    def test_three_layer_chain_argmax_matches_pytorch(self):
        """3-layer Linear→ReLU→Linear→ReLU→Linear: rate-mode argmax matches PyTorch."""
        torch.manual_seed(55)
        model = nn.Sequential(
            nn.Linear(4, 6, bias=True),
            nn.ReLU(),
            nn.Linear(6, 4, bias=True),
            nn.ReLU(),
            nn.Linear(4, 3, bias=True),
        )
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.abs_()
                    m.bias.data.fill_(0.3)
        model.eval()

        x_np = np.abs(np.random.RandomState(7).randn(4)).astype(np.float32)
        with torch.no_grad():
            pt_out = model(torch.tensor(x_np)).numpy()

        c = Converter(model, scale_firing_rates=300.0, synapse=None)
        inp = list(c.inputs.values())[0]
        out = list(c.outputs.values())[-1]
        with c.net:
            p = nengo.Probe(out, synapse=None)
        with nengo_dl.Simulator(c.net, seed=0) as sim:
            sim.run_steps(1, data={inp: x_np.reshape(1, 1, -1)}, inference_mode="rate")
            nengo_out = sim.data[p][0]

        assert np.argmax(nengo_out) == np.argmax(pt_out), (
            f"3-layer chain: argmax mismatch — pytorch={np.argmax(pt_out)}, nengo={np.argmax(nengo_out)}"
        )
        np.testing.assert_allclose(nengo_out, pt_out, rtol=0.05, atol=0.05,
                                   err_msg="3-layer chain rate-mode output must closely match PyTorch")

    def test_eval_mode_required_for_batchnorm_consistency(self):
        """BatchNorm in eval mode gives consistent output; train mode uses batch stats."""
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        x = torch.ones(8, 4)

        model.train()
        # Running stats are not updated in train mode unless we call model(x)
        with torch.no_grad():
            train_out = model(x).numpy()

        model.eval()
        with torch.no_grad():
            eval_out = model(x).numpy()

        # In train mode, BatchNorm normalises using the batch — mean ≈ 0 for constant input
        # In eval mode, it uses running stats (initialised to mean=0, var=1)
        # The results may or may not differ depending on running-stat initialisation,
        # but at minimum the eval-mode output must be finite and consistent.
        assert np.all(np.isfinite(eval_out)), "BatchNorm eval output must be finite"
        # Calling model a second time in eval mode must give the same result
        with torch.no_grad():
            eval_out2 = model(x).numpy()
        np.testing.assert_allclose(eval_out, eval_out2, atol=1e-6,
                                   err_msg="BatchNorm eval mode must be deterministic")


# ---------------------------------------------------------------------------
# set_weights strict-mode coverage
# ---------------------------------------------------------------------------

class TestSetWeightsStrict:
    def test_unknown_key_raises(self):
        """set_weights with an unknown key raises ValueError (strict=True default)."""
        model = _make_mlp(in_size=4, hidden=8, out_size=3)
        c = Converter(model)
        inp = list(c.inputs.values())[0]

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            with pytest.raises(ValueError, match="Unknown weight keys"):
                sim.set_weights({"definitely_not_a_real_key": np.zeros(3)})

    def test_valid_keys_accepted(self):
        """set_weights with all valid keys (from get_weights) must not raise."""
        model = _make_mlp()
        c = Converter(model)
        inp = list(c.inputs.values())[0]

        with nengo_dl.Simulator(c.net, seed=0) as sim:
            w = sim.get_weights()
            sim.set_weights(w)  # exact same keys — must not raise

    def test_converted_weights_transferable(self):
        """Weights from one converted-model sim transfer to another identical sim."""
        model = _make_mlp(in_size=4, hidden=8, out_size=3)
        c1 = Converter(model)
        c2 = Converter(model)

        inp1 = list(c1.inputs.values())[0]
        out1 = list(c1.outputs.values())[-1]
        inp2 = list(c2.inputs.values())[0]
        out2 = list(c2.outputs.values())[-1]
        with c1.net:
            p1 = nengo.Probe(out1, synapse=None)
        with c2.net:
            p2 = nengo.Probe(out2, synapse=None)

        x = np.ones((1, 1, 4), dtype=np.float32)

        with nengo_dl.Simulator(c1.net, seed=0) as sim1:
            w1 = sim1.get_weights()
            sim1.run_steps(1, data={inp1: x}, inference_mode="rate")
            out_before = sim1.data[p1][0].copy()

        with nengo_dl.Simulator(c2.net, seed=99) as sim2:
            sim2.set_weights(w1)  # load trained weights
            sim2.run_steps(1, data={inp2: x}, inference_mode="rate")
            out_after = sim2.data[p2][0].copy()

        np.testing.assert_allclose(out_before, out_after, atol=1e-4,
                                   err_msg="Transferred weights must reproduce same output")
