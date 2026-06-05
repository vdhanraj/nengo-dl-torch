"""Tests for nengo_dl.tensor_graph.TensorGraph."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import nengo
import nengo_dl
from nengo_dl.tensor_graph import (
    TensorGraph,
    _is_trainable_signal,
    _is_trainable_param_role,
    topo_sort,
)
from nengo.builder import Builder as NengoBuilder, Model as NengoModel
from nengo.builder.signal import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(net, dt=0.001, minibatch_size=1, device="cpu"):
    model = NengoModel(dt=dt)
    NengoBuilder.build(model, net)
    return TensorGraph(model, dt=dt, minibatch_size=minibatch_size, device=device)


def _simple_net(seed=0):
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node(np.zeros(1))
        ens = nengo.Ensemble(10, 1, seed=seed)
        nengo.Connection(inp, ens, synapse=None)
        p = nengo.Probe(ens, synapse=None)
    return net, inp, p


def _linear_net():
    """Node(input) → transform=W → probe: exact linear mapping."""
    with nengo.Network(seed=0) as net:
        inp = nengo.Node(np.zeros(2))
        out = nengo.Node(size_in=2)
        # transform = [[1,0],[0,2]] → out[0]=in[0], out[1]=2*in[1]
        W = np.array([[1.0, 0.0], [0.0, 2.0]])
        nengo.Connection(inp, out, transform=W, synapse=None)
        p = nengo.Probe(out, synapse=None)
    return net, inp, p


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestTensorGraphConstruction:
    def test_is_nn_module(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert isinstance(tg, nn.Module)

    def test_device_cpu(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert tg.device == torch.device("cpu")

    def test_dtype_float32(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert tg.dtype == torch.float32

    def test_minibatch_size_stored(self):
        net, inp, p = _simple_net()
        tg = _build(net, minibatch_size=8)
        assert tg.minibatch_size == 8

    def test_signals_populated(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert len(tg.signals._data) > 0

    def test_param_dict_is_nn_parameter_dict(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert isinstance(tg._param_dict, nn.ParameterDict)

    def test_probe_sig_map_populated(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert p in tg._probe_sig_map

    def test_input_node_sigs_populated(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        assert inp in tg._input_node_sigs

    def test_extra_repr(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        r = tg.extra_repr()
        assert "dt=" in r
        assert "minibatch_size=" in r


# ---------------------------------------------------------------------------
# Forward pass — shape and value correctness
# ---------------------------------------------------------------------------

class TestTensorGraphForward:
    def test_forward_returns_probe_dict(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        result = tg.forward(3)
        assert isinstance(result, dict)
        assert p in result

    def test_output_shape_batch1(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        result = tg.forward(5)
        # (batch=1, n_steps=5, probe_size=1)
        assert result[p].shape == (1, 5, 1)

    def test_output_shape_batch_n(self):
        net, inp, p = _simple_net()
        tg = _build(net, minibatch_size=4)
        result = tg.forward(5)
        assert result[p].shape == (4, 5, 1)

    def test_input_injection_exact_value(self):
        """Injected input must appear in output via an identity transform."""
        net, inp, p = _linear_net()
        tg = _build(net)
        # W = diag(1, 2): [3, 4] → [3, 8]
        data = np.array([[[3.0, 4.0]]])  # (batch=1, steps=1, size=2)
        result = tg.forward(1, input_data={inp: data})
        np.testing.assert_allclose(
            result[p][0, 0].detach().cpu().numpy(), [3.0, 8.0], atol=1e-5,
            err_msg="Forward pass must apply the connection transform correctly"
        )

    def test_scalar_transform_exact_value(self):
        """transform=3.0, input=2.0 must give output=6.0."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            out = nengo.Node(size_in=1)
            nengo.Connection(inp, out, transform=3.0, synapse=None)
            p = nengo.Probe(out, synapse=None)

        model = NengoModel()
        NengoBuilder.build(model, net)
        tg = TensorGraph(model, dt=0.001, minibatch_size=1, device="cpu")

        data = np.array([[[2.0]]])  # (batch=1, steps=1, dim=1)
        result = tg.forward(1, input_data={inp: data})
        np.testing.assert_allclose(
            result[p][0, 0].detach().cpu().numpy(), [6.0], atol=1e-5,
            err_msg="transform=3.0 applied to input=2.0 must give 6.0"
        )

    def test_multiple_forward_calls_independent(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        r1 = tg.forward(3)
        tg.reset_state()
        r2 = tg.forward(3)
        np.testing.assert_allclose(
            r1[p].detach().cpu().numpy(), r2[p].detach().cpu().numpy(), rtol=1e-4
        )

    def test_forward_training_flag(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        tg.forward(1, training=False)
        tg.forward(1, training=True)

    def test_forward_output_not_nan(self):
        """Forward pass must not produce NaN in any probe output."""
        net, inp, p = _simple_net()
        tg = _build(net)
        result = tg.forward(10)
        out = result[p].detach().cpu().numpy()
        assert not np.any(np.isnan(out)), "Forward pass produced NaN"

    def test_multi_step_constant_input_reproducible(self):
        """N steps with constant zero input gives the same output each step."""
        net, inp, p = _linear_net()
        tg = _build(net)
        data = np.zeros((1, 5, 2))  # zero input, 5 steps
        result = tg.forward(5, input_data={inp: data})
        out = result[p].detach().cpu().numpy()  # (1, 5, 2)
        # Each step has the same input → each step should give the same output
        for t in range(1, 5):
            np.testing.assert_allclose(
                out[0, t], out[0, 0], atol=1e-5,
                err_msg=f"Step {t} output differs from step 0 for constant input"
            )


# ---------------------------------------------------------------------------
# Signal collection ordering
# ---------------------------------------------------------------------------

class TestSignalOrdering:
    def test_deterministic_signal_order(self):
        """Building the same network twice gives the same param key order."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        keys1 = list(_build(net).signals.get_all_parameters().keys())
        keys2 = list(_build(net).signals.get_all_parameters().keys())
        assert keys1 == keys2

    def test_stable_param_keys(self):
        """Param keys must use stable indices, not memory addresses."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        tg = _build(net)
        for key in tg.signals.get_all_parameters():
            assert key.startswith("param_"), f"Unstable key: {key}"
            assert key[6:].isdigit(), f"Unexpected key format: {key}"

    def test_param_keys_are_strings(self):
        """All parameter keys must be strings."""
        net, inp, p = _simple_net()
        tg = _build(net)
        for key in tg.signals.get_all_parameters():
            assert isinstance(key, str)


# ---------------------------------------------------------------------------
# Trainable detection
# ---------------------------------------------------------------------------

class TestTrainableDetection:
    def test_readonly_signal_without_owner_is_not_trainable(self):
        """Trainable detection requires an explicit supported owner/role pair."""
        sig = Signal(np.ones(5), name="w")
        sig._readonly = True

        model = NengoModel()
        assert _is_trainable_signal(sig, model) is False

    def test_non_readonly_signal_is_not_trainable(self):
        """A signal that is not readonly is not trainable."""
        sig = Signal(np.ones(5), name="v")
        # readonly=False by default

        model = NengoModel()
        assert _is_trainable_signal(sig, model) is False

    def test_connection_weights_role_is_trainable(self):
        with nengo.Network(seed=0) as net:
            pre = nengo.Ensemble(10, 1, seed=0)
            post = nengo.Ensemble(10, 1, seed=1)
            conn = nengo.Connection(pre, post, synapse=None)

        model = NengoModel()
        NengoBuilder.build(model, net)
        sig = model.sig[conn]["weights"]
        sig_owner = {sig.base: (conn, "weights"), sig: (conn, "weights")}
        assert _is_trainable_signal(sig, model, sig_owner=sig_owner) is True

    def test_ensemble_encoders_role_is_trainable(self):
        with nengo.Network(seed=0) as net:
            ens = nengo.Ensemble(10, 1, seed=0)

        model = NengoModel()
        NengoBuilder.build(model, net)
        sig = model.sig[ens]["encoders"]
        sig_owner = {sig.base: (ens, "encoders"), sig: (ens, "encoders")}
        assert _is_trainable_signal(sig, model, sig_owner=sig_owner) is True

    def test_neuron_bias_role_is_trainable(self):
        with nengo.Network(seed=0) as net:
            ens = nengo.Ensemble(10, 1, seed=0)

        model = NengoModel()
        NengoBuilder.build(model, net)
        neurons = ens.neurons
        sig = model.sig[neurons]["bias"]
        sig_owner = {sig.base: (neurons, "bias"), sig: (neurons, "bias")}
        assert _is_trainable_signal(sig, model, sig_owner=sig_owner) is True

    def test_non_parameter_role_is_not_trainable(self):
        with nengo.Network(seed=0) as net:
            ens = nengo.Ensemble(10, 1, seed=0)

        model = NengoModel()
        NengoBuilder.build(model, net)
        sig = model.sig[ens.neurons]["voltage"]
        sig_owner = {
            sig.base: (ens.neurons, "voltage"),
            sig: (ens.neurons, "voltage"),
        }
        assert _is_trainable_signal(sig, model, sig_owner=sig_owner) is False

    def test_synapse_state_signal_is_not_trainable(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=0.1)

        tg = _build(net)
        lowpass_states = [
            sig
            for obj, roles in tg.model.sig.items()
            if type(obj).__name__ == "Lowpass"
            for role, sig in roles.items()
            if role == "_state_X"
        ]
        assert lowpass_states, "Expected a Lowpass state signal in the built model"
        for sig in lowpass_states:
            assert tg.signals.get_parameter(sig) is None

    def test_voltage_signal_is_not_registered_as_parameter(self):
        with nengo.Network(seed=0) as net:
            ens = nengo.Ensemble(10, 1, neuron_type=nengo.LIF(), seed=0)

        tg = _build(net)
        voltage_sig = tg.model.sig[ens.neurons]["voltage"]
        assert tg.signals.get_parameter(voltage_sig) is None

    def test_trainable_role_helper(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            conn = nengo.Connection(inp, ens, synapse=None)

        assert _is_trainable_param_role(conn, "weights") is True
        assert _is_trainable_param_role(ens, "encoders") is True
        assert _is_trainable_param_role(ens.neurons, "bias") is True
        assert _is_trainable_param_role(ens.neurons, "voltage") is False

    def test_scalar_zero_signal_not_trainable(self):
        """A signal named ZERO with size=1 is never trainable."""
        sig = Signal(np.array([0.0]), name="ZERO")
        sig._readonly = True

        model = NengoModel()
        assert _is_trainable_signal(sig, model) is False

    def test_step_signal_not_trainable(self):
        """A signal named 'step' is not trainable (it's a simulation counter)."""
        sig = Signal(np.array([0.0, 0.0]), name="step")
        sig._readonly = True

        model = NengoModel()
        assert _is_trainable_signal(sig, model) is False

    def test_trainable_params_nonempty_for_ensemble(self):
        """A network with an Ensemble must have at least one trainable parameter."""
        net, inp, p = _simple_net()
        tg = _build(net)
        params = tg.signals.get_all_parameters()
        assert len(params) > 0, "Ensemble network must have trainable parameters"

    def test_trainable_params_are_nn_parameters(self):
        """All values returned by get_all_parameters() must be nn.Parameter."""
        net, inp, p = _simple_net()
        tg = _build(net)
        for key, val in tg.signals.get_all_parameters().items():
            assert isinstance(val, nn.Parameter), (
                f"Parameter '{key}' is {type(val).__name__}, not nn.Parameter"
            )


# ---------------------------------------------------------------------------
# Weight management
# ---------------------------------------------------------------------------

class TestWeightManagement:
    def test_get_weights_returns_dict(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        w = tg.get_weights()
        assert isinstance(w, dict)

    def test_get_weights_numpy_values(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        for k, v in tg.get_weights().items():
            assert isinstance(v, np.ndarray)

    def test_set_weights_roundtrip(self):
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(10, 1, seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        tg1 = _build(net)
        tg2 = _build(net)
        w1 = tg1.get_weights()
        tg2.set_weights(w1)
        w2 = tg2.get_weights()
        for k in w1:
            np.testing.assert_allclose(w1[k], w2[k], rtol=1e-5)

    def test_set_weights_strict_raises_on_unknown_key(self):
        """strict=True (default) raises ValueError for unknown weight keys."""
        net, inp, p = _simple_net()
        tg = _build(net)
        with pytest.raises(ValueError, match="Unknown weight keys"):
            tg.set_weights({"nonexistent_key_xyz": np.zeros(5)})

    def test_set_weights_lenient_ignores_unknown_keys(self):
        """strict=False silently ignores keys that don't exist in the model."""
        net, inp, p = _simple_net()
        tg = _build(net)
        tg.set_weights({"nonexistent_key_xyz": np.zeros(5)}, strict=False)  # must not raise

    def test_set_weights_changes_forward_output(self):
        """set_weights with different values must change the forward output."""
        net, inp, p = _simple_net()
        tg1 = _build(net)
        tg2 = _build(net)

        # Perturb tg2's weights significantly
        w1 = tg1.get_weights()
        w_perturbed = {k: v + 10.0 for k, v in w1.items()}
        tg2.set_weights(w_perturbed)

        r1 = tg1.forward(1)
        r2 = tg2.forward(1)

        out1 = r1[p].detach().cpu().numpy()
        out2 = r2[p].detach().cpu().numpy()
        # With +10 on all params, outputs should differ
        assert not np.allclose(out1, out2, atol=1e-3), (
            "Perturbed weights should change the forward output"
        )


# ---------------------------------------------------------------------------
# reset_state
# ---------------------------------------------------------------------------

class TestResetState:
    def test_reset_state_gives_reproducible_output(self):
        net, inp, p = _simple_net()
        tg = _build(net)
        r1 = tg.forward(5)
        tg.reset_state()
        r2 = tg.forward(5)
        np.testing.assert_allclose(
            r1[p].detach().cpu().numpy(), r2[p].detach().cpu().numpy(), rtol=1e-4
        )

    def test_reset_restores_lif_state(self):
        """After resetting LIF state, voltage probe matches a fresh run."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            ens = nengo.Ensemble(5, 1, neuron_type=nengo.LIF(), seed=0)
            nengo.Connection(inp, ens, synapse=None)
            p_v = nengo.Probe(ens.neurons, "voltage", synapse=None)

        model = NengoModel()
        NengoBuilder.build(model, net)
        tg = TensorGraph(model, dt=0.001, minibatch_size=1, device="cpu")

        r1 = tg.forward(10)
        tg.reset_state()
        r2 = tg.forward(10)

        v1 = r1[p_v].detach().cpu().numpy()
        v2 = r2[p_v].detach().cpu().numpy()
        np.testing.assert_allclose(v1, v2, rtol=1e-4,
                                   err_msg="After reset_state, voltage trajectory must repeat")


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_trainable_params_have_grad_after_backward(self):
        """After forward + backward on a differentiable network, params must have grads."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            ens = nengo.Ensemble(
                10, 2,
                neuron_type=nengo.RectifiedLinear(),
                seed=0,
            )
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        model = NengoModel()
        NengoBuilder.build(model, net)
        tg = TensorGraph(model, dt=0.001, minibatch_size=1, device="cpu")

        data = np.ones((1, 1, 2), dtype=np.float32)
        result = tg.forward(1, input_data={inp: data}, training=True)
        loss = result[p].sum()
        loss.backward()

        params_with_grad = [
            (k, v) for k, v in tg._param_dict.items()
            if v.grad is not None
        ]
        assert len(params_with_grad) > 0, (
            "At least one trainable parameter must have a gradient after backward"
        )

    def test_zero_loss_gives_zero_grad(self):
        """If loss is zero, no gradient update is needed."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(1))
            out = nengo.Node(size_in=1)
            nengo.Connection(inp, out, synapse=None)
            p = nengo.Probe(out, synapse=None)

        model = NengoModel()
        NengoBuilder.build(model, net)
        tg = TensorGraph(model, dt=0.001, minibatch_size=1, device="cpu")

        data = np.zeros((1, 1, 1), dtype=np.float32)
        result = tg.forward(1, input_data={inp: data}, training=True)
        # For a passthrough node-to-node network with zero input, output is zero
        out_val = result[p]
        assert out_val.abs().max().item() < 1e-5

    def test_optimizer_step_changes_params(self):
        """An optimizer step on a trained network must change trainable parameters."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            ens = nengo.Ensemble(
                8, 2,
                neuron_type=nengo.RectifiedLinear(),
                seed=0,
            )
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        model = NengoModel()
        NengoBuilder.build(model, net)
        tg = TensorGraph(model, dt=0.001, minibatch_size=1, device="cpu")

        # Snapshot initial params
        w_before = {k: v.detach().clone() for k, v in tg._param_dict.items()}

        opt = torch.optim.SGD(tg.parameters(), lr=0.1)
        data = np.ones((1, 1, 2), dtype=np.float32) * 2.0
        result = tg.forward(1, input_data={inp: data}, training=True)
        loss = result[p].sum()
        loss.backward()
        opt.step()

        # At least one parameter must have changed
        any_changed = any(
            not torch.equal(tg._param_dict[k], w_before[k])
            for k in w_before
        )
        assert any_changed, "Optimizer step must change at least one parameter"


# ---------------------------------------------------------------------------
# Rate vs spiking mode
# ---------------------------------------------------------------------------

class TestInferenceModes:
    def test_rate_mode_flag_accepted(self):
        """forward(rate_mode=True) must not raise."""
        net, inp, p = _simple_net()
        tg = _build(net)
        tg.forward(1, rate_mode=True)
        tg.forward(1, rate_mode=False)

    def test_rectifiedlinear_rate_mode_equals_spiking_mode(self):
        """RectifiedLinear is a rate neuron — rate and spiking mode give same output."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            ens = nengo.Ensemble(
                5, 2,
                neuron_type=nengo.RectifiedLinear(),
                seed=0,
            )
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        model = NengoModel()
        NengoBuilder.build(model, net)

        data = np.ones((1, 1, 2), dtype=np.float32)

        tg_rate = TensorGraph(model, dt=0.001, minibatch_size=1, device="cpu")
        r_rate = tg_rate.forward(1, input_data={inp: data}, rate_mode=True)

        tg_spike = TensorGraph(model, dt=0.001, minibatch_size=1, device="cpu")
        r_spike = tg_spike.forward(1, input_data={inp: data}, rate_mode=False)

        out_rate = r_rate[p].detach().cpu().numpy()
        out_spike = r_spike[p].detach().cpu().numpy()
        np.testing.assert_allclose(out_rate, out_spike, rtol=1e-4,
                                   err_msg="RectifiedLinear rate=spiking in any mode")


# ---------------------------------------------------------------------------
# TorchNode module discovery
# ---------------------------------------------------------------------------

class TestTorchModuleDiscovery:
    def test_torch_modules_list_populated(self):
        lin = nn.Linear(4, 2)
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(4))
            node = nengo_dl.TorchNode(lin, size_in=4, size_out=2)
            nengo.Connection(inp, node, synapse=None)
            p = nengo.Probe(node, synapse=None)

        tg = _build(net)
        assert len(tg._torch_modules) > 0

    def test_torch_module_on_correct_device(self):
        lin = nn.Linear(4, 2)
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(4))
            node = nengo_dl.TorchNode(lin, size_in=4, size_out=2)
            nengo.Connection(inp, node, synapse=None)
            p = nengo.Probe(node, synapse=None)

        tg = _build(net)
        for module in tg._torch_modules:
            for param in module.parameters():
                assert param.device == tg.device

    def test_torch_module_output_exact_value(self):
        """TorchNode wrapping a known linear layer produces exact output."""
        weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        bias = torch.zeros(2)
        lin = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            lin.weight.copy_(weight)
            lin.bias.copy_(bias)

        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.zeros(2))
            node = nengo_dl.TorchNode(lin, size_in=2, size_out=2, pass_time=False)
            nengo.Connection(inp, node, synapse=None)
            p = nengo.Probe(node, synapse=None)

        x = np.array([[[3.0, 4.0]]])  # (batch=1, steps=1, dim=2)
        # Expected: W @ x = [3*1+4*0, 3*0+4*2] = [3, 8]
        with nengo_dl.Simulator(net, seed=0) as sim:
            sim.run_steps(1, data={inp: x})
        np.testing.assert_allclose(sim.data[p][0], [3.0, 8.0], atol=1e-4,
                                   err_msg="TorchNode linear output must match W@x")


# ---------------------------------------------------------------------------
# nengo.Simulator reference comparisons for TensorGraph
# ---------------------------------------------------------------------------

class TestNengoReference:
    def test_forward_constant_input_matches_nengo(self):
        """TensorGraph.forward on Node(2.0)→transform=3.0 matches nengo.Simulator."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([2.0]))
            b = nengo.Node(size_in=1)
            nengo.Connection(inp, b, transform=3.0, synapse=None)
            p = nengo.Probe(b, synapse=None)

        model = NengoModel()
        NengoBuilder.build(model, net)
        tg = TensorGraph(model, dt=0.001, minibatch_size=1, device="cpu")

        with nengo.Simulator(net, dt=0.001) as ref:
            ref.run_steps(5)
            ref_out = ref.data[p].copy()

        result = tg.forward(5)
        dl_out = result[p].detach().cpu().numpy().squeeze(0)  # (5, 1)

        np.testing.assert_allclose(ref_out, dl_out, atol=1e-5,
                                   err_msg="TensorGraph must match nengo.Simulator on linear net")

    def test_forward_relu_rate_matches_nengo(self):
        """TensorGraph ReLU rate mode matches nengo.Simulator."""
        with nengo.Network(seed=0) as net:
            inp = nengo.Node(np.array([1.5]))
            ens = nengo.Ensemble(
                6, 1, neuron_type=nengo.RectifiedLinear(),
                gain=nengo.dists.Choice([1.0]),
                bias=nengo.dists.Choice([0.0]),
                encoders=nengo.dists.Choice([[1.0]]),
                seed=0,
            )
            nengo.Connection(inp, ens, synapse=None)
            p = nengo.Probe(ens, synapse=None)

        model = NengoModel()
        NengoBuilder.build(model, net)
        tg = TensorGraph(model, dt=0.001, minibatch_size=1, device="cpu")

        with nengo.Simulator(net, dt=0.001) as ref:
            ref.run_steps(5)
            ref_out = ref.data[p].copy()

        result = tg.forward(5, rate_mode=True)
        dl_out = result[p].detach().cpu().numpy().squeeze(0)  # (5, 1)

        np.testing.assert_allclose(ref_out, dl_out, atol=1e-4,
                                   err_msg="TensorGraph ReLU rate mode must match nengo.Simulator")
