"""Builders for Nengo Process operators (PyTorch backend).

Handles the ``SimProcess`` operator. Lowpass (IIR) filters are implemented
natively in PyTorch so gradients flow through them. Other processes fall
back to Nengo's numpy step functions.
"""

import inspect
import numpy as np
import torch

import nengo
import nengo.synapses
from nengo.builder.processes import SimProcess
from nengo.processes import WhiteSignal, WhiteNoise

from .builder import Builder, BuildConfig, OpBuilder


@Builder.register(SimProcess)
class SimProcessBuilder(OpBuilder):
    """Translates a ``SimProcess`` operator to PyTorch.

    Known process types (Lowpass, Alpha, LinearFilter) are implemented with
    native PyTorch ops. Everything else falls through to a numpy step
    function (no gradient support through those ops).
    """

    def build_pre(self, ops, signals, config):
        self._process_steps = []
        for op in ops:
            process = op.process
            step_info = self._build_process(op, process, signals, config)
            self._process_steps.append(step_info)

    def _build_process(self, op, process, signals, config):
        dt = config.dt
        if isinstance(process, nengo.synapses.Lowpass):
            alpha = float(np.exp(-dt / process.tau))
            return {
                "type": "lowpass",
                "alpha": alpha,
                "op": op,
            }
        elif isinstance(process, nengo.synapses.Alpha):
            # Alpha synapse: two cascaded Lowpass filters
            alpha = float(np.exp(-dt / process.tau))
            return {
                "type": "alpha",
                "alpha": alpha,
                "op": op,
            }
        elif isinstance(process, nengo.synapses.LinearFilter):
            # General linear (IIR/FIR) filter – use scipy to get coefficients
            try:
                from scipy.signal import lfilter
                num, den = process.analog_coefficients(dt)
                return {
                    "type": "linear_filter",
                    "num": num,
                    "den": den,
                    "op": op,
                }
            except Exception:
                pass
            # Fallback to numpy
            return self._numpy_fallback(op, process, signals, config)
        else:
            return self._numpy_fallback(op, process, signals, config)

    def _numpy_fallback(self, op, process, signals, config):
        """Build a numpy step function as fallback."""
        dt = config.dt
        shape_in = op.input.shape if op.input is not None else (0,)
        shape_out = op.output.shape

        rng = config.rng if config.rng is not None else np.random.default_rng(0)

        fns = []
        for _ in range(config.minibatch_size):
            # Nengo 4.x: make_state is separate; build it first, then make_step
            if hasattr(process, "make_state"):
                state = process.make_state(shape_in, shape_out, dt)
            else:
                state = {}
            step_fn = process.make_step(shape_in, shape_out, dt, rng, state)
            fns.append(step_fn)

        return {
            "type": "numpy",
            "fns": fns,
            "op": op,
        }

    def build_step(self, ops, signals, config):
        for step_info in self._process_steps:
            op = step_info["op"]
            ptype = step_info["type"]

            if ptype == "lowpass":
                self._lowpass_step(step_info, signals, config)
            elif ptype == "alpha":
                self._alpha_step(step_info, signals, config)
            elif ptype == "linear_filter":
                self._linear_filter_step(step_info, signals, config)
            elif ptype == "numpy":
                self._numpy_step(step_info, signals, config)

    def _lowpass_step(self, step_info, signals, config):
        """First-order IIR filter: y[t] = alpha*y[t-1] + (1-alpha)*x[t]."""
        op = step_info["op"]
        alpha = step_info["alpha"]

        x = signals.gather(op.input).to(config.dtype)  # (batch, n)
        # State X stores previous output – may have extra leading dims like (batch, 1, n)
        state_sig = op.state.get("X")
        if state_sig is not None:
            y_prev_full = signals.gather(state_sig).to(config.dtype)
            # Flatten to match x for computation
            y_prev = y_prev_full.reshape(x.shape)
            y_new = alpha * y_prev + (1.0 - alpha) * x
            # Store back with original state shape
            signals.scatter(state_sig, y_new.reshape(y_prev_full.shape), mode="set")
        else:
            # No state – just scale
            y_new = x * (1.0 - alpha)

        signals.scatter(op.output, y_new, mode="set")

    def _alpha_step(self, step_info, signals, config):
        """Alpha synapse: cascaded lowpass filters.

        Nengo stores Alpha state as a single signal with shape (2, signal_dim)
        where index 0 is the first-stage state and index 1 the second-stage.
        """
        op = step_info["op"]
        alpha = step_info["alpha"]

        x = signals.gather(op.input).to(config.dtype)  # (batch, n)

        state = op.state
        state_keys = list(state.keys())

        if len(state_keys) >= 1:
            x_sig = state[state_keys[0]]
            # Gathered shape: (batch, 2, n) where dim -2 indexes the two stages
            X = signals.gather(x_sig).to(config.dtype)
            orig_shape = X.shape  # remember for scatter

            # Reshape to (batch, 2, n) if needed
            batch = x.shape[0]
            n = x.shape[-1]
            X = X.reshape(batch, 2, n)

            y1 = X[:, 0, :]   # first filter state  (batch, n)
            y2 = X[:, 1, :]   # second filter state (batch, n)

            y1_new = alpha * y1 + (1.0 - alpha) * x
            y2_new = alpha * y2 + (1.0 - alpha) * y1_new

            X_new = torch.stack([y1_new, y2_new], dim=1)  # (batch, 2, n)
            signals.scatter(x_sig, X_new.reshape(orig_shape), mode="set")
            signals.scatter(op.output, y2_new, mode="set")
        else:
            signals.scatter(op.output, x, mode="set")

    def _linear_filter_step(self, step_info, signals, config):
        """General linear IIR filter (numpy fallback without gradient)."""
        from scipy.signal import lfilter
        op = step_info["op"]
        num = step_info["num"]
        den = step_info["den"]

        x = signals.gather(op.input)  # (batch, n)
        batch = x.shape[0]
        x_np = x.detach().cpu().numpy()
        out_list = []
        for b in range(batch):
            out = lfilter(num, den, x_np[b])
            out_list.append(out)
        out_tensor = torch.tensor(
            np.stack(out_list), dtype=config.dtype, device=config.device
        )
        signals.scatter(op.output, out_tensor, mode="set")

    def _numpy_step(self, step_info, signals, config):
        """Call Nengo's numpy step function for unsupported processes.

        Handles different step function signatures dynamically:
        - ``fn(t)``              → returns ndarray (source process, e.g. WhiteSignal)
        - ``fn(t, signal)``      → returns ndarray (filter process with input)
        - ``fn(t, output)``      → fills output buffer in-place (no input)
        - ``fn(t, signal, out)`` → fills output buffer in-place (old-style)
        """
        op = step_info["op"]
        fns = step_info["fns"]
        batch = config.minibatch_size

        # Get time
        t_val = signals.gather(op.t)
        t = float(t_val.flatten()[0].item())

        x_np = None
        if op.input is not None:
            x = signals.gather(op.input)
            x_np = x.detach().cpu().numpy()

        # Introspect number of parameters once (all batch fns have same sig)
        try:
            sig = inspect.signature(fns[0])
            nparams = len(sig.parameters)
        except (ValueError, TypeError):
            nparams = 3  # assume old-style fn(t, signal, out)

        out_list = []
        for b in range(batch):
            out = np.zeros(op.output.shape, dtype=np.float32)
            if nparams == 1:
                # Source process: fn(t) → returns ndarray
                result = fns[b](t)
                if result is not None:
                    out[:] = np.asarray(result, dtype=np.float32).reshape(out.shape)
            elif nparams == 2:
                if x_np is not None:
                    # Filter process: fn(t, signal) → returns ndarray
                    result = fns[b](t, x_np[b])
                    if result is not None:
                        out[:] = np.asarray(result, dtype=np.float32).reshape(out.shape)
                else:
                    # No input: fn(t, out) → fills buffer in-place
                    fns[b](t, out)
            else:
                # Old-style: fn(t, signal, out) → fills buffer in-place
                if x_np is not None:
                    fns[b](t, x_np[b], out)
                else:
                    fns[b](t, out)
            out_list.append(out)

        out_tensor = torch.tensor(
            np.stack(out_list), dtype=config.dtype, device=config.device
        )
        signals.scatter(op.output, out_tensor, mode="set")
