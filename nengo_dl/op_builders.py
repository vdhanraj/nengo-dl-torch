"""Builders for core Nengo operators (PyTorch backend).

Handles: TimeUpdate, Reset, Copy, ElementwiseInc, DotInc, SimPyFunc,
SimProbe.
"""

import numpy as np
import torch

from nengo.builder.operator import (
    Copy,
    DotInc,
    ElementwiseInc,
    Reset,
    SimPyFunc,
    TimeUpdate,
)
from nengo.builder.probe import SimProbe

from .builder import Builder, BuildConfig, OpBuilder


# ---------------------------------------------------------------------------
# TimeUpdate
# ---------------------------------------------------------------------------

@Builder.register(TimeUpdate)
class TimeUpdateBuilder(OpBuilder):
    """Updates the simulation step counter and time signal."""

    def build_pre(self, ops, signals, config):
        self._step_sig = ops[0].step
        self._time_sig = ops[0].time
        self._dt = config.dt

    def build_step(self, ops, signals, config):
        step = signals.gather(self._step_sig)
        new_step = step + 1
        signals.scatter(self._step_sig, new_step, mode="set")
        signals.scatter(self._time_sig, (new_step.float() * self._dt), mode="set")


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

@Builder.register(Reset)
class ResetBuilder(OpBuilder):
    """Resets a signal to a constant value."""

    def build_pre(self, ops, signals, config):
        self._resets = []
        for op in ops:
            dst = op.dst
            # Pre-compute the reset value as a tensor
            val = op.value
            if not isinstance(val, np.ndarray):
                val = np.full(dst.shape, val, dtype=float)
            val_tensor = torch.tensor(val, dtype=config.dtype, device=config.device)
            self._resets.append((dst, val_tensor))

    def build_step(self, ops, signals, config):
        for dst, val in self._resets:
            signals.scatter(dst, val, mode="set")


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

@Builder.register(Copy)
class CopyBuilder(OpBuilder):
    """Copies one signal to another (with optional slicing/reshaping)."""

    def build_pre(self, ops, signals, config):
        self._copies = [(op.src, op.dst, op.inc) for op in ops]

    def build_step(self, ops, signals, config):
        for src, dst, inc in self._copies:
            val = signals.gather(src)
            # Reshape to destination shape if needed
            if val.shape[1:] != dst.shape:
                batch = val.shape[0] if val.dim() > len(src.shape) else 1
                val = val.reshape(batch, *dst.shape)
            mode = "inc" if inc else "set"
            signals.scatter(dst, val, mode=mode)


# ---------------------------------------------------------------------------
# ElementwiseInc
# ---------------------------------------------------------------------------

@Builder.register(ElementwiseInc)
class ElementwiseIncBuilder(OpBuilder):
    """Performs element-wise multiply-accumulate: Y += A * X."""

    def build_pre(self, ops, signals, config):
        self._ops = ops

    def build_step(self, ops, signals, config):
        for op in self._ops:
            A = signals.gather(op.A)
            X = signals.gather(op.X)
            Y = signals.gather(op.Y)

            # A and X may need shape adjustment to broadcast with Y (batch, *shape).
            # - If A/X is a parameter (no batch dim): add leading batch dim
            # - If A/X is state (already has batch as dim 0) but has fewer total
            #   dims than Y: add trailing dim(s) to enable broadcasting.
            a_is_param = signals._is_param.get(id(op.A.base), True)
            x_is_param = signals._is_param.get(id(op.X.base), True)

            if A.dim() < Y.dim():
                if a_is_param:
                    # Param has no batch dim: prepend
                    while A.dim() < Y.dim():
                        A = A.unsqueeze(0)
                else:
                    # State already has batch as dim 0: append trailing dims
                    while A.dim() < Y.dim():
                        A = A.unsqueeze(-1)

            if X.dim() < Y.dim():
                if x_is_param:
                    while X.dim() < Y.dim():
                        X = X.unsqueeze(0)
                else:
                    while X.dim() < Y.dim():
                        X = X.unsqueeze(-1)

            signals.scatter(op.Y, Y + A * X, mode="set")


# ---------------------------------------------------------------------------
# DotInc
# ---------------------------------------------------------------------------

@Builder.register(DotInc)
class DotIncBuilder(OpBuilder):
    """Performs matrix-vector multiply-accumulate: Y += A @ X."""

    def build_pre(self, ops, signals, config):
        self._ops = ops

    def build_step(self, ops, signals, config):
        for op in self._ops:
            A = signals.gather(op.A)  # (m, n) or (batch, m, n)
            X = signals.gather(op.X)  # (n,) or (batch, n)
            Y = signals.gather(op.Y)  # (m,) or (batch, m)

            # Handle scalar/0-d cases
            if A.dim() == 0 or X.dim() == 0:
                result = A * X
                if result.dim() < Y.dim():
                    result = result.unsqueeze(0)
                signals.scatter(op.Y, Y + result, mode="set")
                continue

            # Determine batch dimension
            # A: typically (m, n) – shared weight matrix (parameter)
            # X: (batch, n) – batch of input vectors
            # Y: (batch, m) – batch of output vectors

            if A.dim() == 2 and X.dim() == 2:
                # Batched: Y += X @ A.T  (vectorized over batch)
                inc = torch.matmul(X, A.t())  # (batch, m)
            elif A.dim() == 2 and X.dim() == 1:
                # Unbatched X
                inc = torch.mv(A, X)  # (m,)
                inc = inc.unsqueeze(0)  # (1, m)
            elif A.dim() == 3:
                # A is batched: A:(batch, m, n), X:(batch, n)
                inc = torch.bmm(A, X.unsqueeze(-1)).squeeze(-1)  # (batch, m)
            elif A.dim() == 1 and X.dim() == 1:
                # Both 1-D: dot product
                inc = (A * X).sum().unsqueeze(0).unsqueeze(0)
            else:
                # Fallback: try generic matmul
                inc = torch.matmul(A, X.unsqueeze(-1)).squeeze(-1)
                if inc.dim() < Y.dim():
                    inc = inc.unsqueeze(0)

            signals.scatter(op.Y, Y + inc, mode="set")


# ---------------------------------------------------------------------------
# SimPyFunc  (Python-function Nodes)
# ---------------------------------------------------------------------------

@Builder.register(SimPyFunc)
class SimPyFuncBuilder(OpBuilder):
    """Executes Python-function Nodes.

    The function is called with numpy arrays (current time, input) and
    produces a numpy output that is converted to a tensor. Gradient flow
    through these nodes is not supported.
    """

    def build_pre(self, ops, signals, config):
        self._fn_ops = []
        for op in ops:
            self._fn_ops.append({
                "fn": op.fn,
                "t": op.t,
                "x": op.x,
                "output": op.output,
            })

    def build_step(self, ops, signals, config):
        for fn_op in self._fn_ops:
            fn = fn_op["fn"]
            t_sig = fn_op["t"]
            x_sig = fn_op["x"]
            out_sig = fn_op["output"]

            # Get current time (scalar)
            t_val = signals.gather(t_sig)
            t = float(t_val.flatten()[0].item())

            if x_sig is not None:
                x_tensor = signals.gather(x_sig)
                # Use only the first batch item for the function call
                x_np = x_tensor[0].detach().cpu().numpy()
                result = fn(t, x_np)
            else:
                result = fn(t)

            if result is None:
                continue

            result_np = np.asarray(result, dtype=np.float32)
            result_t = torch.tensor(
                result_np, dtype=config.dtype, device=config.device
            )
            # Broadcast across batch
            result_t = result_t.unsqueeze(0).expand(
                config.minibatch_size, *result_t.shape
            )
            signals.scatter(out_sig, result_t, mode="set")


# ---------------------------------------------------------------------------
# SimProbe
# ---------------------------------------------------------------------------

@Builder.register(SimProbe)
class SimProbeBuilder(OpBuilder):
    """Records probe data every timestep.

    The recorded tensors are stored in ``signals._probe_data`` which is
    a dict mapping signal → list of per-step tensors.
    """

    def build_pre(self, ops, signals, config):
        # Initialise probe data storage on the SignalDict
        if not hasattr(signals, "_probe_data"):
            signals._probe_data = {}
        for op in ops:
            sig = op.signal
            if sig not in signals._probe_data:
                signals._probe_data[sig] = []

    def build_step(self, ops, signals, config):
        for op in ops:
            sig = op.signal
            val = signals.gather(sig)
            # Detach if we are not computing gradients, to save memory
            signals._probe_data[sig].append(val)
