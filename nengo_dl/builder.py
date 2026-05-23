"""Builder infrastructure for nengo-dl (PyTorch backend).

Each Nengo operator type has a corresponding ``OpBuilder`` subclass that
translates the operator into PyTorch computations.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Type


@dataclasses.dataclass
class BuildConfig:
    """Configuration passed to operator builders during build.

    Parameters
    ----------
    dt : float
        Simulation timestep in seconds.
    minibatch_size : int
        Number of samples per minibatch.
    training : bool
        Whether the network is in training mode.
    rate_mode : bool
        Whether spiking neuron builders should use rate approximations during
        inference. Training also uses rate approximations.
    lif_smoothing : float
        Smoothing for LIF surrogate gradients. 0 = pure spiking,
        positive values give smoother rate approximations.
    inference_only : bool
        If True, skip training-specific operators.
    device : torch.device
        Computation device.
    dtype : torch.dtype
        Default floating-point dtype.
    rng : numpy.random.Generator or None
        Random number generator for stochastic ops.
    """

    dt: float = 0.001
    minibatch_size: int = 1
    training: bool = False
    rate_mode: bool = False
    lif_smoothing: float = 0.0
    inference_only: bool = False
    device: Any = None
    dtype: Any = None
    rng: Any = None


class OpBuilder:
    """Base class for all operator builders.

    Subclasses translate a list of *merged* Nengo operators of the same
    type into PyTorch operations.  The lifecycle is:

    1. ``build_pre(ops, signals, config)`` – called once after the signal
       dict is populated but before the simulation loop starts.  Use this
       to pre-compute constants and create any required tensors/parameters.

    2. ``build_step(ops, signals, config)`` – called every timestep.
       Reads from ``signals.gather(sig)`` and writes back via
       ``signals.scatter(sig, val)``.

    3. ``build_post(ops, signals, config)`` – called once after the
       simulation loop.  Rarely needed; use for cleanup.
    """

    def build_pre(self, ops, signals, config: BuildConfig):
        """Pre-build: set up tensors / parameters used every step."""

    def build_step(self, ops, signals, config: BuildConfig):
        """Execute one timestep for all operators in the group."""
        raise NotImplementedError

    def build_post(self, ops, signals, config: BuildConfig):
        """Post-build: any cleanup after simulation."""

    @staticmethod
    def mergeable(x, y) -> bool:
        """Return True if operators *x* and *y* can be merged.

        Merged operators are executed together in a single ``build_step``
        call which may be more efficient (fewer kernel launches).
        """
        return False


class Builder:
    """Registry and orchestrator for ``OpBuilder`` instances.

    Usage::

        @Builder.register(SomeNengoOperator)
        class SomeOpBuilder(OpBuilder):
            ...
    """

    #: Maps Nengo operator type → OpBuilder class.
    _registry: Dict[type, Type[OpBuilder]] = {}

    @classmethod
    def register(cls, op_type: type):
        """Decorator to register a builder for an operator type."""
        def decorator(builder_cls: Type[OpBuilder]):
            cls._registry[op_type] = builder_cls
            return builder_cls
        return decorator

    @classmethod
    def get_builder_cls(cls, op_type: type) -> Optional[Type[OpBuilder]]:
        """Return the builder class for a given operator type, or None."""
        return cls._registry.get(op_type)

    def __init__(self, operators: List, signals, config: BuildConfig):
        """Create and pre-build all operator builders.

        Parameters
        ----------
        operators : list
            List of Nengo operators in execution order.
        signals : SignalDict
            The signal dictionary.
        config : BuildConfig
            Build configuration.
        """
        self.config = config
        self.signals = signals
        # Group consecutive operators of the same type that are mergeable
        self._groups: List = self._group_operators(operators)
        # Instantiate a builder for each group
        self._builders: List[OpBuilder] = []
        for op_type, ops in self._groups:
            builder_cls = Builder.get_builder_cls(op_type)
            if builder_cls is None:
                raise ValueError(
                    f"No builder registered for operator type {op_type.__name__}. "
                    "If you are using a custom operator, register a builder with "
                    "@Builder.register(YourOperatorType)."
                )
            b = builder_cls()
            b.build_pre(ops, signals, config)
            self._builders.append(b)

    @staticmethod
    def _group_operators(operators: List):
        """Group operators by type (simple sequential grouping).

        Returns list of (op_type, [ops]) tuples in execution order.
        """
        if not operators:
            return []
        groups = []
        current_type = type(operators[0])
        current_ops = [operators[0]]
        for op in operators[1:]:
            if (type(op) is current_type and
                    Builder._registry.get(current_type, OpBuilder).mergeable(current_ops[0], op)):
                current_ops.append(op)
            else:
                groups.append((current_type, current_ops))
                current_type = type(op)
                current_ops = [op]
        groups.append((current_type, current_ops))
        return groups

    def run_step(self):
        """Execute one simulation timestep (all operators in order)."""
        for (op_type, ops), builder in zip(self._groups, self._builders):
            builder.build_step(ops, self.signals, self.config)

    def build_post_all(self):
        """Call build_post on all builders."""
        for (op_type, ops), builder in zip(self._groups, self._builders):
            builder.build_post(ops, self.signals, self.config)
