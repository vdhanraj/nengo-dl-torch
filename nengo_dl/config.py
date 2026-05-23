"""Configuration system for nengo-dl (PyTorch backend).

Usage::

    with nengo.Network() as net:
        nengo_dl.configure_settings(stateful=False, lif_smoothing=0.01)
        ...
"""

import nengo


# ---------------------------------------------------------------------------
# Network-level configuration store
# ---------------------------------------------------------------------------

# We store global defaults here for when configure_settings is called outside
# a network context (uncommon but harmless).
_global_settings = {}


def configure_settings(
    trainable=None,
    inference_only=None,
    lif_smoothing=None,
    dtype=None,
    keep_history=None,
    stateful=None,
    use_loop=None,
):
    """Configure nengo-dl settings on the current Network.

    Must be called inside a ``with nengo.Network() as net:`` block.

    Parameters
    ----------
    trainable : bool, optional
        Whether parameters are trainable by default.
    inference_only : bool, optional
        If True, skip training-specific ops.
    lif_smoothing : float, optional
        Smoothing for LIF surrogate gradients (0 = spiking, >0 = smoothed rate).
    dtype : str, optional
        Default float dtype (''float32'' or ''float64'').
    keep_history : bool, optional
        Keep probe data from all timesteps; if False, only the last timestep.
    stateful : bool, optional
        If True, preserve simulation state between ``run_steps`` calls.
        Default is False (state is reset between calls).
    use_loop : bool, optional
        Unused; kept for API compatibility.
    """
    # Try to get the active Nengo network config
    try:
        ctx = nengo.Config.context
        cfg = ctx[-1] if len(ctx) > 0 else None
    except Exception:
        cfg = None

    settings = {
        "trainable": trainable,
        "inference_only": inference_only,
        "lif_smoothing": lif_smoothing,
        "dtype": dtype,
        "keep_history": keep_history,
        "stateful": stateful,
    }

    if cfg is not None:
        for key, val in settings.items():
            if val is None:
                continue

            # --- Register on Network (not in default_config; configures() required)
            try:
                if nengo.Network not in cfg.params:
                    cfg.configures(nengo.Network)
            except Exception:
                pass
            try:
                cp_net = cfg.params.get(nengo.Network)
                if cp_net is not None:
                    if key not in cp_net._extra_params:
                        cp_net._extra_params[key] = nengo.params.Parameter(
                            key, default=None, optional=True
                        )
                    cp_net._extra_params[key].set_default(cp_net, val)
            except Exception:
                pass

            # --- For "trainable", also register on Ensemble/Connection/Node so that
            # net.config[nengo.Ensemble].trainable = True/False works (original API).
            # Inject directly into _extra_params to avoid ClassParams.set_param()
            # validation issues across Nengo versions.
            if key == "trainable":
                _tp = nengo.params.Parameter("trainable", default=None, optional=True)
                for obj_type in [nengo.Ensemble, nengo.Connection, nengo.Node]:
                    try:
                        cp = cfg.params.get(obj_type)
                        if cp is None:
                            cfg.configures(obj_type)
                            cp = cfg.params.get(obj_type)
                        if cp is not None:
                            if "trainable" not in cp._extra_params:
                                cp._extra_params["trainable"] = _tp
                            # Set network-level default; user can override per-type
                            cp._extra_params["trainable"].set_default(cp, val)
                    except Exception:
                        pass

    # Always stash in global settings (fallback for Simulator to read)
    for key, val in settings.items():
        if val is not None:
            _global_settings[key] = val


def get_setting(network_or_model, setting, default=None):
    """Retrieve a nengo-dl setting from a Network or Model.

    Parameters
    ----------
    network_or_model : nengo.Network or nengo.builder.Model
        The network or model to read settings from.
    setting : str
        The setting name.
    default : any, optional
        Value to return if the setting is not found.
    """
    # Try reading from network config
    try:
        import nengo.builder
        if isinstance(network_or_model, nengo.builder.Model):
            network = network_or_model.toplevel
        else:
            network = network_or_model

        if network is not None:
            val = getattr(network.config[type(network)], setting, None)
            if val is not None:
                return val
    except Exception:
        pass

    # Fall back to global settings
    return _global_settings.get(setting, default)
