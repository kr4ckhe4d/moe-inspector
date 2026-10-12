"""Inspect and ablate Mixture-of-Experts models."""

__all__ = ["MoELayer", "find_moe_layers", "Inspector"]


def __getattr__(name):
    # Lazy, so the Strata view (moe_inspector.strata) runs without torch installed.
    if name in ("MoELayer", "find_moe_layers"):
        from . import detect
        return getattr(detect, name)
    if name == "Inspector":
        from .session import Inspector
        return Inspector
    raise AttributeError(name)
