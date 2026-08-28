"""Locate the routing machinery inside a Mixture-of-Experts model.

Every HF MoE implementation follows the same shape: a *router* (usually a plain
Linear producing one logit per expert) picks top-k out of a bag of *experts*.
The module names differ per architecture, so we find them structurally instead
of hardcoding paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from torch import nn

# Config keys used by various architectures for the same two numbers.
_NUM_EXPERT_KEYS = (
    "num_local_experts",
    "num_experts",
    "n_routed_experts",
    "moe_num_experts",
    "num_experts_per_layer",
)
_TOPK_KEYS = (
    "num_experts_per_tok",
    "moe_top_k",
    "num_experts_per_token",
    "n_activated_experts",
    "top_k",
)

_LAYER_PATH_RE = re.compile(r"^(.*\.(?:layers|h|blocks|block)\.\d+)(?:\.|$)")


@dataclass
class MoELayer:
    """One MoE block, plus everything we need to hook it."""

    index: int
    block_path: str
    router_path: str | None
    num_experts: int
    top_k: int | None
    decoder_layer_path: str | None
    expert_paths: list[str] = field(default_factory=list)
    router_kind: str = "linear"  # linear | module | none
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "block_path": self.block_path,
            "router_path": self.router_path,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "decoder_layer_path": self.decoder_layer_path,
            "router_kind": self.router_kind,
            "can_mask": self.router_path is not None,
            "can_sleep": self.decoder_layer_path is not None,
            "notes": self.notes,
        }


def _config_lookup(config: Any, keys: tuple[str, ...]) -> int | None:
    """Search a config (and nested sub-configs like DBRX's ffn_config) for a key."""
    seen: set[int] = set()
    stack = [config]
    while stack:
        node = stack.pop(0)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        for key in keys:
            value = getattr(node, key, None)
            if isinstance(value, int) and value > 0:
                return value
        for value in vars(node).values():
            if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
                stack.append(value)
    return None


def _find_router(block: nn.Module, num_experts: int) -> tuple[str | None, str]:
    """Find the module inside `block` that emits one score per expert."""
    linear_hits: list[tuple[int, str]] = []
    module_hits: list[tuple[int, str]] = []

    for name, sub in block.named_modules():
        if not name or name.count(".") > 2:
            continue
        leaf = name.rsplit(".", 1)[-1].lower()
        # Prefer things that are actually called gate/router.
        rank = 0 if any(k in leaf for k in ("gate", "router", "wg")) else 1

        if isinstance(sub, nn.Linear):
            if sub.out_features == num_experts:
                linear_hits.append((rank, name))
            continue

        # DeepSeek-style MoEGate: a bare Parameter of shape (num_experts, hidden).
        weight = getattr(sub, "weight", None)
        if weight is not None and getattr(weight, "ndim", 0) == 2 and weight.shape[0] == num_experts:
            module_hits.append((rank, name))

    if linear_hits:
        return min(linear_hits)[1], "linear"
    if module_hits:
        return min(module_hits)[1], "module"
    return None, "none"


def _decoder_layer_path(block_path: str) -> str | None:
    match = _LAYER_PATH_RE.match(block_path)
    return match.group(1) if match else None


def find_moe_layers(model: nn.Module) -> list[MoELayer]:
    """Walk the model and return every MoE block in forward order."""
    config = getattr(model, "config", None)
    cfg_experts = _config_lookup(config, _NUM_EXPERT_KEYS)
    cfg_topk = _config_lookup(config, _TOPK_KEYS)

    layers: list[MoELayer] = []
    for path, module in model.named_modules():
        # An `experts` submodule is the one reliable marker of an MoE block —
        # matching on class names instead would catch every module in models
        # whose whole architecture is named *Moe* (Qwen3MoeAttention, etc).
        experts = getattr(module, "experts", None)
        if not isinstance(experts, nn.Module):
            continue
        # Skip the expert container itself; we want the block that owns it.
        if isinstance(module, (nn.ModuleList, nn.Sequential)):
            continue

        notes: list[str] = []
        expert_paths: list[str] = []
        if isinstance(experts, nn.ModuleList):
            num_experts = len(experts)
            expert_paths = [f"{path}.experts.{i}" for i in range(num_experts)]
        else:
            # transformers>=5 stores experts as fused 3D tensors, so there is no
            # per-expert submodule; ablation happens at the router instead.
            num_experts = getattr(experts, "num_experts", None) or cfg_experts or 0

        if not num_experts:
            continue

        router_path, router_kind = _find_router(module, num_experts)
        if router_kind == "none":
            notes.append(f"{path}: no router found, expert masking unavailable")
        else:
            router = get_module(module, router_path)
            # Sigmoid + group-limited routers (DeepSeek-V3 family) don't derive
            # their weights from a plain softmax, so a recomputed reroute is an
            # approximation of what the model would have done.
            if any(
                hasattr(router, attr)
                for attr in ("e_score_correction_bias", "routed_scaling_factor", "n_group")
            ):
                notes.append(
                    f"{type(router).__name__}: grouped/sigmoid routing — "
                    "reroute weights are approximated (silence mode is exact)"
                )

        top_k = None
        for attr in ("top_k", "num_experts_per_tok", "moe_top_k"):
            value = getattr(module, attr, None)
            if isinstance(value, int) and value > 0:
                top_k = value
                break
        top_k = top_k or cfg_topk

        if getattr(module, "shared_expert", None) is not None or getattr(
            module, "shared_experts", None
        ) is not None:
            notes.append("has a shared expert that always runs (not maskable)")

        layers.append(
            MoELayer(
                index=len(layers),
                block_path=path,
                router_path=f"{path}.{router_path}" if router_path else None,
                num_experts=num_experts,
                top_k=top_k,
                decoder_layer_path=_decoder_layer_path(path),
                expert_paths=expert_paths,
                router_kind=router_kind,
                notes=notes,
            )
        )

    return layers


def get_module(model: nn.Module, path: str) -> nn.Module:
    module: nn.Module = model
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module
