"""Check detection + ablation against every MoE architecture we care about.

Models are built from tiny configs with random weights, so this runs in seconds
and downloads nothing.
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from moe_inspector.detect import find_moe_layers
from moe_inspector.session import Inspector

BASE = dict(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
)

ARCHITECTURES = {
    "mixtral": dict(num_local_experts=8, num_experts_per_tok=2),
    "qwen3_moe": dict(
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        decoder_sparse_step=1,
    ),
    "qwen2_moe": dict(
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, decoder_sparse_step=1,
    ),
    "olmoe": dict(num_experts=8, num_experts_per_tok=2),
    "granitemoe": dict(num_local_experts=8, num_experts_per_tok=2),
    "gpt_oss": dict(num_local_experts=8, num_experts_per_tok=2, head_dim=8),
    "deepseek_v3": dict(
        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
        moe_intermediate_size=32, first_k_dense_replace=0, n_group=1, topk_group=1,
        q_lora_rank=None, kv_lora_rank=16, qk_rope_head_dim=8, v_head_dim=8,
        qk_nope_head_dim=8, num_key_value_heads=4,
    ),
}


def build(name: str, overrides: dict):
    base = {k: v for k, v in BASE.items() if k not in overrides}
    config = AutoConfig.for_model(name, **base, **overrides)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def check(name: str, overrides: dict) -> tuple[bool, str]:
    try:
        model = build(name, overrides)
    except Exception as exc:
        return False, f"could not build: {type(exc).__name__}: {exc}"

    layers = find_moe_layers(model)
    if not layers:
        return False, "no MoE layers detected"

    inspector = Inspector()
    inspector.model, inspector.layers, inspector.state = model, layers, "ready"
    inspector._install_hooks()

    ids = torch.randint(0, 100, (1, 12))
    inspector._capture = True
    with torch.no_grad():
        model(ids)
    inspector._capture = False
    used = {e for row in inspector._buf.get(0, []) for e in row[0]}
    if not used:
        return False, "captured no routing"

    n_experts = layers[0].num_experts
    blocked = sorted(used)[:2]

    # reroute: blocked experts must not appear in the routing at all
    inspector.set_mask({"0": blocked}, [], "reroute")
    inspector._buf = {}
    inspector._capture = True
    with torch.no_grad():
        out_reroute = model(ids).logits
    inspector._capture = False
    still = {e for row in inspector._buf.get(0, []) for e in row[0]} & set(blocked)
    if still:
        return False, f"reroute leaked experts {sorted(still)}"

    # silence: routing is untouched, but the output must change
    inspector.set_mask({"0": blocked}, [], "silence")
    inspector._buf = {}
    inspector._capture = True
    with torch.no_grad():
        out_silence = model(ids).logits
    inspector._capture = False
    kept = {e for row in inspector._buf.get(0, []) for e in row[0]} & set(blocked)
    if not kept:
        return False, "silence changed routing (it should not)"

    # sleeping a whole layer must also change the output
    inspector.set_mask({}, [0], "reroute")
    with torch.no_grad():
        out_sleep = model(ids).logits

    # and clearing everything must restore the original output exactly
    inspector.set_mask({}, [], "reroute")
    with torch.no_grad():
        out_clean = model(ids).logits
    inspector._clear_hooks()
    with torch.no_grad():
        out_nohooks = model(ids).logits

    if not torch.equal(out_clean, out_nohooks):
        return False, "cleared mask did not restore original behaviour"
    for label, tensor in (("reroute", out_reroute), ("silence", out_silence), ("sleep", out_sleep)):
        if torch.equal(tensor, out_clean):
            return False, f"{label} had no effect on the output"

    return True, (
        f"{len(layers)} layers x {n_experts} experts, top-{layers[0].top_k}, "
        f"router={layers[0].router_path.split('.')[-1]}"
    )


def main() -> int:
    failures = 0
    for name, overrides in ARCHITECTURES.items():
        try:
            ok, detail = check(name, overrides)
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"

        print(f"{'PASS' if ok else 'FAIL'}  {name:<14} {detail}")
        failures += not ok
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
