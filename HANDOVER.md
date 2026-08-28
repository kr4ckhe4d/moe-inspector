# Handover — moving to the AI PC

Written 2026-08-28, at the end of the session that built this. Everything below
is the state as pushed in the first commit.

## The one thing to know

**This has never run on a real MoE model.** It was built on a 16 GB M-series Mac,
where nothing interesting fits. Every check so far used either
`hf-internal-testing/tiny-random-MixtralForCausalLM` or models built from tiny
configs with **random weights**. So:

- The plumbing is verified — routing is captured, masks change the output, and
  clearing a mask restores bit-identical logits.
- **Nothing about the *findings* is verified.** Every heatmap produced so far is
  noise from untrained weights. Whether the routing patterns are legible, whether
  disabling an expert produces an interesting degradation rather than instant
  gibberish, whether "areas" of a model are separable at all — all still open.

That's the point of the move. Get a real model in and look at an actual heatmap
before touching any of the code.

## What is verified

`tests/test_architectures.py` builds a tiny random model per architecture and
asserts, for each: MoE blocks are found, routing is captured, reroute never lets
a blocked expert appear, silence leaves routing untouched but changes the output,
sleeping a layer changes the output, and clearing the mask reproduces the
hook-free logits exactly.

```
PASS  mixtral   qwen3_moe   qwen2_moe   olmoe   granitemoe   gpt_oss   deepseek_v3
```

```bash
PYTHONPATH=. .venv/bin/python tests/test_architectures.py
```

Run this first on the new box — it takes seconds, downloads nothing, and catches
anything the environment change broke.

The GUI itself was driven end-to-end in a browser: load → generate → heatmap →
click an expert off → re-run → output changes and the baseline still matches the
original. Working, but only ever against a 2-layer, 4-expert toy.

## Setup on the new box

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
```

On **Linux**, PyPI's torch already ships CUDA wheels, so the above is enough. On
**Windows** it is CPU-only and you need the CUDA index:

```bash
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
```

Then:

```bash
PYTHONPATH=. .venv/bin/python -m moe_inspector.server --port 8731
```

Pick `cuda` and `bfloat16` in the header (auto resolves to cuda but picks fp16;
bf16 is the better default on Ampere and later). Set `HF_TOKEN` in the
environment for gated repos.

If the box is headless, **don't** use `--host 0.0.0.0` — there is no auth and the
API loads arbitrary models. Tunnel instead:

```bash
ssh -N -L 8731:127.0.0.1:8731 you@ai-pc
```

### Model sizing

Rule of thumb: bf16 needs ~2 bytes per *total* parameter — MoE active-parameter
counts don't help, since every expert has to be resident. Add ~10-15% for
activations and cache.

| model | total params | bf16 footprint |
|---|---|---|
| `allenai/OLMoE-1B-7B-0924-Instruct` | 6.9B | ~14 GB |
| `openai/gpt-oss-20b` | 21B | ~42 GB (ships MXFP4, much smaller) |
| `Qwen/Qwen3-30B-A3B` | 30B | ~61 GB |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 47B | ~94 GB |

OLMoE is the best first real target regardless of how much VRAM you have: 16
layers × 64 experts, top-8, fully trained, small enough to iterate on in seconds.
64 experts also makes for a much more readable heatmap than an 8-expert model.

## Known gaps, in the order they'll bite

1. **Single device only.** The loader does `model.to(device)`
   ([session.py:143](moe_inspector/session.py:143)) — there is no
   `device_map="auto"`. Anything that doesn't fit on one GPU won't load. Wiring
   accelerate in is a small change, but the sleep hook reads `hidden_states` off
   the layer's inputs, so check it still behaves across a device boundary.
2. **No quantization path.** No bitsandbytes / GPTQ wiring. If you add 4-bit,
   the thing to verify is router *detection*: `_find_router`
   ([detect.py](moe_inspector/detect.py)) matches either an `nn.Linear` whose
   `out_features == num_experts`, or a module with a 2-D `weight` of shape
   `(num_experts, hidden)`. Quantizers usually skip the tiny gate modules, so it
   will probably survive — but the symptom if it doesn't is a `notes` entry
   reading "no router found, expert masking unavailable", visible in the UI
   sidebar. Check that before assuming the mask is working.
3. **`trust_remote_code` is deliberately off.** transformers 5.x has native
   support for all seven architectures in the test matrix, so this only matters
   for models that ship custom modelling code. If you enable it, understand
   you're executing code from the repo.
4. **Trace is capped at 512 tokens** (`MAX_TRACE_TOKENS`,
   [session.py:24](moe_inspector/session.py:24)). Beyond that, routing stops
   being recorded — generation still completes, the heatmap just covers the first
   512 positions. Raising it grows the JSON payload as
   `layers × tokens × top_k × 2`; at 60 layers and top-8 that's already a few MB
   per run, so raise it deliberately.
5. **Batch size 1, greedy or plain sampling only.** `num_beams=1` is hardcoded
   ([session.py:390](moe_inspector/session.py:390)) because the trace assumes one
   row per token per forward pass. Beam search or `num_return_sequences > 1`
   would silently misalign the heatmap against the token strip.
6. **Generation blocks the request.** A long run makes the UI sit there. No
   streaming, no progress. Fine for 64 tokens, annoying at 512.
7. **Sleep isn't a speedup.** The layer still runs; its output is replaced with
   its input. That keeps the KV cache consistent, which is what makes it a clean
   ablation — but if you want actual "put unneeded areas to sleep" *performance*,
   that's a different piece of work (skipping the module entirely, and dealing
   with the cache).
8. **DeepSeek-V3 reroute is approximate.** Reroute recomputes top-k as
   softmax-then-renormalise, which is exactly right for Mixtral/Qwen/OLMoE/
   GraniteMoE/GPT-OSS. DeepSeek's sigmoid + group-limited router with a
   correction bias doesn't work that way, so its rerouted *weights* are an
   approximation. The UI flags it. Silence mode is exact everywhere.

## Code map

- [detect.py](moe_inspector/detect.py) — finds MoE blocks by structure (a module
  with an `experts` submodule). Deliberately **not** by class name: models like
  Qwen3-MoE name every module `Qwen3Moe*`, so name matching caught attention
  layers and norms. If a new architecture isn't detected, this is the file.
- [session.py](moe_inspector/session.py) — the hooks. `_make_router_hook` does
  both capture and ablation. Routers in transformers 5.x return
  `(logits, weights, indices)` but **the order varies** (GraniteMoe returns
  indices first), so `_router_slots` identifies them by dtype and shape.
- [server.py](moe_inspector/server.py) — thin FastAPI wrapper. `Inspector` is
  usable directly from a script; see the README.
- [static/index.html](moe_inspector/static/index.html) — the whole GUI, vanilla
  JS, no build step. The heatmap is aggregated client-side from the trace, which
  is why token-subset selection is instant.

## Worth trying first

1. Load OLMoE, run a prompt, and just **look**. Are some experts near-dead? Do
   the same experts fire for related tokens? Is there visible structure per layer?
2. Select a single token in the strip and compare its routing against a
   semantically different one. This is the closest thing to "which area handles
   what" the tool currently offers.
3. Use **keep top-N per layer** with N = 4, 2, 1 and watch where output quality
   falls off a cliff. That's the cheapest read on how much redundancy is there.
4. Contrast **reroute** vs **silence** on the same expert. If reroute barely
   changes the output but silence wrecks it, a neighbouring expert is covering —
   which is the actually interesting result.
5. Sleep individual layers from the top down. Later layers usually tolerate it
   better than early ones; worth confirming per-model.

If the heatmaps turn out legible, the obvious next feature is a diff view —
routing for prompt A vs prompt B on the same grid — which would answer the
"which areas are for what" question far more directly than the current one-run
view.
