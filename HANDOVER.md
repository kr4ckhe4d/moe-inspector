# Handover — moving to the AI PC

Written 2026-08-28, at the end of the session that built this. Everything below
is the state as pushed in the first commit.

> **Update, 2026-08-28 (on the new box).** It runs. Setup was corrected: the box
> is **AMD, not NVIDIA** — Radeon RX 9070 XT (Navi 48, `gfx1201`), ROCm 7.2.4,
> 15.92 GiB VRAM — so the CUDA-wheel instructions below were wrong and have been
> replaced. Test matrix passes, the server serves, and the full ablation pipeline
> is verified on the GPU in bf16. See *What is verified*.
>
> **A real model has now run.** OLMoE-1B-7B-0924-Instruct loads on the GPU at
> 12.89 GiB and generates coherent text at ~57 tok/s. The headline question below
> is answered: routing *is* legible and topic-separable — but not the way the
> heatmap-as-drawn shows it. See *First real findings*.

## The one thing to know

*(Superseded 2026-08-28 — a real model has now run. Kept for context; the answers
are in [First real findings](#first-real-findings).)*

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

### Verified on the AMD box (2026-08-28)

All seven architectures pass under `torch 2.13.0+rocm7.2` / `transformers 5.16.1`
(they run on CPU, so they prove the logic, not the GPU path). Separately, on the
GPU itself, with a tiny OLMoE config moved to `cuda` in `bfloat16`:

```
auto device resolves to: cuda      # torch exposes AMD through torch.cuda
bf16 matmul on gfx1201:  OK
hooks installed, logits identical to clean:   True
reroute / silence / sleep each change output: True
cleared mask restores bit-identical logits:   True
```

So the hook machinery, all three ablation modes, and the bit-identical restore
hold on ROCm in bf16 — not just on CPU. The server boots and serves
`/api/status` and the GUI on port 8731.

## First real findings

First real model, 2026-08-28: `allenai/OLMoE-1B-7B-0924-Instruct`, bf16 on the
GPU, 12.89 GiB resident, ~57 tok/s. 16 layers × 64 experts, top-8, `can_mask` and
`can_sleep` true on every layer, no detection warnings. Output is coherent, so
this is finally signal rather than noise from random weights.

### Raw utilisation looks uniform — and that is a trap

Aggregated over one 87-token run, routing is almost perfectly balanced:

```
entropy per layer   5.30 - 5.71 bits   (max = log2(64) = 6.00)
busiest expert      5 - 10% of traffic in its layer
never-used slots    40 / 1024
```

There are **no near-dead experts**. Read naively this says "no structure, nothing
is separable" — which is what the current one-run heatmap shows you. It is
misleading. OLMoE is trained with a load-balancing loss, so flat *marginal*
utilisation is the training objective being satisfied, not an absence of
specialisation. Don't conclude anything from a single run's heatmap.

### Conditional routing is strongly topic-specific

The structure is in *which* expert fires for *which* input, which only shows up
by contrasting runs. Six prompts, two each across three topics, ~185 tokens each;
per-layer expert distributions compared by Jensen-Shannon divergence. The
same-topic pairs are the control — they set the noise floor:

```
mean within-topic  JS = 0.0293 bits   <- noise floor
mean between-topic JS = 0.2643 bits   <- 9.02x the floor
```

Two things fall out of the per-layer breakdown:

- **Selectivity deepens with depth.** Layers 0-3 sit at 4-6x the floor; layers
  9-15 run 11-13.5x, peaking at **layer 9 (13.6x)**. Early layers route
  generically, later layers route by content.
- **The split is formal-vs-prose, not per-subject.** Code and maths route
  *similarly to each other* (JS 0.055-0.182) while poetry is the outlier against
  both (0.165-0.475). The model separates symbolic/technical from natural prose
  far more sharply than it separates code from maths.

### What this means for the tool

The "which areas are for what" question is answerable, but **the current one-run
view is the wrong instrument** — it shows the flat marginal that hides the effect.
The diff view floated at the bottom of this handover (routing for prompt A vs
prompt B on one grid) is no longer a nice-to-have; it is the feature that makes
the tool show what is actually there. Build it next, and put the per-layer
divergence number on screen — layer 9 being 13.6x is the kind of thing you want
ranked, not eyeballed.

Caveat on scope: one model, three topics, six prompts. The within-topic control
is what makes the 9x credible, but this is a first look, not a study.

## Setup on the new box

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
```

**This box is AMD, so the default torch is wrong.** `-e .` pulls
`torch==2.13.0+cu130` from PyPI, which is an NVIDIA build:
`torch.cuda.is_available()` comes back `False` and everything silently falls back
to CPU. Replace it with the ROCm build matching the installed ROCm (7.2.4 here):

```bash
uv pip install --python .venv/bin/python --reinstall-package torch torch --index-url https://download.pytorch.org/whl/rocm7.2
```

`--reinstall-package torch` is load-bearing. Without it uv sees a `torch` that
already satisfies the constraint, prints `Audited 1 package`, and changes
nothing — you are left on the CUDA wheel thinking you fixed it. Confirm with:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
# 2.13.0+rocm7.2 7.2.53211 True
```

The download is ~2 GB and took ~6 minutes. On an actual **NVIDIA** box none of
this applies — PyPI's default wheel is already CUDA and `-e .` alone is enough.

Then fetch the weights — `./scripts/download-model.sh` pulls OLMoE pinned to
revision `7f1c97f4`, the exact weights the findings above were measured on.

Then:

```bash
PYTHONPATH=. .venv/bin/python -m moe_inspector.server --port 8731
```

Pick `cuda` and `bfloat16` in the header (auto resolves to cuda but picks fp16).
`cuda` is still the correct choice on ROCm — PyTorch exposes AMD GPUs through the
`torch.cuda` API, and `_resolve_device` returns `"cuda"` here; there is no `hip`
or `rocm` device string to add. bf16 is confirmed working on `gfx1201`. Set
`HF_TOKEN` in the environment for gated repos.

If the box is headless, **don't** use `--host 0.0.0.0` — there is no auth and the
API loads arbitrary models. Tunnel instead:

```bash
ssh -N -L 8731:127.0.0.1:8731 you@ai-pc
```

### Model sizing

Rule of thumb: bf16 needs ~2 bytes per *total* parameter — MoE active-parameter
counts don't help, since every expert has to be resident. Add ~10-15% for
activations and cache.

This box has **15.92 GiB** of VRAM, which settles the table. Mind the units: a
"~14 GB" checkpoint is 14 decimal GB = 13.0 GiB, and the card is measured in GiB,
so the two numbers are closer than they look.

| model | total params | bf16 footprint | fits in 15.92 GiB? |
|---|---|---|---|
| `ibm-granite/granite-3.1-3b-a800m-instruct` | 3.3B | ~6.6 GB (6.2 GiB) | yes, comfortably |
| `allenai/OLMoE-1B-7B-0924-Instruct` | 6.9B | ~13.8 GB (12.9 GiB) | yes, ~3 GiB spare |
| `openai/gpt-oss-20b` | 21B | ~42 GB bf16, but ships MXFP4 at ~12 GB | only as MXFP4, tight |
| `Qwen/Qwen3-30B-A3B` | 30B | ~61 GB | no |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 47B | ~94 GB | no |

OLMoE is still the best first real target, and it does fit here: 16 layers × 64
experts, top-8, fully trained, small enough to iterate on in seconds. 64 experts
also makes for a much more readable heatmap than an 8-expert model. Keep
`max_new_tokens` modest on the first run — the ~3 GiB of headroom is what the KV
cache and activations have to live in.

The two "no" rows are hard blocks, not slow paths: with no `device_map="auto"`
and no quantization path (gaps 2 and 3 below), a model that doesn't fit on the
one GPU simply won't load.

## Known gaps, in the order they'll bite

1. **No GGUF support — your local models are all unusable.** `$LLAMA_DIR/models`
   (`~/llama.cpp/models`) holds 14 GGUFs, six of them MoE, and **not one can be
   inspected today.** Three separate reasons, and all three have to be solved:
   - `_load` only calls `from_pretrained(model_id, ...)` and never passes
     `gguf_file`, so there is no GGUF code path at all.
   - transformers 5.16.1 converts only `gemma4`, `gpt_oss`, `qwen2_moe` and
     `qwen3_moe`. Of the local files that covers `gpt-oss-20b-mxfp4` and
     `gemma-4-26B-A4B` only. `GLM-4.7-Flash` (`deepseek2`), `Laguna-XS-2.1`
     (`laguna`), `Qwen3-Coder-Next` (`qwen3next`) and `Qwen3.6-35B-A3B`
     (`qwen35moe`) are all unsupported. Note `qwen35moe` is *not* the
     `qwen3_moe` on the supported list — different architecture string.
   - Conversion **dequantises**: transformers unpacks GGUF back to full
     precision rather than keeping the quantisation. gpt-oss-20b becomes ~42 GB
     bf16 and gemma-4-26B ~52 GB, so neither fits the 15.92 GiB GPU. gpt-oss
     would fit in the 62 GB of system RAM, but CPU-only and slow.

   Net: the quantisation that makes these files practical for llama.cpp is
   exactly what makes them unusable here. Adding `gguf_file=` is easy; making
   the result *fit* is the real problem, and wiring in a quantisation path
   (gap 3) is the more useful fix.

2. **Single device only.** The loader does `model.to(device)`
   ([session.py:143](moe_inspector/session.py:143)) — there is no
   `device_map="auto"`. Anything that doesn't fit on one GPU won't load. Wiring
   accelerate in is a small change, but the sleep hook reads `hidden_states` off
   the layer's inputs, so check it still behaves across a device boundary.
3. **No quantization path.** No bitsandbytes / GPTQ wiring. If you add 4-bit,
   the thing to verify is router *detection*: `_find_router`
   ([detect.py](moe_inspector/detect.py)) matches either an `nn.Linear` whose
   `out_features == num_experts`, or a module with a 2-D `weight` of shape
   `(num_experts, hidden)`. Quantizers usually skip the tiny gate modules, so it
   will probably survive — but the symptom if it doesn't is a `notes` entry
   reading "no router found, expert masking unavailable", visible in the UI
   sidebar. Check that before assuming the mask is working.
4. **`trust_remote_code` is deliberately off.** transformers 5.x has native
   support for all seven architectures in the test matrix, so this only matters
   for models that ship custom modelling code. If you enable it, understand
   you're executing code from the repo.
5. **Trace is capped at 512 tokens** (`MAX_TRACE_TOKENS`,
   [session.py:24](moe_inspector/session.py:24)). Beyond that, routing stops
   being recorded — generation still completes, the heatmap just covers the first
   512 positions. Raising it grows the JSON payload as
   `layers × tokens × top_k × 2`; at 60 layers and top-8 that's already a few MB
   per run, so raise it deliberately.
6. **Batch size 1, greedy or plain sampling only.** `num_beams=1` is hardcoded
   ([session.py:390](moe_inspector/session.py:390)) because the trace assumes one
   row per token per forward pass. Beam search or `num_return_sequences > 1`
   would silently misalign the heatmap against the token strip.
7. **Generation blocks the request.** A long run makes the UI sit there. No
   streaming, no progress. Fine for 64 tokens, annoying at 512.
8. **Sleep isn't a speedup.** The layer still runs; its output is replaced with
   its input. That keeps the KV cache consistent, which is what makes it a clean
   ablation — but if you want actual "put unneeded areas to sleep" *performance*,
   that's a different piece of work (skipping the module entirely, and dealing
   with the cache).
9. **DeepSeek-V3 reroute is approximate.** Reroute recomputes top-k as
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
