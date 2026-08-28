# MoE Inspector

A GUI for looking inside a Mixture-of-Experts model while it generates — and for
switching parts of it off.

Every token in an MoE model is routed to a handful of experts by a small gating
network. This tool taps that router with forward hooks so you can:

- **watch** which experts fire, per layer, per token, as a heatmap
- **disable individual experts** and see how the output changes
- **put whole layers to sleep** (turn them into identity functions)
- **compare** an ablated run against the unmodified baseline, side by side

Nothing is written to the checkpoint. All of it is forward hooks on the live
model, so clearing the mask restores bit-identical behaviour (there's a test
for exactly that).

## Running it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
```

```bash
PYTHONPATH=. .venv/bin/python -m moe_inspector.server --port 8731
```

Then open http://127.0.0.1:8731 and load a model by its HF id. Pass
`--model <id>` to load one at startup instead.

### Models that fit in 16 GB

| model | total / active | shape |
|---|---|---|
| `ibm-granite/granite-3.1-3b-a800m-instruct` | 3.3B / 0.8B | 32 layers × 40 experts, top-8 |
| `allenai/OLMoE-1B-7B-0924-Instruct` | 6.9B / 1.3B | 16 layers × 64 experts, top-8 (tight in fp16) |
| `hf-internal-testing/tiny-random-MixtralForCausalLM` | tiny | 2 layers × 4 experts — for checking the wiring |

Anything larger (Mixtral-8x7B, Qwen3-30B-A3B, DeepSeek-V3) works too if you
have the RAM or a CUDA box; the code is architecture-agnostic.

## The three ablation modes

|  | what happens | use it for |
|---|---|---|
| **reroute** | the expert's router logit is forced to −inf, so tokens pick their next-best expert instead | "does this expert matter, or is a neighbour redundant with it?" |
| **silence** | routing is untouched, but the expert contributes zero — the token wastes a slot | "how much of the output actually comes from this expert?" |
| **sleep** | a whole decoder layer returns its input unchanged | "how load-bearing is this layer?" |

Sleep still *runs* the layer and throws the result away, so the KV cache stays
consistent — it's an ablation, not a speedup.

## Using the GUI

- **Click a cell** to disable that expert; drag to paint across many.
- **`z` next to a layer** puts the whole layer to sleep.
- **Click a token** in the strip to see only that token's routing; shift-click
  for a range, cmd/ctrl-click to add. "all tokens" resets.
- **Keep top-N / layer** disables everything except the N most-used experts in
  each layer, based on the last run — a quick way to find how much of the model
  a given prompt actually needs.
- **Compare vs baseline** runs the prompt twice, ablated and clean.
- **Export** writes the current mask to JSON.

## Architecture support

Blocks are found structurally (a module with an `experts` submodule) rather than
by name, and the router's output tuple is identified by dtype and shape, so tuple
ordering differences don't matter. Verified against `tests/test_architectures.py`:

```
PASS  mixtral       PASS  qwen3_moe    PASS  qwen2_moe    PASS  olmoe
PASS  granitemoe    PASS  gpt_oss      PASS  deepseek_v3
```

```bash
PYTHONPATH=. .venv/bin/python tests/test_architectures.py
```

**One caveat:** reroute recomputes the top-k from masked logits using
softmax-then-renormalise. That's exactly what Mixtral, Qwen, OLMoE, GraniteMoE
and GPT-OSS routers do. DeepSeek-V3's router uses sigmoid scoring with group
limiting and a correction bias, so its rerouted *weights* are an approximation —
the UI flags this when it detects such a router. Silence mode is exact everywhere.

Models with a **shared expert** that runs on every token (Qwen2-MoE, DeepSeek)
are flagged too: that expert isn't routed, so it can't be masked.

## Scripting it

The server is a thin wrapper over `Inspector`, which is usable on its own:

```python
from moe_inspector.session import Inspector

ins = Inspector()
ins._load("ibm-granite/granite-3.1-3b-a800m-instruct", "mps", "float16")

run = ins.generate("Write a haiku about routing.", max_new_tokens=64)
run["trace"][layer][token]  # -> the expert ids that fired

ins.set_mask({"5": [0, 1, 2]}, sleeping=[10], mode="reroute")
ablated = ins.generate("Write a haiku about routing.", max_new_tokens=64)
```

HTTP: `GET /api/status`, `POST /api/load`, `POST /api/mask`, `POST /api/generate`.
