#!/usr/bin/env bash
# Fetch a MoE checkpoint into the local Hugging Face cache.
#
# Defaults to OLMoE, pinned to the exact revision the routing numbers in
# HANDOVER.md were measured on. Weights are not in this repo -- they are 13 GB
# and GitHub caps files at 100 MB, so we pin a revision instead of shipping bytes.
#
#   ./scripts/download-model.sh                       # OLMoE at the pinned revision
#   ./scripts/download-model.sh <repo_id> [revision]  # anything else
#
# Set HF_TOKEN in the environment for gated repos.
set -euo pipefail

MODEL="${1:-allenai/OLMoE-1B-7B-0924-Instruct}"
# Pinned so everyone measures against identical weights; pass a revision (or
# "main") as $2 to override.
DEFAULT_REVISION="7f1c97f440f06ce36705e4f2b843edb5925f4498"
if [ "$MODEL" = "allenai/OLMoE-1B-7B-0924-Instruct" ]; then
  REVISION="${2:-$DEFAULT_REVISION}"
else
  REVISION="${2:-main}"
fi

cd "$(dirname "$0")/.."
if [ -x .venv/bin/hf ]; then
  HF=.venv/bin/hf
elif command -v hf >/dev/null 2>&1; then
  HF=hf
else
  echo "error: 'hf' not found. Create the venv first:" >&2
  echo "  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ." >&2
  exit 1
fi

echo "model    : $MODEL"
echo "revision : $REVISION"
echo "cache    : ${HF_HOME:-$HOME/.cache/huggingface}"
echo

# Already-present files are skipped, so re-running this is cheap.
"$HF" download "$MODEL" --revision "$REVISION"

echo
echo "Done. Load it with:"
echo "  PYTHONPATH=. .venv/bin/python -m moe_inspector.server --port 8731 --model $MODEL"
