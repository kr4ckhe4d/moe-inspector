"""Strata routing traces and expert-cache profiles, for the /strata view.

Strata (github.com/Niko1221/Strata) runs Qwen3.8-Flash-Next, 48 layers x 512
experts, top-10, far too big to load here with transformers. Its engine can
write the real routing instead (`--dump-routing FILE`), and its expert cache
is seeded from a ranked profile. This module reads both; nothing here needs
torch.

Trace format, one record per (position, layer), little-endian:
    int32 layer, int32 k, int32 ids[k], float32 weights[k]
Records come in verify windows, layer-major: layer 0 for each of the window's
T positions, then layer 1 for the same T, ... A drop in the layer number starts
the next window. Drafted positions the verifier rejected are routed too, so a
trace counts routed positions, not only kept tokens.

Profile format (`tools/make_profile.py`): b"STRP", then uint32 version,
n_layer, n_expert, slots, n, then n (uint16 layer, uint16 expert) pairs, most
wanted first. The first `slots` pairs are what a cache of that size holds.
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path

import numpy as np

DEFAULT_DIRS = [
    Path("/mnt/fast/strata-exp"),
    Path.home() / "code/local-llm-benchmarks/benchmarks/strata-expert-modes",
    Path.home() / "code/Strata/data",
]


def _kind(path: Path) -> str | None:
    """"profile", "trace" or None (other .bin files sit beside them, e.g. Strata's draft_vocab*.bin)."""
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:4] == b"STRP":
        return "profile"
    if len(head) == 8:
        layer, k = struct.unpack("<ii", head)
        if layer == 0 and 1 <= k <= 64 and path.stat().st_size % (8 + 8 * k) == 0:
            return "trace"
    return None


def list_files(dirs: list[Path]) -> dict:
    out = {"trace": [], "profile": []}
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.bin")):
            try:
                kind = _kind(p)
            except OSError:
                continue
            if kind:
                out[kind].append({"name": p.stem, "path": str(p), "mb": round(p.stat().st_size / 1e6, 1)})
    return {"traces": out["trace"], "profiles": out["profile"]}


def read_trace(path: Path) -> dict:
    """Positions x layers x k expert ids, rebuilt from the window-ordered records."""
    blob = Path(path).read_bytes()
    if len(blob) < 8:
        raise ValueError(f"{path}: empty trace")
    _, k = struct.unpack_from("<ii", blob, 0)
    rec = 8 + 8 * k
    if len(blob) % rec:
        raise ValueError(f"{path}: not a fixed-k trace (k={k})")
    raw = np.frombuffer(blob, dtype="<i4").reshape(-1, rec // 4)
    if (raw[:, 1] != k).any():
        raise ValueError(f"{path}: k changes within the trace")
    layer, ids = raw[:, 0], raw[:, 2:2 + k]
    n_layer = int(layer.max()) + 1

    # one pass: a record whose layer is below the previous one opens a window
    starts = np.flatnonzero(np.r_[True, layer[1:] < layer[:-1]])
    ends = np.r_[starts[1:], len(layer)]
    out = []
    for s, e in zip(starts, ends):
        t = int(np.count_nonzero(layer[s:e] == 0))
        if t == 0 or (e - s) != t * n_layer:
            continue                                   # a partial window at the end of the file
        out.append(ids[s:e].reshape(n_layer, t, k).transpose(1, 0, 2))
    routed = np.concatenate(out) if out else np.zeros((0, n_layer, k), dtype="<i4")
    return {"n_layer": n_layer, "k": k, "positions": int(routed.shape[0]), "routed": routed.astype("<u2")}


def trace_payload(path: Path) -> dict:
    t = read_trace(path)
    return {
        "name": Path(path).stem, "n_layer": t["n_layer"], "k": t["k"], "positions": t["positions"],
        "n_expert": int(t["routed"].max()) + 1 if t["positions"] else 0,
        # positions x layers x k, uint16, base64: ~4.7 MB for 3,700 positions
        "routed_b64": base64.b64encode(t["routed"].tobytes()).decode(),
    }


def read_profile(path: Path) -> dict:
    blob = Path(path).read_bytes()
    if blob[:4] != b"STRP":
        raise ValueError(f"{path}: not a Strata profile")
    _, n_layer, n_expert, slots, n = struct.unpack_from("<5I", blob, 4)
    pairs = np.frombuffer(blob, dtype="<u2", count=2 * n, offset=24).reshape(n, 2)
    rank = np.full(n_layer * n_expert, n, dtype=np.int64)        # unranked pairs come last
    rank[pairs[:, 0].astype(np.int64) * n_expert + pairs[:, 1]] = np.arange(n)
    return {"name": Path(path).stem, "n_layer": n_layer, "n_expert": n_expert, "slots": slots, "ranked": n,
            "rank": rank.tolist()}
