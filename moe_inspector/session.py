"""Model session: loads an MoE model, records routing, and ablates parts of it.

Three independent knobs, all implemented with forward hooks so the checkpoint
weights are never modified:

  reroute : push a router logit to -inf, so the token picks its next-best
            expert instead. The expert is unreachable.
  silence : let routing happen, but zero the expert's output. The token keeps
            burning a slot on a dead expert.
  sleep   : replace a whole decoder layer's output with its input, turning the
            layer into an identity function.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import torch

from .detect import MoELayer, find_moe_layers, get_module

MAX_TRACE_TOKENS = 512


def _dead_mask(blocked: set[int], n_experts: int, device) -> torch.Tensor:
    mask = torch.zeros(n_experts, dtype=torch.bool, device=device)
    mask[sorted(blocked)] = True
    return mask


def _router_slots(output, n_experts: int):
    """Work out which element of a router's output tuple is which.

    Returns (parts, indices_i, weights_i, logits_i) or None if there is no
    integer index tensor to work with.
    """
    parts = list(output)
    tensors = [
        (i, t) for i, t in enumerate(parts) if isinstance(t, torch.Tensor)
    ]
    idx_i = next((i for i, t in tensors if not t.is_floating_point()), None)
    if idx_i is None:
        return None
    idx_shape = parts[idx_i].shape

    weight_candidates = [
        i for i, t in tensors if t.is_floating_point() and t.shape == idx_shape
    ]
    if len(weight_candidates) > 1:
        # top-k == num_experts: disambiguate by which one sums to 1 per row.
        weight_candidates.sort(
            key=lambda i: abs(float(parts[i].reshape(-1, idx_shape[-1])[0].sum()) - 1.0)
        )
    w_i = weight_candidates[0] if weight_candidates else None

    logits_i = next(
        (i for i, t in tensors
         if t.is_floating_point() and t.shape[-1] == n_experts and i != w_i),
        None,
    )
    return parts, idx_i, w_i, logits_i


class Inspector:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.state = "idle"  # idle | loading | ready | error
        self.message = ""
        self.model_id: str | None = None
        self.device: str | None = None
        self.dtype: str | None = None
        self.model = None
        self.tokenizer = None
        self.layers: list[MoELayer] = []

        # Ablation state.
        self.disabled: dict[int, set[int]] = {}
        self.mode = "reroute"  # reroute | silence
        self.sleeping: set[int] = set()
        self.ablation_enabled = True

        self._handles: list[Any] = []
        self._capture = False
        self._buf: dict[int, list[tuple[list[int], list[float]]]] = {}

    # ---------------------------------------------------------------- loading

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "message": self.message,
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype,
            "layers": [layer.to_json() for layer in self.layers],
            "mask": self.mask_json(),
        }

    def load(self, model_id: str, device: str = "auto", dtype: str = "auto") -> None:
        # Set the state before returning, so a status poll right after this call
        # can't still see "idle" and conclude nothing is happening.
        self.state = "loading"
        self.message = f"loading {model_id}"
        thread = threading.Thread(
            target=self._load, args=(model_id, device, dtype), daemon=True
        )
        thread.start()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load(self, model_id: str, device: str, dtype: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with self.lock:
            self._clear_hooks()
            self.state = "loading"
            self.message = f"loading {model_id}"
            self.model = self.tokenizer = None
            self.layers = []
            self.disabled = {}
            self.sleeping = set()

        try:
            device = self._resolve_device(device)
            if dtype == "auto":
                torch_dtype = torch.float32 if device == "cpu" else torch.float16
            else:
                torch_dtype = getattr(torch, dtype)

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch_dtype, low_cpu_mem_usage=True
            )
            model.to(device)
            model.eval()

            layers = find_moe_layers(model)
            if not layers:
                raise RuntimeError(
                    f"no MoE blocks found in {model_id} — is it a dense model?"
                )

            with self.lock:
                self.model = model
                self.tokenizer = tokenizer
                self.layers = layers
                self.model_id = model_id
                self.device = device
                self.dtype = str(torch_dtype).replace("torch.", "")
                self._install_hooks()
                self.state = "ready"
                total = sum(layer.num_experts for layer in layers)
                self.message = (
                    f"{len(layers)} MoE layers, {total} experts, "
                    f"top-{layers[0].top_k} routing"
                )
        except Exception as exc:  # surfaced in the UI
            with self.lock:
                self.state = "error"
                self.message = f"{type(exc).__name__}: {exc}"
                self.model = self.tokenizer = None

    # ------------------------------------------------------------------ hooks

    def _clear_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _install_hooks(self) -> None:
        for layer in self.layers:
            if layer.router_path:
                router = get_module(self.model, layer.router_path)
                self._handles.append(
                    router.register_forward_hook(self._make_router_hook(layer))
                )
            for expert_idx, path in enumerate(layer.expert_paths):
                expert = get_module(self.model, path)
                self._handles.append(
                    expert.register_forward_hook(
                        self._make_expert_hook(layer.index, expert_idx)
                    )
                )
            if layer.decoder_layer_path:
                decoder = get_module(self.model, layer.decoder_layer_path)
                self._handles.append(
                    decoder.register_forward_hook(
                        self._make_sleep_hook(layer.index), with_kwargs=True
                    )
                )

    def _blocked(self, layer_index: int) -> set[int]:
        if not self.ablation_enabled:
            return set()
        return self.disabled.get(layer_index, set())

    def _make_router_hook(self, layer: MoELayer):
        index = layer.index
        n_experts = layer.num_experts
        top_k = max(1, min(layer.top_k or 1, n_experts))

        def hook(module, args, output):
            blocked = self._blocked(index)

            # transformers>=5: routers return (logits, weights, indices) in some
            # order — identify the three by dtype and shape rather than position.
            if isinstance(output, (tuple, list)):
                slots = _router_slots(output, n_experts)
                if slots is None:
                    return output
                parts, idx_i, w_i, logits_i = slots
                indices = parts[idx_i]
                k = indices.shape[-1]

                if blocked:
                    dead = torch.zeros(n_experts, dtype=torch.bool, device=indices.device)
                    dead[sorted(blocked)] = True

                    if self.mode == "reroute" and logits_i is not None:
                        logits = parts[logits_i]
                        flat = logits.reshape(-1, n_experts).float()
                        flat = flat.masked_fill(dead, torch.finfo(torch.float32).min / 2)
                        probs = torch.softmax(flat, dim=-1)
                        values, new_idx = torch.topk(probs, k, dim=-1)
                        new_w = values / values.sum(-1, keepdim=True).clamp_min(1e-20)
                        parts[idx_i] = new_idx.reshape(indices.shape).to(indices.dtype)
                        parts[logits_i] = flat.reshape(logits.shape).to(logits.dtype)
                        if w_i is not None:
                            parts[w_i] = new_w.reshape(parts[w_i].shape).to(parts[w_i].dtype)
                        indices = parts[idx_i]
                    elif w_i is not None:
                        # silence: routing stands, but the expert contributes 0.
                        parts[w_i] = parts[w_i].masked_fill(dead[indices.long()], 0)

                if self._capture:
                    weights = parts[w_i] if w_i is not None else None
                    self._record_pairs(index, indices.reshape(-1, k), weights)
                return type(output)(parts)

            # Older/custom routers that just emit logits; downstream code does
            # the top-k itself, so -inf is enough to make an expert unreachable.
            if isinstance(output, torch.Tensor) and output.shape[-1] == n_experts:
                logits = output
                if blocked and self.mode == "reroute":
                    floor = torch.finfo(logits.dtype).min / 2
                    logits = logits.masked_fill(
                        _dead_mask(blocked, n_experts, logits.device), floor
                    )
                if self._capture:
                    self._record(index, logits.reshape(-1, n_experts), top_k)
                return logits

            return output

        return hook

    def _make_expert_hook(self, layer_index: int, expert_index: int):
        def hook(module, args, output):
            if self.mode != "silence" or expert_index not in self._blocked(layer_index):
                return output
            if isinstance(output, torch.Tensor):
                return torch.zeros_like(output)
            if isinstance(output, (tuple, list)):
                head = torch.zeros_like(output[0])
                return type(output)([head, *output[1:]])
            return output

        return hook

    def _make_sleep_hook(self, layer_index: int):
        def hook(module, args, kwargs, output):
            if not self.ablation_enabled or layer_index not in self.sleeping:
                return output
            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            if not isinstance(hidden, torch.Tensor):
                return output
            if isinstance(output, torch.Tensor):
                return hidden
            if isinstance(output, (tuple, list)):
                return type(output)([hidden, *output[1:]])
            return output

        return hook

    # ---------------------------------------------------------------- capture

    def _record(self, layer_index: int, logits: torch.Tensor, top_k: int) -> None:
        rows = self._buf.setdefault(layer_index, [])
        if len(rows) >= MAX_TRACE_TOKENS:
            return
        values, indices = torch.topk(logits.float(), top_k, dim=-1)
        # softmax over the top-k logits == full softmax renormalised over top-k
        weights = torch.softmax(values, dim=-1).detach().cpu().tolist()
        indices = indices.detach().cpu().tolist()
        for idx_row, weight_row in zip(indices, weights):
            if len(rows) >= MAX_TRACE_TOKENS:
                break
            rows.append((idx_row, [round(w, 4) for w in weight_row]))

    def _record_pairs(
        self, layer_index: int, indices: torch.Tensor, weights: torch.Tensor | None
    ) -> None:
        rows = self._buf.setdefault(layer_index, [])
        if len(rows) >= MAX_TRACE_TOKENS:
            return
        idx_list = indices.detach().cpu().tolist()
        k = indices.shape[-1]
        if weights is None:
            weight_list = [[1.0 / k] * k] * len(idx_list)
        else:
            weight_list = weights.detach().float().reshape(-1, k).cpu().tolist()
        for idx_row, weight_row in zip(idx_list, weight_list):
            if len(rows) >= MAX_TRACE_TOKENS:
                break
            rows.append((idx_row, [round(w, 4) for w in weight_row]))

    # ------------------------------------------------------------- generation

    def mask_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "enabled": self.ablation_enabled,
            "disabled": {str(k): sorted(v) for k, v in self.disabled.items() if v},
            "sleeping": sorted(self.sleeping),
        }

    def set_mask(
        self,
        disabled: dict[str, list[int]] | None = None,
        sleeping: list[int] | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if disabled is not None:
                self.disabled = {
                    int(k): set(v) for k, v in disabled.items() if v
                }
            if sleeping is not None:
                self.sleeping = set(sleeping)
            if mode in ("reroute", "silence"):
                self.mode = mode
        return self.mask_json()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        chat_template: bool = True,
        ablate: bool = True,
    ) -> dict[str, Any]:
        if self.state != "ready" or self.model is None:
            raise RuntimeError("no model loaded")

        with self.lock:
            tokenizer, model = self.tokenizer, self.model
            text = prompt
            if chat_template and getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            n_prompt = int(inputs["input_ids"].shape[1])

            self._buf = {}
            self._capture = True
            self.ablation_enabled = ablate
            started = time.perf_counter()
            try:
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        top_p=0.95 if temperature > 0 else None,
                        num_beams=1,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
            finally:
                self._capture = False
                self.ablation_enabled = True
            elapsed = time.perf_counter() - started

            sequence = output[0]
            generated = sequence[n_prompt:]
            all_ids = sequence.tolist()

            trace_len = min(
                MAX_TRACE_TOKENS,
                min((len(rows) for rows in self._buf.values()), default=0),
            )
            tokens = [
                {
                    "s": tokenizer.decode([tid]),
                    "gen": i >= n_prompt,
                }
                for i, tid in enumerate(all_ids[:trace_len])
            ]
            trace = [
                [row[0] for row in self._buf.get(layer.index, [])[:trace_len]]
                for layer in self.layers
            ]
            weights = [
                [row[1] for row in self._buf.get(layer.index, [])[:trace_len]]
                for layer in self.layers
            ]
            self._buf = {}

            return {
                "text": tokenizer.decode(generated, skip_special_tokens=True),
                "prompt_text": text,
                "n_prompt": n_prompt,
                "new_tokens": int(generated.shape[0]),
                "elapsed": round(elapsed, 2),
                "tok_per_s": round(int(generated.shape[0]) / max(elapsed, 1e-6), 1),
                "tokens": tokens,
                "trace": trace,
                "weights": weights,
                "ablated": ablate and bool(self.mask_json()["disabled"] or self.sleeping),
            }
