# File: model_bytes.py
# Location: flightrec/model_bytes.py
# Purpose: Derive analytic bytes-moved + FLOPs for a forward pass from an HF config.json (kills the hand-math).
# Dependencies: stdlib (json, pathlib)

"""Analytic roofline inputs from an HF model config.

``measure`` needs bytes_moved + flops, and on GB10 they cannot be counted (no DRAM
counter), so they are MODELLED. Hand-modelling them is the tool's biggest UX wart
and its biggest accuracy risk: the DiffusionGemma probe was eyeballed at "~32% of
wall" when the real expert-activation made it ~64%. This module derives them from
the model's own config.json + the runtime shape (tokens/forward ``m``, forward
count, dtype).

Decode-time model (weights dominate DRAM traffic; attention-score FLOPs and KV
traffic are excluded — small for short contexts, and the point is the weight-
streaming roofline):

    flops/forward = 2 * active_params_per_token * m
    bytes/forward = bytes_per_param * (always_read_params + routed_experts_read)

The MoE crux is ``routed_experts_read``: at ``m`` tokens with top-k routing, the
UNIQUE experts touched per layer follow the coupon-collector expectation
``E*(1-(1-1/E)^(m*k))`` — ~k at m=1, saturating to E (all experts) at large m. At
m=256/k=8/E=128 that is already ~128, i.e. the whole expert bank streams every
forward. Pure functions; no torch, no GPU. An ESTIMATE: cross-check
``active_params_per_token`` against the model card once per model.
"""

from __future__ import annotations

import json
from pathlib import Path

# Effective bytes/param including quant scale overhead. nvfp4 = 4-bit weight +
# 8-bit scale per 16-elem block = 4.5 bits = 0.5625 B (group_size 16 confirmed).
BYTES_PER_PARAM = {
    "bf16": 2.0,
    "fp16": 2.0,
    "bfloat16": 2.0,
    "float16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "nvfp4": 0.5625,
    "fp4": 0.5625,
    "mxfp4": 0.5625,
    "int4": 0.5625,
    "w4a16": 0.5625,
}

# config.json key aliases across model families (gemma/qwen/deepseek/etc.).
_TOP_K = ("num_experts_per_tok", "top_k_experts", "moe_topk", "num_selected_experts")
_N_EXPERTS = ("num_experts", "num_local_experts", "n_routed_experts")


def _alias(t, c, names, default=0):
    """First present value among *names*, checking text_config then root."""
    for name in names:
        if t.get(name) is not None:
            return t[name]
        if c.get(name) is not None:
            return c[name]
    return default


def _dims(config):
    """Pull the dims we need, tolerating a nested ``text_config`` (gemma/VL)."""
    t = config.get("text_config", config)
    heads = t.get("num_attention_heads", 0)
    return {
        "layers": t.get("num_hidden_layers", 0),
        "hidden": t.get("hidden_size", 0),
        "experts": _alias(t, config, _N_EXPERTS),
        "top_k": _alias(t, config, _TOP_K),
        "moe_inter": t.get("moe_intermediate_size", 0),
        "dense_inter": t.get("intermediate_size", 0),
        "shared_inter": t.get("shared_expert_intermediate_size", 0),
        "vocab": t.get("vocab_size", 0),
        "heads": heads,
        "kv_heads": t.get("num_key_value_heads", heads),
        "head_dim": t.get("head_dim", 0),
    }


def _attn_params(d):
    """Q/K/V/O projection params for one layer (GQA-aware)."""
    hd = d["head_dim"] or (d["hidden"] // max(d["heads"], 1))
    q = d["hidden"] * d["heads"] * hd
    kv = 2 * d["hidden"] * d["kv_heads"] * hd
    o = d["heads"] * hd * d["hidden"]
    return q + kv + o


def _ffn_params(hidden, inter):
    """Gated FFN params (gate + up + down) for one expert/dense block."""
    return 3 * hidden * inter


def expected_experts(e, m, k):
    """Expected UNIQUE experts touched per layer at m tokens, top-k (coupon collector)."""
    if e <= 0 or k <= 0:
        return 0.0
    return e * (1.0 - (1.0 - 1.0 / e) ** (m * k))


def _experts_read(e, m, k, mode):
    """Unique experts read per layer under the chosen activation model."""
    if mode == "all":
        return float(e)
    if mode == "topk":
        return float(min(e, m * k))
    return expected_experts(e, m, k)


def _bytes_per_param(dtype):
    key = dtype.lower()
    if key not in BYTES_PER_PARAM:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(BYTES_PER_PARAM)}")
    return BYTES_PER_PARAM[key]


def _components(d, dense_ffn):
    """Per-layer param pieces: (attn, one_expert, dense/shared_ffn, router)."""
    attn = _attn_params(d)
    one_expert = _ffn_params(d["hidden"], d["moe_inter"]) if d["experts"] else 0
    if d["experts"]:
        dense = _ffn_params(
            d["hidden"], d["shared_inter"] or (d["dense_inter"] if dense_ffn else 0)
        )
    else:
        dense = _ffn_params(d["hidden"], d["dense_inter"])
    router = d["hidden"] * d["experts"]
    return attn, one_expert, dense, router


def derive(config, dtype, m, forwards, expert_mode="expected", dense_ffn=False):
    """Bytes-moved + FLOPs for ``forwards`` passes at ``m`` tokens/forward.

    Args:
        config: parsed HF config dict.
        dtype: weight dtype key (see ``BYTES_PER_PARAM``).
        m: tokens processed per forward (1 for AR decode; canvas for diffusion).
        forwards: number of forward passes in the measured window.
        expert_mode: ``expected`` (coupon collector), ``all``, or ``topk``.
        dense_ffn: also count a per-layer dense FFN alongside experts (hybrid models).

    Returns:
        Dict with bytes_moved, flops (the two roofline inputs), plus the
        per-forward breakdown and assumptions for the reader to sanity-check.
    """
    d = _dims(config)
    bpp = _bytes_per_param(dtype)
    attn, one_expert, dense, router = _components(d, dense_ffn)
    lm_head = d["hidden"] * d["vocab"]
    layers = d["layers"]
    always = layers * (attn + dense + router) + lm_head
    read = _experts_read(d["experts"], m, d["top_k"], expert_mode)
    bytes_fwd = bpp * (always + layers * read * one_expert)
    active = layers * (attn + dense + d["top_k"] * one_expert) + lm_head
    flops_fwd = 2 * active * m
    return {
        "bytes_moved": int(bytes_fwd * forwards),
        "flops": int(flops_fwd * forwards),
        "bytes_per_forward": int(bytes_fwd),
        "active_params_per_token": int(active),
        "experts_read_per_layer": round(read, 1),
        "total_experts": d["experts"],
        "bytes_per_param": bpp,
        "assumptions": (
            f"dtype={dtype} m={m} forwards={forwards} expert_mode={expert_mode} "
            f"dense_ffn={dense_ffn}; weights-only (excl attn-score FLOPs + KV traffic). "
            "Validate active_params_per_token against the model card."
        ),
    }


def from_config_file(path, dtype, m, forwards, expert_mode="expected", dense_ffn=False):
    """Load an HF config.json (file or model dir) and derive the roofline inputs."""
    p = Path(path)
    if p.is_dir():
        p = p / "config.json"
    config = json.loads(p.read_text(encoding="utf-8"))
    return derive(config, dtype, m, forwards, expert_mode, dense_ffn)
