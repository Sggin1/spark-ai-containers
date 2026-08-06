# File: test_model_bytes.py
# Location: tests/test_model_bytes.py
# Purpose: model_bytes deriver — param math, coupon-collector expert count, dtype bytes.
# Dependencies: flightrec.model_bytes

"""Analytic bytes/FLOPs from an HF config: pure math, no torch/GPU. A tiny
synthetic MoE config pins the arithmetic; the coupon-collector and dtype tables
are checked at their limits."""

import json

import pytest

from flightrec import model_bytes as mb
from flightrec.model_bytes import derive, expected_experts, from_config_file

# Tiny MoE: 2 layers, hidden 4, 2 heads x head_dim 2, 4 experts top-2, ffn 8, vocab 10.
_TINY = {
    "num_hidden_layers": 2,
    "hidden_size": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 2,
    "num_experts": 4,
    "top_k_experts": 2,
    "moe_intermediate_size": 8,
    "intermediate_size": 8,
    "vocab_size": 10,
}


def test_dtype_table_and_unknown():
    assert mb.BYTES_PER_PARAM["nvfp4"] == 0.5625  # 4b + 8b scale / 16-block
    assert mb.BYTES_PER_PARAM["bf16"] == 2.0
    with pytest.raises(ValueError):
        derive(_TINY, "fp3", m=1, forwards=1)


def test_expected_experts_limits():
    assert expected_experts(128, 1, 1) == pytest.approx(1.0)  # one pick -> one expert
    assert expected_experts(128, 256, 8) == pytest.approx(128.0, abs=0.01)  # saturates to E
    assert expected_experts(0, 5, 2) == 0.0  # dense model: no experts


def test_derive_tiny_config_math():
    # per layer: attn 64, one_expert 96, dense 96, router 16; lm_head 40.
    r = derive(_TINY, "bf16", m=1, forwards=1, dense_ffn=True)
    assert r["active_params_per_token"] == 744  # 2*(64+96+2*96)+40
    assert r["flops"] == 1488  # 2 * 744 * 1
    assert r["experts_read_per_layer"] == 1.8  # expected unique at m=1,k=2 (1.75 -> 1.8)
    assert r["bytes_per_param"] == 2.0


def test_expert_mode_changes_bytes_not_flops():
    base = derive(_TINY, "bf16", m=4, forwards=1, expert_mode="expected")
    allm = derive(_TINY, "bf16", m=4, forwards=1, expert_mode="all")
    topk = derive(_TINY, "bf16", m=4, forwards=1, expert_mode="topk")
    assert allm["experts_read_per_layer"] == 4.0  # every expert
    assert topk["experts_read_per_layer"] == 4.0  # min(4, 4*2)=4
    assert allm["bytes_moved"] >= base["bytes_moved"]
    assert allm["flops"] == base["flops"]  # FLOPs are routing-independent


def test_dense_ffn_flag_lifts_active_params():
    moe_only = derive(_TINY, "bf16", m=1, forwards=1, dense_ffn=False)
    hybrid = derive(_TINY, "bf16", m=1, forwards=1, dense_ffn=True)
    assert hybrid["active_params_per_token"] > moe_only["active_params_per_token"]


def test_from_config_file_dir_and_file(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_TINY))
    by_dir = from_config_file(tmp_path, "nvfp4", m=1, forwards=1)
    by_file = from_config_file(tmp_path / "config.json", "nvfp4", m=1, forwards=1)
    assert by_dir["bytes_moved"] == by_file["bytes_moved"] > 0


def test_top_k_alias_resolves():
    # uses top_k_experts (gemma) not num_experts_per_tok — must still resolve k=2
    r = derive(_TINY, "bf16", m=1, forwards=1)
    assert r["total_experts"] == 4
