# Atlas Omni Multimodal-Wrapper Fix — Research + Patch

Diagnosis and a tested patch enabling [Atlas](https://github.com/Avarok-Cybersecurity/atlas) to load `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (and other Nemotron-H Omni multimodal-wrapper checkpoints) for **text-only inference**.

The underlying LLM is plain `nemotron_h` — same architecture Atlas already serves at ~88 tok/s on GB10 — but the upstream config wraps it as `llm_config` and prefixes every text-tensor with `language_model.`, which the existing parser and weight loader don't handle. Two small additions plus a refactor make this work without touching any kernel code.

> Atlas maintainers: take what you want from this. The patch is in `working-tree.diff`. If you'd prefer a formal PR with CLA, happy to file it — just let me know.

## What's here

| File | Purpose |
|---|---|
| `working-tree.diff` | The patch (5 files changed, +157 / -32). Apply with `git apply` from atlas repo root. |
| `HANDOFF-omni-fix.md` | Original diagnostic + suggested-fix doc. The starting playbook. |
| `ADDENDUM-2-source-confirmed.md` | Findings after reading the actual atlas-core source. Refines the playbook with: confirmed `lm_head` lookup pattern, top-level `quantization_config` location, four answered design questions. |
| `omni_safetensors_keys.txt` | All 25,205 tensor names from Omni's safetensors index, sorted. The empirical basis for the prefix-strip approach. |

## What changes (5 files)

- **`crates/atlas-core/src/config.rs`** — new `pub lm_head_prefix: String` on `ModelConfig` (default `""`, `#[serde(skip)]`). Distinct from `weight_prefix` because `lm_head` lives at the language-model root, while `weight_prefix` points at the backbone one level deeper.
- **`crates/atlas-core/src/config/dispatch.rs`** —
  - New arm matching top-level `"NemotronH_Nano_Omni_Reasoning_V3"`. Lifts `llm_config`, runs the existing nemotron_h normalizations, sets `weight_prefix = "language_model.backbone"`, `lm_head_prefix = "language_model."`, `nested_config = true`, then hands off to `finalize_config(&mut config, &raw)` — which picks up the wrapper-root `quantization_config` automatically via `parse_quantization_config(raw)`, no merge required.
  - Extracted nemotron-h normalization (field-name remap + `hybrid_override_pattern` → `layer_types` expansion) into a private `normalize_nemotron_h(&mut config)` helper, shared between the existing `nemotron_h` arm and the new Omni arm. Pure refactor; behavior on plain Nemotron-H configs is unchanged.
- **`crates/spark-model/src/weight_loader/nemotron.rs`** — `load_lm_head` now uses `format!("{}lm_head.weight", config.lm_head_prefix)`. Empty prefix preserves the original `lm_head.weight` lookup for plain checkpoints.
- **`crates/atlas-core/src/config/factory.rs`** — initializes `lm_head_prefix: String::new()` in the `qwen3_next_80b_nvfp4()` ctor (only struct-literal `ModelConfig` initializer in the tree).
- **`crates/atlas-core/src/config/tests.rs`** — new `test_parse_nemotron_h_omni_wrapper` covering: `model_type` passthrough to inner, all four nemotron-h field remappings, layer-type pattern expansion, both prefixes, `layer_prefix(3)` composing correctly, end-to-end survival of top-level `quantization_config` through the lift.

## Why this shape

- Matches the existing wrapper-handling convention at `dispatch.rs:28-79` (qwen3_vl_moe / qwen3_5_moe / qwen3_5 with `text_config`). Top-level `model_type` literal match, nested config lift, `nested_config = true`.
- Two prefixes (vs one) is necessary because the safetensors store contains `language_model.backbone.layers.0.mixer.A_log` etc. but `language_model.lm_head.weight` is a sibling of `backbone`, not under it. A single `weight_prefix` couldn't cover both without name-mangling.
- The `quantization_config` hoist that an earlier draft of this fix proposed turned out to be unnecessary: `finalize_config` already calls `parse_quantization_config` against the raw wrapper root, so top-level quant info is found correctly. The unit test asserts this end-to-end (`quant_method == "modelopt"`, `quant_algo == "NVFP4"`).

## Empirical basis (Omni safetensors layout)

Counted 25,205 tensors across the 3 shards of `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`:

| Prefix | Count | Action |
|---|---|---|
| `language_model.*` | 24,103 | keep, strip prefix via `weight_prefix` |
| `sound_encoder.*` | 708 | unused (text-only) |
| `vision_model.*` | 388 | unused (text-only) |
| `mlp1.*` | 3 | unused (vision projection) |
| `sound_projection.*` | 3 | unused (audio projection) |

Confirmed `llm_config.tie_word_embeddings = false` and `language_model.lm_head.weight` is present in the safetensors — i.e. lm_head is **untied**, so the loader genuinely needs to find it (a fallback to embedding would be incorrect).

Confirmed `llm_config.model_type == "nemotron_h"`, dispatching to the existing `NemotronHWeightLoader` — no new loader needed.

Full key list in `omni_safetensors_keys.txt` for verification.

## Out of scope (deferred)

- **Filtering vision/sound encoder tensors at WeightStore build time.** The loader does targeted lookups, not iteration, so the unused encoder tensors sit in the store unread — pure memory waste, not a correctness issue. ~1100 unused NVFP4 tensors ≈ low-GB on a 128 GB unified memory system. A `TODO(omni-fix)` comment marks the spot in the dispatcher. Straightforward follow-up: plumb a `wanted_prefix` filter through `weight_map.rs`.
- **Multimodal input paths.** Image and audio inputs require encoder weight loaders + tokenizer/template work — separate, larger effort.

## Verification on the contributor side

- `cargo fmt --all -- --check` — pass
- `ATLAS_SKIP_BUILD=1 cargo clippy --workspace --tests --all-features -- -Dwarnings` — pass
- `cargo test -p atlas-core` — 15/15 pass (incl. new `test_parse_nemotron_h_omni_wrapper`)
- `typos` — clean
- No new files; existing SPDX headers preserved on touched files

No local CUDA runtime test was attempted (build environment not set up for it). The pre-PR check gauntlet from `AGENTS.md` passes; runtime verification would happen in CI.

## Hardware / context

Diagnosed and patched on a NVIDIA DGX Spark (ASUS Ascent GX10): GB10 Blackwell SM 12.1, 128 GB unified, kernel 6.17, NVIDIA driver 580.142, CUDA 13.0, Rust 1.93.1 (per `rust-toolchain.toml`). Atlas main branch at commit `9182c15` (depth-1 clone of `main`).

## License + use

This research is published under [AGPL-3.0-only](https://github.com/Avarok-Cybersecurity/atlas/blob/main/LICENSE) to match Atlas's license. Maintainers are welcome to copy / adapt / re-implement / merge as fits their process. If a formal PR with CLA is preferable, ping me on Discord.

— Sggin1
