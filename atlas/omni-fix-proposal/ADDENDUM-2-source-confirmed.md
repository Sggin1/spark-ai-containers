# Addendum 2 — answers to the 4 source-read questions

Verified against atlas-src on gx10 + Omni's actual `config.json`.

---

## Q1 — `weight_prefix` approach is cleaner — with one wrinkle

**Yes, take this route.** `weight_prefix` is in `ModelConfig` (config.rs:251) and is the canonical pattern: qwen35, qwen35_dense, qwen3_vl, gemma4, and nemotron all read it. Plain nemotron_h must default to `weight_prefix = "backbone"`.

**The wrinkle is `lm_head`.** nemotron.rs:370-375:

```rust
fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
    if store.contains("lm_head.weight") {        // <- HARDCODED bare key, no prefix
        dense(store, "lm_head.weight")
    } else {
        self.load_embedding(store, config)        // tied-fallback
    }
}
```

For plain Nemotron-H: lm_head sits at top-level `lm_head.weight` → works.
For Omni: actual tensor is `language_model.lm_head.weight` → `store.contains("lm_head.weight")` returns **false**, silently falls back to embedding. **And Omni is untied** (Q3), so this is wrong — model would inference with the embedding as the head matrix.

**Three ways to fix it; pick one:**

- **(A) Two-prefix model**: add `lm_head_prefix: String` to `ModelConfig` (defaults `""`). Change line 371 to:
  ```rust
  let key = format!("{}lm_head.weight", config.lm_head_prefix);
  if store.contains(&key) { dense(store, &key) } else { self.load_embedding(store, config) }
  ```
  Dispatcher sets `weight_prefix = "language_model.backbone"` AND `lm_head_prefix = "language_model."` for Omni. **Recommended — minimal, explicit, no name-mangling.**

- **(B) Derive lm_head prefix from weight_prefix**: strip trailing `.backbone` to get the model root. Works for Nemotron-H but fragile across other architectures.

- **(C) Rename keys at WeightStore-build time**: strip `language_model.` from every key. Heavier, intersects with the issue HANDOFF flagged ("filter applied earlier, during weight-map construction"). Don't.

**(A) is what I'd ship.**

### Possibly relevant precedent

`ModelConfig` has a `nested_config: bool` field at config.rs:234 with the comment _"Whether config.json wraps the LLM config in a nested field (e.g., text_config). Determines weight prefix auto-detection behavior."_ Worth grepping `nested_config` across the codebase before writing your own — Qwen3-VL might already do almost exactly this, in which case follow that pattern.

```bash
grep -RnE 'nested_config|text_config' atlas-src/crates/ | head -30
```

---

## Q2 — Omni's config.json structure (confirmed)

Top-level keys (29 total, abridged):
```
architectures = ["NemotronH_Nano_Omni_Reasoning_V3"]
model_type = "NemotronH_Nano_Omni_Reasoning_V3"
auto_map, max_sequence_length, downsample_ratio, ...
sound_config, llm_config, vision_config       <- multimodal triple
quantization_config                              <- AT TOP LEVEL
```

`llm_config.model_type = "nemotron_h"` ✓ (matches existing factory.rs arm at line 67).

**Wrapper key is `llm_config`** — confirms HANDOFF, contradicts the Qwen-VL `text_config` pattern. Different convention; both wrappers exist in the wild.

### Important: `quantization_config` lives at TOP LEVEL, not in `llm_config`

Verified: `'quantization_config' in c == True`, `'quantization_config' in c['llm_config'] == False`.

So the dispatch-side `is_multimodal_wrapper` lift must **merge top-level `quantization_config` into the lifted llm_config**, not just substitute. Otherwise Atlas drops the NVFP4 quant info silently and falls back to BF16 (or errors). HANDOFF §"Risks" flagged this; it's now confirmed.

---

## Q3 — `lm_head` location & tying (confirmed)

```
$ grep -E 'lm_head' omni_safetensors_keys.txt
language_model.lm_head.weight              <- exists, 1 occurrence

$ python json: llm_config.tie_word_embeddings = False
```

Untied. The loader **needs to find** `language_model.lm_head.weight`. This is exactly the wrinkle in Q1.

---

## Q4 — Encoder tensors in WeightStore: ship v1 without filter

**Recommendation: accept the waste.** Add a `// TODO(omni-fix): filter encoder tensors at WeightStore build time` comment + mention it in the PR body as deferred work.

Reasoning:
- Loader does targeted lookups, not iteration → encoders sit unread; only memory waste, no correctness issue.
- ~1100 unused tensors at NVFP4 ≈ low single-digit GB, on 128 GB unified memory = noise.
- HANDOFF §"Risks" flagged that filtering at WeightStore-build time may need plumbing through `weight_map.rs` — non-trivial; risks expanding scope and stalling the PR.
- Maintainers can request it in review if they consider it gating; if not, follow-up PR.
- The primary value is "text-only inference works at all on Omni." Ship lean.

If the maintainer pushes back, the cleanest follow-up is a single `wanted_prefix: Option<&str>` arg to whatever builds the WeightStore index — small, isolated.

---

## tl;dr fix shape

**dispatch.rs**:
- Detect wrapper (`hidden_size` missing && `llm_config` present)
- Lift `llm_config` to be the active config, **merging top-level `quantization_config`** in
- Set `config.weight_prefix = "language_model.backbone"`
- Set `config.lm_head_prefix = "language_model."` (new field, see (A) above)

**config.rs**:
- Add `pub lm_head_prefix: String` with `#[serde(skip)]` and default `""`

**nemotron.rs**:
- Two-line tweak in `load_lm_head` (line 371-372) per (A)

**factory.rs**:
- No change. Wrapper's `model_type = "NemotronH_Nano_Omni_Reasoning_V3"` is fine because dispatch.rs lifts the inner `llm_config.model_type = "nemotron_h"` first, which the existing arm at factory.rs:67 already handles.

**Tests**:
- Unit test in dispatch.rs: synthetic wrapper config → asserts `weight_prefix == "language_model.backbone"`, `lm_head_prefix == "language_model."`, `quantization_config` survives the merge.

That's the whole PR.
