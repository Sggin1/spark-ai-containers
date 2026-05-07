# Atlas Omni Config-Parser Fix — HANDOFF

Continuation document for the work-in-progress to enable Atlas text-only inference on multimodal Nemotron-H wrapper configs (e.g. `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`).

This doc is self-contained — anyone (you, future-you, a fresh Claude session) can pick up from here without re-reading the prior conversation.

---

## Goal

Land an upstream PR at `Avarok-Cybersecurity/atlas` that:

1. Lets Atlas's config parser read `Nemotron-3-Nano-Omni-*` HF model configs by handling **multimodal wrapper configs** (`{ llm_config: {...}, vision_config: {...}, sound_config: {...} }`) — falling back to `.llm_config.*` when top-level `hidden_size` etc. are absent.
2. Lets the weight loader **skip vision/sound encoder tensors** during loading, so text-only inference works without implementing the encoders.

Full multimodal (image + audio inputs) is out of scope — separate, larger effort. Text-only is the path of least resistance and unlocks immediate user value.

---

## Why this is worth doing

- Atlas's `CONTRIBUTING.md` **explicitly states "All PRs are expected to be AI-generated"** — this is exactly the kind of contribution they want.
- The fix is small and well-scoped (a parser fallback + a weight-loader filter).
- The underlying LLM in Omni is **`nemotron_h`** (verified by reading Omni's `config.json` → `.llm_config.model_type == "nemotron_h"`), which Atlas already supports text-side at ~88 tok/s on this hardware (confirmed today in `/home/gx10/projects/atlas/runs/full-bench-2026-05-07/`).
- The maintainer is responsive on Discord.
- Builds toward future multimodal support (Qwen3-VL, etc. — same wrapper pattern).

---

## CLA — decision required before submitting

**`/home/gx10/projects/atlas-src/CLA.md`** is the standard Apache-style "we can re-license" CLA:

- You **retain copyright ownership** of your contribution (clause 4)
- You grant Atlas a **perpetual, irrevocable license** including the right to **re-license under commercial/proprietary terms** for their Enterprise Edition (clause 2)
- Standard for dual-licensed projects (CockroachDB, MongoDB historical pattern)
- Signing is automated via CLA-assistant on first PR

If this is acceptable, proceed. If not, the alternative is filing a detailed GitHub issue (already drafted in the prior conversation) and letting maintainer fix it.

---

## State at handoff

### Already done

- ✅ Atlas source cloned to **`/home/gx10/projects/atlas-src`** (depth=1 of `main`)
- ✅ Reproducible failure on Omni: `Error: Failed to parse config.json` → `missing field hidden_size at line 30243 column 1`
- ✅ Root cause confirmed: Omni's `config.json` has nested `llm_config.hidden_size` (no top-level)
- ✅ Underlying LLM verified Atlas-compatible: `model_type=nemotron_h, hidden_size=2688, layers=52, vocab=131072` — identical to non-Omni Nemotron-3-Nano
- ✅ Atlas-side parser entry point identified: **`crates/atlas-core/src/config/dispatch.rs::parse_config()`**
- ✅ Atlas-side `ModelConfig` struct identified: **`crates/atlas-core/src/config.rs:26`** — `pub hidden_size: usize` is the exact field that fails (no `#[serde(default)]`)
- ✅ Weight loader for nemotron_h identified: **`crates/spark-model/src/weight_loader/nemotron.rs`** + dispatch in **`crates/spark-model/src/factory.rs:46-86::loader_for_config()`**

### Not done

- ❌ Rust toolchain not installed on this machine. Need `rustup` → Rust 1.93.1 (pinned in `rust-toolchain.toml`)
- ❌ Fix not written yet — still in source-reading phase
- ❌ No build attempted

---

## Atlas's pre-PR checks (from `AGENTS.md`)

These must all pass before PR is mergeable. **All can run without a CUDA build** thanks to `ATLAS_SKIP_BUILD=1`:

```bash
cd /home/gx10/projects/atlas-src

# 1. Format (auto-fixable)
cargo fmt --all -- --check

# 2. Lints — most rigorous gate; runs without CUDA
ATLAS_SKIP_BUILD=1 cargo clippy --workspace --tests --all-features -- -Dwarnings

# 3. SPDX license headers
bash scripts/check-license-headers.sh

# 4. Typo check
typos     # cargo install typos-cli
```

A real end-to-end CUDA build is needed only if we want to runtime-test before submitting. PR CI will do this on their side.

---

## Suggested fix shape (sketch — needs source verification)

### Part 1 — `crates/atlas-core/src/config/dispatch.rs::parse_config()`

Detect multimodal wrapper schemas and remap. Pseudocode:

```rust
pub fn parse_config(json: &str) -> Result<ModelConfig> {
    let raw: serde_json::Value = serde_json::from_str(json)?;

    // Multimodal wrapper detection: top-level lacks `hidden_size` AND has `llm_config`
    let is_multimodal_wrapper = raw.get("hidden_size").is_none()
        && raw.get("llm_config").is_some();

    if is_multimodal_wrapper {
        // Pull `llm_config` up to be the active config root.
        // Preserve top-level wrapper architecture name for telemetry / logging.
        let llm = raw.get("llm_config")
            .ok_or_else(|| anyhow!("multimodal wrapper has no llm_config"))?;
        // Optionally: preserve top-level `quantization_config` if absent in inner
        // (Omni's quant config is at top level in the model we tested)
        let mut llm_cfg: ModelConfig = serde_json::from_value(llm.clone())?;
        // Mark the model so the weight loader knows to filter encoder tensors:
        llm_cfg.is_multimodal_text_only = true;  // new field, see Part 1b
        return Ok(llm_cfg);
    }

    // Existing path
    serde_json::from_str(json).context("Failed to parse config.json")
}
```

**Part 1b — add a flag field on `ModelConfig`** so the weight loader knows to skip encoder tensors:

```rust
// In crates/atlas-core/src/config.rs
pub struct ModelConfig {
    // ... existing fields ...
    
    /// Set when this config was extracted from a multimodal wrapper
    /// (e.g. `Nemotron-H Omni`). Tells weight loaders to silently skip
    /// vision/sound encoder safetensors entries. Does NOT enable any
    /// multimodal input path — text-only inference only.
    #[serde(default, skip_deserializing)]
    pub is_multimodal_text_only: bool,
}
```

### Part 2 — `crates/spark-model/src/weight_loader/nemotron.rs`

Filter weight names during loading. Most multimodal models use prefixes:

```rust
// In NemotronHWeightLoader::load (or wherever weights are iterated):
fn is_encoder_tensor(name: &str) -> bool {
    name.starts_with("vision_model.") ||
    name.starts_with("audio_model.") ||
    name.starts_with("sound_model.") ||
    name.starts_with("vision.") ||
    name.starts_with("sound.") ||
    // Verify exact prefixes against Omni's safetensors index
    false
}

// During iteration:
if config.is_multimodal_text_only && is_encoder_tensor(weight_name) {
    continue;  // silently skip
}
```

**Critical: verify the actual prefix names** by inspecting Omni's safetensors index BEFORE writing the filter. Run:

```bash
python3 -c "
from safetensors import safe_open
import os, glob
for f in sorted(glob.glob('/mnt/ai/cache/hub/hub/models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/snapshots/*/*.safetensors')):
    with safe_open(f, framework='pt') as st:
        for k in st.keys():
            print(k)
" | sort -u | head -50
```

This tells us the actual prefix structure (`vision_model.*` vs `vision.*` vs `image_encoder.*` etc.).

### Part 3 — register `nemotron_h_omni` in factory.rs (optional)

If we want to handle the wrapper `model_type` directly:

```rust
// In factory.rs::loader_for_config, add an arm:
"nemotron_h_nano_omni_reasoning_v3" | "nemotron_h_omni" => {
    Ok(Box::new(NemotronHWeightLoader))
}
```

Or keep it simpler: since `parse_config` already remaps the wrapper to the inner `llm_config` (which has `model_type: "nemotron_h"`), the existing `nemotron_h` arm in `factory.rs` will be hit. **This is cleaner — fewer files touched.**

---

## Concrete next-step order

1. **(5 min)** Re-read this doc + decide on CLA acceptability
2. **(5 min)** Inspect Omni's safetensors index to confirm encoder-tensor prefixes (the `is_encoder_tensor` function depends on this)
3. **(5 min)** Install rustup + Rust 1.93.1 on this machine
4. **(15 min)** Read the actual `dispatch.rs::parse_config` and `nemotron.rs` weight loader code in detail. Verify the assumptions in the sketch above match reality.
5. **(30-60 min)** Implement the fix in a feature branch:
   ```bash
   cd /home/gx10/projects/atlas-src
   git checkout -b fix/multimodal-wrapper-config
   # edit:
   #   crates/atlas-core/src/config/dispatch.rs
   #   crates/atlas-core/src/config.rs
   #   crates/spark-model/src/weight_loader/nemotron.rs
   ```
6. **(10 min)** Run pre-PR checks (`cargo fmt`, `cargo clippy`, `check-license-headers.sh`, `typos`)
7. **(?)** **Decision point:** runtime-test locally, OR submit PR and let their CI test?
   - Local runtime test = full CUDA build (~30-90 min, possibly painful)
   - CI test = let them run it; faster turnaround for us
   - Recommend: **submit without local CUDA build**, note in PR that the parser path passed clippy + fmt and we'd appreciate CI verification before merge
8. **(10 min)** Fork `Avarok-Cybersecurity/atlas` (using `gh repo fork` with the `Sggin1` account), push branch, open PR
9. **(0 min)** CLA-assistant kicks in on first PR — sign via the auto-prompt

Total: ~90-150 min of focused work, modulo build-environment surprises.

---

## Risks / gotchas to watch for

- **`ModelConfig` may have many required fields besides `hidden_size`** that Omni's wrapper doesn't have at top level. If so, the parser detects-and-remaps approach (Part 1) is correct; trying to add `#[serde(default)]` to every field would silently mask other config issues. Keep the wrapper-detection clean.
- **Weight loader may iterate weights via a precomputed index** (`weight_map.rs` exists in spark-model). If so, filter encoder tensors at index-build time, not at load iteration time.
- **`quantization_config` location.** In Omni's config it's at the top level (we verified). The wrapper may need to merge top-level `quantization_config` + nested `llm_config.*` rather than just substituting `llm_config`.
- **Weight loader may panic on unexpected `vision_model.*` tensors** *before* it gets to the filter, depending on iteration order. May need filter applied earlier (during weight-map construction).
- **Tokenizer path:** Atlas auto-detects tool-call parser and reasoning parser from `model_type`. Wrapper config has `model_type: "NemotronH_Nano_Omni_Reasoning_V3"` which won't match the existing `nemotron_h` heuristic. If we substitute `llm_config` cleanly, this is handled (inner `model_type=nemotron_h`). If we add a separate factory arm for the wrapper, we need to map it through.
- **Rust 1.93.1** (pinned) is recent — may not be in default rustup channels yet. Use `rustup install 1.93.1 && rustup override set 1.93.1`.

---

## Files to touch (final list)

- `crates/atlas-core/src/config/dispatch.rs` — wrapper detection + remap
- `crates/atlas-core/src/config.rs` — add `is_multimodal_text_only` field with `#[serde(default, skip_deserializing)]`
- `crates/spark-model/src/weight_loader/nemotron.rs` — filter encoder tensors when `is_multimodal_text_only`
- (Maybe) `crates/spark-model/src/weight_loader/weight_map.rs` — if filtering needs to happen at index-build time
- **New test:** `crates/atlas-core/src/config/dispatch.rs` — unit test that `parse_config` correctly remaps a synthetic wrapper config to the inner `llm_config`

---

## Fallback paths if anything goes wrong

| Failure mode | Fallback |
|---|---|
| Local Rust build fails / takes too long | Submit PR with `cargo fmt + clippy` passing; note no local runtime test; let their CI verify |
| Fix has unforeseen complications | Convert WIP branch into a GitHub issue with the analysis + draft code; maintainer can pick it up |
| CLA terms not acceptable to user | File the issue with full analysis (already drafted in prior conversation); let maintainer fix |
| Maintainer pushes their own fix first | Discard branch, no harm done; the diagnosis was the high-leverage contribution either way |

---

## Cross-references

- Original Omni probe + analysis: `/home/gx10/projects/atlas/runs/full-bench-2026-05-07/` (the Atlas vs vLLM benchmark) — the Phase 5b ngram-speculative attempt is the closest analogue (also a "tried, didn't work, reported with diagnosis" outcome)
- Atlas source clone: `/home/gx10/projects/atlas-src/`
- Maintainer's AI-contributor guide: `/home/gx10/projects/atlas-src/AGENTS.md`
- CLA: `/home/gx10/projects/atlas-src/CLA.md`
- Pre-PR check commands: `AGENTS.md` § "Local checks before a PR"
- Discord context: maintainer asked for Qwen3.6-27B-dense ± MTP test (separate workstream — `TODO.md` priority section)

---

## When picking this up next

```bash
# Quick re-orient
less /home/gx10/projects/atlas/runs/accuracy-prep-2026-05-07/HANDOFF-omni-fix.md
ls /home/gx10/projects/atlas-src/

# Verify clone is still valid (git wasn't touched since clone)
cd /home/gx10/projects/atlas-src && git log --oneline -1

# Then resume at "Concrete next-step order" §1 above
```

If the source clone is gone (e.g. /tmp wiped between sessions, or accidental delete), re-clone with:

```bash
git clone --depth=1 https://github.com/Avarok-Cybersecurity/atlas /home/gx10/projects/atlas-src
```
