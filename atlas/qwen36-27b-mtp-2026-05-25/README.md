# Qwen3.6-27B-FP8 — Atlas ± MTP vs vLLM (rerun, 2026-05-25)

Re-run against latest `avarok/atlas-gb10:latest` (image built 2026-05-22 21:19 EDT,
**post-#63 dense MTP + sampling stack, pre-#71 `return_token_ids`** — see invariant
section below). Methodology identical to 2026-05-07 run except bench.py extended
to opt-into the #71 field for future-runs.

## TL;DR (delta vs 2026-05-07)

Major throughput improvements at c=1 across the board, driven primarily by PR #63
(dense MTP for Qwen3.6-27B and sampling-stack rewrite):

| | 2026-05-07 | 2026-05-25 | Δ |
|---|---|---|---|
| Atlas no-MTP c=1 | 4.38 tok/s | **12.46 tok/s** | +184% |
| Atlas MTP c=1    | 4.64 tok/s | **19.02 tok/s** | +310% |
| vLLM c=1         | 4.00 tok/s | 7.33 tok/s | +83% |
| MTP speedup over no-MTP (c=1) | +5.9% | **+52.6%** | dense MTP is now real work |
| Cold start, Atlas | ~240s | **100s** | -58% |
| Cold start, vLLM  | ~910s | 421s | -54% |

c=4 picture is unchanged from May 7: vLLM's continuous batching still wins
aggregate (28.88 tok/s) while Atlas plateaus around 13 tok/s aggregate at c=4.
MTP at c=4 is a wash (slight regression, -1.5%), suggesting tree-verify overhead
doesn't pay back once the batch is saturating compute.

Single-host, ASUS Ascent GX10 (NVIDIA GB10, SM121), firmware unchanged since
2026-05-07 (BIOS 0104.2026.0326.1657, EC 0x02000005, UEFI/SoC 0x03000006).

## Setup

- **Model:** `Qwen/Qwen3.6-27B-FP8` (27B dense, FP8 weights)
- **Atlas image:** `avarok/atlas-gb10:latest` (digest `sha256:6a0f8b6781e9e3a2f1d30bf8f674437ed75092688a573336f4da1cba91b7279c`)
- **vLLM image:** `vllm/vllm-openai:v0.20.0-aarch64-cu130-ubuntu2404` (digest `sha256:dd8034e5df3abdc472aa0d2fe8e4f3d1779f1e8a7c025e442b9000be41960e61`)
- **Atlas flags (per maintainer):** `--max-seq-len 65536 --kv-cache-dtype fp8 --kv-high-precision-layers auto --gpu-memory-utilization 0.90 --scheduling-policy slai --tool-call-parser qwen3_coder --enable-prefix-caching --disable-thinking`
- **MTP run additionally:** `--speculative` (built-in MTP heads, no separate drafter)
- **vLLM flags:** `--max-model-len 65536 --gpu-memory-utilization 0.90 --trust-remote-code`
- **Bench:** 3 prompts × 3 runs × concurrency {1, 4}, `temperature=0`, `return_token_ids: true`

## Throughput (median tok/s, 9 runs per concurrency)

| Engine             | c=1 median | c=1 max | c=4 per-req median | c=4 aggregate |
|---|---|---|---|---|
| **Atlas (no MTP)** | 12.46 | 13.03 | 3.26 | 13.04 |
| **Atlas (MTP)**    | 19.02 | 19.67 | 3.21 | 12.84 |
| **vLLM**           | 7.33    | 7.39    | 7.22    | 28.88 |

## PR #71 invariant verification — `Σ token_ids == usage.completion_tokens`

| Engine             | result |
|---|---|
| Atlas (no MTP)     | ok=0/18 violation=0 no_token_ids=18 |
| Atlas (MTP)        | ok=0/18 violation=0 no_token_ids=18 |
| vLLM (control)     | ok=18/18 violation=0 no_token_ids=0 |

**Result:** can't verify the #71 invariant on Atlas in this run — the `:latest`
DockerHub tag (image created 2026-05-22T21:19 EDT = 2026-05-23T01:19Z) **predates
the #71 merge** (2026-05-24T17:58Z). A direct probe with `return_token_ids: true`
against this image returns a response with **no `choices[0].token_ids` field**,
which is consistent with #71 not yet being in the build.

For comparison, vLLM 0.20.0-aarch64-cu130-ubuntu2404 honored the flag on 18/18
calls (`len(choices[0].token_ids) == usage.completion_tokens` held for every
response). So the bench wiring is correct end-to-end; Atlas just needs the next
image build to ship #71 before the invariant becomes meaningful on this side.

If a fresh image goes out, the same bench can be re-run unchanged and the
invariant table will populate `ok=N/N violation=0`. Happy to do that when a
post-#71 tag is published.

`no_token_ids` = server did not emit `choices[0].token_ids`.
`violation` = field was emitted but `len(token_ids) != usage.completion_tokens`.

## Cold start (post-cache, container start → /v1/models 200)

| Engine             | seconds |
|---|---|
| Atlas (no MTP)     | 100 |
| Atlas (MTP)        | 100 |
| vLLM               | 421 |

## MTP speedup vs no-MTP (Atlas)

```
c=1: +52.6% (12.46 -> 19.02 tok/s)
c=4 aggregate: +-1.5% (13.04 -> 12.84 tok/s)
```

## Atlas vs vLLM (single-stream)

```
no-MTP: Atlas +70.0% over vLLM (12.46 vs 7.33 tok/s)
MTP:    Atlas +159.5% over vLLM (19.02 vs 7.33 tok/s)
```

## Caveats / observations

- **Output-length divergence (FYI, not a bug claim):** Atlas tends to stop
  generating noticeably earlier than vLLM on the same prompts at `temperature=0`.
  For the "short" prompt, Atlas no-MTP median 21 tokens, Atlas MTP median 10
  tokens, vLLM 96 tokens (capped at `max_tokens=96`). A direct probe of Atlas
  produced visible chat-template artifacts in the assistant content
  (`"user\nassistant\nHi\n..."`), so the `--disable-thinking` path may be
  emitting an EOS-equivalent token earlier than vLLM does. This **does not
  affect `tok/s` validity** (both engines measure throughput of actually
  generated tokens) but it's worth knowing if anyone compares total output sizes.
- **Cold start.** Atlas now ~100s on this model from cold cache (down from ~240s
  on May 7) — a meaningful UX improvement. vLLM cold start dropped from ~910s
  to 421s, likely tied to the v0.20.0 image rev vs the May 7 nightly.
- **MTP at c=4:** -1.5% vs no-MTP at c=4 aggregate. Consistent with the
  intuition that draft+verify overhead doesn't pay back once the batch is
  saturating compute. MTP's big win is at c=1.

## Files

```
results/atlas-no-mtp/bench.json
results/atlas-mtp/bench.json
results/vllm/bench.json
results/*/cold-start.seconds
logs/*.log              (full bench-script output per run)
atlas-image.id          (avarok/atlas-gb10:latest digest at run time)
vllm-image.id           (vllm/vllm-openai digest at run time)
```

## Reproduction

```bash
bash run-mtp.sh
```
