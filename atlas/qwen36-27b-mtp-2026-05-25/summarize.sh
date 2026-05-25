#!/usr/bin/env bash
# Summarize the 3-way Qwen3.6-27B-FP8 ± MTP rerun (2026-05-25) into summary.md.
# Includes PR #71 invariant verification report.

set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUM="$DIR/summary.md"

extract() {
  local f=$1 c=$2
  if [ ! -f "$f" ]; then echo "—"; return; fi
  jq -r ".concurrency_runs[] | select(.concurrency == $c) | .overall.tok_per_s_median" "$f" 2>/dev/null
}
extract_max() {
  local f=$1 c=$2
  if [ ! -f "$f" ]; then echo "—"; return; fi
  jq -r ".concurrency_runs[] | select(.concurrency == $c) | .overall.tok_per_s_max" "$f" 2>/dev/null
}
cs() {
  local cfg=$1
  cat "$DIR/results/$cfg/cold-start.seconds" 2>/dev/null || echo "—"
}
inv() {
  local f=$1
  if [ ! -f "$f" ]; then echo "—"; return; fi
  jq -r '[.concurrency_runs[].invariant_summary] | {ok: (map(.ok)|add), violation: (map(.violation)|add), no_token_ids: (map(.no_token_ids)|add), total: (map(.total)|add)} | "ok=\(.ok)/\(.total) violation=\(.violation) no_token_ids=\(.no_token_ids)"' "$f" 2>/dev/null
}

A_NO=$DIR/results/atlas-no-mtp/bench.json
A_MTP=$DIR/results/atlas-mtp/bench.json
V=$DIR/results/vllm/bench.json
ATLAS_IMG_ID=$(cat "$DIR/atlas-image.id" 2>/dev/null | awk '{print $1}' || echo "—")
VLLM_IMG_ID=$(cat "$DIR/vllm-image.id" 2>/dev/null | awk '{print $1}' || echo "—")

cat > "$SUM" <<EOF
# Qwen3.6-27B-FP8 — Atlas ± MTP vs vLLM (rerun, 2026-05-25)

Re-run against latest \`avarok/atlas-gb10:latest\` (image built 2026-05-23,
post-#63 dense MTP + sampling stack, post-#71 \`return_token_ids\`).
Methodology identical to 2026-05-07 run except bench.py extended to validate
the PR #71 invariant on every response.

Single-host, ASUS Ascent GX10 (NVIDIA GB10, SM121), firmware unchanged since
2026-05-07 (BIOS 0104.2026.0326.1657, EC 0x02000005, UEFI/SoC 0x03000006).

## Setup

- **Model:** \`Qwen/Qwen3.6-27B-FP8\` (27B dense, FP8 weights)
- **Atlas image:** \`avarok/atlas-gb10:latest\` (digest \`${ATLAS_IMG_ID}\`)
- **vLLM image:** \`vllm/vllm-openai:v0.20.0-aarch64-cu130-ubuntu2404\` (digest \`${VLLM_IMG_ID}\`)
- **Atlas flags (per maintainer):** \`--max-seq-len 65536 --kv-cache-dtype fp8 --kv-high-precision-layers auto --gpu-memory-utilization 0.90 --scheduling-policy slai --tool-call-parser qwen3_coder --enable-prefix-caching --disable-thinking\`
- **MTP run additionally:** \`--speculative\` (built-in MTP heads, no separate drafter)
- **vLLM flags:** \`--max-model-len 65536 --gpu-memory-utilization 0.90 --trust-remote-code\`
- **Bench:** 3 prompts × 3 runs × concurrency {1, 4}, \`temperature=0\`, \`return_token_ids: true\`

## Throughput (median tok/s, 9 runs per concurrency)

| Engine             | c=1 median | c=1 max | c=4 per-req median | c=4 aggregate |
|---|---|---|---|---|
| **Atlas (no MTP)** | $(extract $A_NO 1) | $(extract_max $A_NO 1) | $(extract $A_NO 4) | $(awk -v x="$(extract $A_NO 4)" 'BEGIN{print x*4}') |
| **Atlas (MTP)**    | $(extract $A_MTP 1) | $(extract_max $A_MTP 1) | $(extract $A_MTP 4) | $(awk -v x="$(extract $A_MTP 4)" 'BEGIN{print x*4}') |
| **vLLM**           | $(extract $V 1)    | $(extract_max $V 1)    | $(extract $V 4)    | $(awk -v x="$(extract $V 4)" 'BEGIN{print x*4}') |

## PR #71 invariant verification — \`Σ token_ids == usage.completion_tokens\`

| Engine             | result |
|---|---|
| Atlas (no MTP)     | $(inv $A_NO) |
| Atlas (MTP)        | $(inv $A_MTP) |
| vLLM (control)     | $(inv $V) |

\`no_token_ids\` = server did not emit \`choices[0].token_ids\` (expected for
pre-#71 Atlas builds and vLLM, which uses its own counting path).
\`violation\` = field was emitted but \`len(token_ids) != usage.completion_tokens\`
(would indicate the invariant is broken in this build).

## Cold start (post-cache, container start → /v1/models 200)

| Engine             | seconds |
|---|---|
| Atlas (no MTP)     | $(cs atlas-no-mtp) |
| Atlas (MTP)        | $(cs atlas-mtp) |
| vLLM               | $(cs vllm) |

## MTP speedup vs no-MTP (Atlas)

\`\`\`
c=1: $(awk -v a="$(extract $A_NO 1)" -v b="$(extract $A_MTP 1)" 'BEGIN{if(a>0) printf "+%.1f%% (%.2f -> %.2f tok/s)", (b-a)/a*100, a, b}')
c=4 aggregate: $(awk -v a="$(extract $A_NO 4)" -v b="$(extract $A_MTP 4)" 'BEGIN{if(a>0) printf "+%.1f%% (%.2f -> %.2f tok/s)", (b-a)/a*100, a*4, b*4}')
\`\`\`

## Atlas vs vLLM (single-stream)

\`\`\`
no-MTP: $(awk -v a="$(extract $A_NO 1)" -v v="$(extract $V 1)" 'BEGIN{if(v>0) printf "Atlas %+.1f%% over vLLM (%.2f vs %.2f tok/s)", (a-v)/v*100, a, v}')
MTP:    $(awk -v a="$(extract $A_MTP 1)" -v v="$(extract $V 1)" 'BEGIN{if(v>0) printf "Atlas %+.1f%% over vLLM (%.2f vs %.2f tok/s)", (a-v)/v*100, a, v}')
\`\`\`

## Files

\`\`\`
results/atlas-no-mtp/bench.json
results/atlas-mtp/bench.json
results/vllm/bench.json
results/*/cold-start.seconds
logs/*.log              (full bench-script output per run)
atlas-image.id          (avarok/atlas-gb10:latest digest at run time)
vllm-image.id           (vllm/vllm-openai digest at run time)
\`\`\`

## Reproduction

\`\`\`bash
bash run-mtp.sh
\`\`\`
EOF

echo "Wrote $SUM"
cat "$SUM"
