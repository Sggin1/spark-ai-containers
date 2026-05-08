#!/usr/bin/env bash
# Summarize the 3-way Qwen3.6-27B-FP8 ± MTP results into summary.md.

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

A_NO=$DIR/results/atlas-no-mtp/bench.json
A_MTP=$DIR/results/atlas-mtp/bench.json
V=$DIR/results/vllm/bench.json

cat > "$SUM" <<EOF
# Qwen3.6-27B-FP8 — Atlas ± MTP vs vLLM (3-way head-to-head)

Maintainer-requested test (Atlas Discord, 2026-05-07).
Single-host, ASUS Ascent GX10 (NVIDIA GB10, SM121), post-firmware-update
(BIOS 0104.2026.0326.1657, EC 0x02000005, UEFI/SoC 0x03000006).

## Setup

- **Model:** \`Qwen/Qwen3.6-27B-FP8\` (27B dense, FP8 weights)
- **Atlas image:** \`avarok/atlas-gb10:latest\`
- **vLLM image:** \`vllm/vllm-openai:cu130-nightly\`
- **Atlas flags (per maintainer):** \`--max-seq-len 65536 --kv-cache-dtype fp8 --kv-high-precision-layers auto --gpu-memory-utilization 0.90 --scheduling-policy slai --tool-call-parser qwen3_coder --enable-prefix-caching --disable-thinking\`
- **MTP run additionally:** \`--speculative\` (built-in MTP heads, no separate drafter)
- **vLLM flags:** \`--max-model-len 65536 --gpu-memory-utilization 0.90 --trust-remote-code\` (KV-dtype: vLLM default = bf16 / auto)
- **Bench:** 3 prompts × 3 runs × concurrency {1, 4}, \`temperature=0\`

## Throughput (median tok/s, 9 runs per concurrency)

| Engine             | c=1 median | c=1 max | c=4 per-req median | c=4 aggregate |
|---|---|---|---|---|
| **Atlas (no MTP)** | $(extract $A_NO 1) | $(extract_max $A_NO 1) | $(extract $A_NO 4) | $(awk -v x="$(extract $A_NO 4)" 'BEGIN{print x*4}') |
| **Atlas (MTP)**    | $(extract $A_MTP 1) | $(extract_max $A_MTP 1) | $(extract $A_MTP 4) | $(awk -v x="$(extract $A_MTP 4)" 'BEGIN{print x*4}') |
| **vLLM**           | $(extract $V 1)    | $(extract_max $V 1)    | $(extract $V 4)    | $(awk -v x="$(extract $V 4)" 'BEGIN{print x*4}') |

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
\`\`\`

## Reproduction

\`\`\`bash
bash run-mtp.sh
\`\`\`
EOF

echo "Wrote $SUM"
cat "$SUM"
