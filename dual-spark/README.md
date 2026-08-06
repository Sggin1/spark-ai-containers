# Dual DGX Spark Setup — Power & Connection Troubleshooting

A field-tested runbook for pairing two NVIDIA DGX Sparks (ASUS Ascent GX10 /
GB10 / SM121) over a single CX-7 direct-link cable, and for diagnosing the
two failure classes that consume the most forum hours:

1. **Power-induced throttling** — if the AC path sags under inference
   transients (we hit this on an undersized consumer UPS), the GPU can
   clamp into P8 *while under load*. Benchmarks plateau at ~1/3 of spec
   with no clear log line. Many UPS setups are fine — **measure first**
   with `gpu_stress.py` (§0).
2. **"Can't connect 2 boxes"** — Ray timeouts, NCCL falling back to TCP
   over the management interface, `Permission denied (publickey)` from
   launchers, half-bandwidth from a misconfigured second twin. All
   reproducible, all fixable in a known order.

If you're a forum reader who landed here from a Spark-pair thread, jump
straight to [`DUAL_SPARK_SETUP.md`](./DUAL_SPARK_SETUP.md). The TL;DR table
near the top maps symptoms to sections. For cable / twin / PCI pictures,
see [`TOPOLOGY.md`](./TOPOLOGY.md).

---

## Who this is for

- DGX Spark / ASUS Ascent GX10 owners running a 2-node TP=2 cluster
- vLLM users (the runbook targets eugr's
  [`spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) launcher,
  but the network and power sections apply to any stack — Atlas,
  llama.cpp RPC, raw NCCL benchmarks, sparkrun, etc.)
- Anyone whose 2-Spark setup is stuck somewhere between "boots fine" and
  "delivers spec"

If you're scaling past 2 nodes (mesh / 4-port topology), this guide is the
wrong shape — see the
[6x Spark forum thread](https://forums.developer.nvidia.com/t/6x-spark-setup/354399/56).

---

## What's in here

| File | Purpose |
|---|---|
| [`DUAL_SPARK_SETUP.md`](./DUAL_SPARK_SETUP.md) | Full runbook. Power, usernames (A / B / B→A), networking, SSH, UFW, NCCL, validation, troubleshooting. Paste-ready. |
| [`TOPOLOGY.md`](./TOPOLOGY.md) | Visual map: one cable ~200 Gb/s total, two twins / PCIe slots, PCI domains, bandwidth. Mermaid + ASCII. |
| [`gpu_stress.py`](./gpu_stress.py) | bf16 GEMM oracle. Run it before bench. Healthy ≈ 80–125 burst / 80–140 sustained TFLOP/s. Throttled (AC sag) ≈ ~8 / ~45. |
| `LICENSE` | MIT |
| `.gitignore` | Standard Python + transient artifacts |

---

## Prerequisites

- Two DGX Sparks (or compatible GB10 / SM121 boxes), already booted to DGX
  OS, on the same physical bench
- One ConnectX-7 direct-link cable between them (single-cable, dual-twin
  topology — the SKU that ships with two Sparks; **one cable carries the
  whole ~200 Gb/s fabric** — see [`TOPOLOGY.md`](./TOPOLOGY.md))
- Root / sudo on both
- AC that can hold **300 W+** transients per box. Wall is the known-good
  check if `gpu_stress.py` looks sick; a solid UPS is fine if numbers are
  healthy. See §0 of the runbook.

Soft requirements (the runbook assumes these but they're not strict):

- **Username model** (both supported):
  - **A** — same user + UID on both nodes (recommended, least headwind)
  - **B** — different user per box + `~/.ssh/config` map
  - **B → A** — conversion playbook if you started with two users  
  Details: runbook §1.1 / §2.3 / §2.4
- Docker + NVIDIA Container Toolkit, recent driver
- `nmcli`, `ufw`, `perftest` (`apt install perftest`)

---

## How to use this in 30 seconds

```bash
# 1. Verify power FIRST — before any networking work.
docker run --rm --gpus all --ipc=host \
  -v $PWD/gpu_stress.py:/work/gpu_stress.py \
  vllm-node-tf5 python3 -u /work/gpu_stress.py
# < 60 TFLOP/s burst or sustained? Fix AC path (try wall / better source). Re-run.

# 2. Then walk DUAL_SPARK_SETUP.md top-to-bottom on a fresh pair, or
#    jump to the TL;DR table and follow the section that matches your
#    symptom. Skim TOPOLOGY.md if twins / half-bandwidth are confusing.
```

If `gpu_stress.py` reports HEALTHY but the cluster still won't talk to
itself, your problem is in §2 (setup) or §4 (troubleshooting). If
`gpu_stress.py` reports UNHEALTHY or MARGINAL, **fix power first** — every
hour you spend tuning NCCL with a throttled GPU is wasted.

---

## What's verified

The runbook is the as-built record of a working 2-Spark TP=2 cluster.
Specifically:

- **Power**: throttled-vs-healthy TFLOP/s numbers (8.9/46.9 → 82/120) are
  measured on the same physical box, swapping only the AC source (one
  consumer UPS path vs wall). Other UPS/PDU gear may be fine — send
  datapoints.
- **Bandwidth**: ~107 Gb/s per twin via `ib_write_bw`, ~195–197 Gb/s
  aggregate (≈ full single-cable ~200 Gb/s budget, not 2× that).
- **NCCL transport**: confirmed RDMA (`NET/IB`) end-to-end with the patched
  `autodiscover.sh`, no Socket fallback in container logs.
- **User models**: both same-user (A) and two-user + SSH map (B); we used
  B then converted to A.
- **Models**: validated on the eugr stack with the recipes shipped in that
  repo. The example in §6 is a placeholder — substitute whatever you're
  actually running.
- **HW**: GB10 / SM121, DGX OS, single CX-7 cable, two-twin topology.

## What's NOT verified

- **Mesh / 4-port / 3+ node topologies.** This guide stops at 2. Past 2 you
  need a fundamentally different routing setup. See the
  [6x thread](https://forums.developer.nvidia.com/t/6x-spark-setup/354399/56).
- **Non-DGX Spark hardware.** The interface names, DGX-OS-specific defaults
  (UFW preset, `dgx-release`), and mlx5 chassis quirks are all Spark-specific.
- **Non-Docker stacks**: the launcher patch in §2.7 is for eugr's
  containerized launcher. The NCCL env vars themselves apply to any stack
  (bare-metal vLLM, Atlas, raw NCCL benchmarks), but you'll need to plumb
  them in yourself.
- **OTA-driver / DGX-OS combinations beyond what we tested.** The runbook
  flags the version-skew failure mode in §5 but can't enumerate every
  matrix.
- **Every power path.** Wall is verified-good on our box; one consumer UPS
  path was verified-bad under load. Data-center PDUs, line conditioners,
  and other UPS models are untested here — if `gpu_stress.py` is healthy,
  you're good.

If your symptom matches §0 (1/3-spec plateau) but you're already on a wall
outlet, please open an issue with `nvidia-smi -q -d POWER` from before and
during a bench — that's a data point worth collecting.

---

## Contributing

Issues and PRs welcome, especially:

- Additional symptom → section entries for the TL;DR table
- Confirmation (or refutation) of the runbook on Spark variants other than
  the one we tested on
- Power-source data points (UPS model X causes throttling, model Y is fine,
  PDU Z, etc.) — please include the `gpu_stress.py` before/after numbers
  and the AC source description
- Cleaner versions of the `autodiscover.sh` patch in §2.7 (ideally upstreamed
  to eugr's repo)

Please don't submit benchmark numbers from non-Spark hardware as
"verified" — keep this guide narrow.

---

## License

MIT. See [`LICENSE`](./LICENSE).

## Acknowledgements

- [eugr](https://github.com/eugr) for `spark-vllm-docker` and the
  [networking deep-wiki](https://deepwiki.com/eugr/spark-vllm-docker/7-dgx-spark-networking)
- The [NVIDIA dual-Spark playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/README.md)
- [sparkrun](https://sparkrun.dev/getting-started/networking/) and the
  participants in the [6x Spark forum thread](https://forums.developer.nvidia.com/t/6x-spark-setup/354399/56)
- The forum readers whose pain prompted this writeup
