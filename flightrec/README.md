# flightrec — GB10 benchmark flight recorder

Always-on, low-overhead **Tier-1** machine-state recorder for **NVIDIA DGX Spark (GB10 / sm_121a)**.

Wrap a bench, get a self-describing artifact: GPU power / clock / throttle / energy, CPU, and memory pressure on **one timeline**. Runs that throttled are marked **invalid**. Especially useful for **OOM / memory-pressure** triage (`mem_guard`) and for throwing out power-path / thermal wedges before they pollute results.

**Role:** data provider — clean, gated artifacts for analysis, not the analyst.

**Primary target:** DGX Spark / GB10 (validated). Core NVML + `/proc` paths may work on other Linux + NVIDIA hosts; validity, unified-memory semantics, and roofline defaults are Spark-tuned. Discrete-GPU / Intel host support is deferred.

Sibling: two-box fabric notes in [`../dual-spark/`](../dual-spark/).

---

## Install

Python **≥ 3.12**. From this directory:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
# optional plots:
# uv pip install -e ".[dev,plot]"
```

Or: `pip install -e ".[dev]"`.

CLI entrypoint: `flightrec` (also `python -m flightrec`).

### Calibrate once per box (recommended)

Needs a torch+CUDA interpreter for STREAM-triad wall measurement:

```bash
flightrec calibrate --python /path/to/torch-venv/bin/python
# or: export FLIGHTREC_TORCH_PYTHON=...
```

Persists under `~/.cache/flightrec/calibration-<hostname>.json`.  
Example measured walls on GB10 LPDDR5X: **~220–226 GB/s** (~80–83% of the 273 GB/s spec). Uncalibrated fallback is a modeled **195 GB/s** constant.

---

## Why it exists (OOM / throttle)

On unified-memory GB10, “GPU OOM” often looks like **host freezes**, swap thrash, or silent stalls — not a clean CUDA OOM. Flightrec gives you:

| Arm | When | What |
|---|---|---|
| `record` + validity | every bench | clock+power droop invalidates wedged runs |
| `mem_guard` | long jobs / daemons | kill/warn on low `MemAvailable` |
| `smoke` | before multi-hour work | short slice → bottleneck + ETA |
| `watch` | during long jobs | stall = marker silent **and** GPU idle |
| `live` | interactive | live throttle alarm (same wedge rule as gate) |

Example unit file: [`systemd/flightrec-memguard.service`](systemd/flightrec-memguard.service) (edit user/paths).  
**Note:** on kill threshold, `mem_guard` can **SIGKILL** matching processes and `docker stop` non-`registry` containers — intentional for GB10 freeze recovery; review flags before enabling as a daemon.

---

## Command reference

| command | what it does | key flags (defaults) |
|---|---|---|
| `record` | wrap a command → artifact | `--out` *(req)*, `--hz`(20), `--parse-tokens-re` |
| `report` | provenance + validity + rollup + phase tree | positional `run_dir` |
| `measure` | validity-gated utilization vector | `--bytes`, `--flops`(0), `--tokens` |
| `aggregate` | N-run median/IQR/CV% + bootstrap CI (drops INVALID) | `--metric`(kernel_s), … |
| `sample` | adaptive-N until CI half-width &lt; target | `--out-prefix` *(req)*, `--cmd …`, `--until-ci`(3.0) |
| `gate` | nonzero exit unless VALID + CI-tight (+ replicate) | `--min-n`(20), `--max-ci-pct`(5.0) |
| `compare` | 2-artifact hardware diff **or** cross-box REPLICATED | `--replicate-with`, `--metric` |
| `calibrate` | measure BW wall (STREAM-triad), persist per box | `--python`, `--reps`(50) |
| `calibration` | score predicted-vs-measured jsonl | `--pred-key`, `--meas-key` |
| `smoke` | pre-flight bottleneck scan | `--out`, `--marker-re`, `--units`, `--total` |
| `watch` | stall watchdog | `--marker-re`, `--stall-s`(600), `--on-stall` |
| `live` | live GPU/CPU monitor + throttle alarm | `--hz`(2.0), `--duration`, `--once` |

Full flags: `flightrec <cmd> --help`.

---

## Core workflow

### Single run

```bash
flightrec record --out results/run01 -- python bench.py --tokens 128
flightrec measure results/run01 --bytes 105e9 --tokens 128
flightrec report results/run01
```

If `valid=False`, the run throttled (or energy non-monotonic) — **do not average it in**.

### Adaptive-N + gate

```bash
flightrec sample \
    --out-prefix results/my-config \
    --until-ci 3.0 --min-n 5 --max-n 30 \
    --metric kernel_s --tokens 128 \
    --cmd python bench.py --tokens 128

flightrec gate results/my-config_* --metric kernel_s --min-n 5 --max-ci-pct 5.0
flightrec aggregate results/my-config_* --metric kernel_s --tokens 128
```

### Cross-box replication

```bash
flightrec compare results/boxA_run01 results/boxB_run01
flightrec compare results/boxA_* --replicate-with results/boxB_* --metric kernel_s
```

### Monitoring arms

```bash
flightrec smoke --out results/preflight \
    --marker-re 'layer' --units 3 --total 64 \
    -- python long_job.py

flightrec watch --marker-re 'step' --stall-s 600 --on-stall notify \
    -- python long_job.py

flightrec live                 # until Ctrl-C
python -m flightrec.mem_guard --kill-gb 20 --warn-used-gb 90
```

### In-process phases

```python
from flightrec.recorder import FlightRecorder

with FlightRecorder("results/run01") as rec:
    with rec.phase("decode"):
        run_benchmark()
```

---

## Artifact layout

Each `--out` directory is self-describing:

| file | contents |
|---|---|
| `header.json` | driver, GPU name, clock ceiling, BW wall source, recorder overhead |
| `samples.parquet` | per-tick GPU/CPU/mem/PSI timeline |
| `phases.parquet` | nested phase marks (if used) |

---

## Limitations (GB10 hardware)

- **No live DRAM BW counter** — achieved GB/s is analytic; score vs STREAM wall; watch `above_wall` (L2 residency).
- **NVML throttle bits can fire spuriously** — validity uses **clock + power droop**, not the bitmask alone.
- **NVML memory-info often N/A** on unified LPDDR5X — trust `/proc/meminfo` (`MemAvailable`).
- **Memory clock N/A** via NVML on this platform.
- **No CPU/board power** on stock interfaces.

Deeper measurement protocols (clean-room, Tier-2, etc.): [`PROTOCOLS.md`](PROTOCOLS.md).

---

## Layout (this folder)

```text
flightrec/           # Python package
tests/               # pytest (synthetic fixtures; no GPU required)
pyproject.toml
PROTOCOLS.md
systemd/             # example mem_guard unit
examples/baseline/   # optional sample aggregate JSON (illustrative)
LICENSE              # MIT
```

---

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/ -q
```

Suite uses synthetic fixtures — no GPU, no network. Optional plot tests skip without `.[plot]`.

---

## Support matrix

| Platform | Status |
|---|---|
| NVIDIA DGX Spark (GB10, aarch64, unified mem) | **Primary — validated** |
| Other Linux + NVIDIA (e.g. discrete RTX) | Experimental / unvalidated; some constants Spark-shaped |
| Intel GPU / non-NVIDIA | Not supported |

---

*Primary development target: NVIDIA DGX Spark (GB10). Intel / discrete-GPU profiles are deferred.*
