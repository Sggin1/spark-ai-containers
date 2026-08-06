# File: cli.py
# Location: flightrec/cli.py
# Purpose: CLI to record machine state around any command and to report an artifact.
# Dependencies: flightrec.recorder, flightrec.report

"""Command-line entry for the flight recorder.

    flightrec record --out results/run01 -- <command ...>
    flightrec report results/run01
    flightrec calibrate --python /path/to/torch-venv/bin/python

``record`` wraps an arbitrary command (e.g. a vLLM bench) and captures the whole
machine for its duration; ``report`` prints the provenance, validity verdict, and
rollup of a recorded artifact; ``calibrate`` measures this box's bandwidth wall.
"""

import argparse
import json
import subprocess
import sys
import time

from flightrec.recorder import FlightRecorder
from flightrec.report import summarize_run
from flightrec.compare import compare_runs, replication_over_runs
from flightrec.roofline import calibrate
from flightrec.measure import measure_artifact
from flightrec.aggregate import aggregate
from flightrec.gate import gate
from flightrec.calibration import calibration_report
from flightrec import benchy
from flightrec.sample import adaptive_sample
from flightrec.smoke import smoke, format_card
from flightrec.watch import watch, format_result
from flightrec.live import live
from flightrec import quiesce
from flightrec.model_bytes import from_config_file
from flightrec.compute_cal import calibrate_compute, compute_peak_for


def main(argv=None):
    """Parse args and dispatch to a subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = {
        "record": _do_record,
        "report": _do_report,
        "compare": _do_compare,
        "calibrate": _do_calibrate,
        "measure": _do_measure,
        "aggregate": _do_aggregate,
        "gate": _do_gate,
        "calibration": _do_calibration,
        "sample": _do_sample,
        "smoke": _do_smoke,
        "watch": _do_watch,
        "live": _do_live,
        "quiesce": _do_quiesce,
        "model-bytes": _do_model_bytes,
    }.get(args.cmd)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


def console():
    """setuptools console-script entry: exit with main()'s return code."""
    raise SystemExit(main())


def _build_parser():
    parser = argparse.ArgumentParser(prog="flightrec", description="GB10 Tier-1 flight recorder")
    sub = parser.add_subparsers(dest="cmd")
    rec = sub.add_parser("record", help="record state while a command runs")
    rec.add_argument("--out", required=True)
    rec.add_argument("--hz", type=int, default=20)
    rec.add_argument(
        "--quiesce-window",
        type=float,
        default=1.0,
        dest="quiesce_window",
        help="pre-flight resting-vitals window in seconds, stamped into the "
        "header before the workload runs (default 1.0)",
    )
    rec.add_argument(
        "--no-quiesce",
        action="store_true",
        dest="no_quiesce",
        help="skip the pre-flight quiescence sample (saves ~1s of startup latency)",
    )
    rec.add_argument(
        "--parse-tokens-re",
        default=None,
        dest="parse_tokens_re",
        help="regex to auto-parse tok/s from bench stdout (first capture group or full match)",
    )
    rec.add_argument("command", nargs=argparse.REMAINDER)
    rep = sub.add_parser("report", help="summarize a recorded artifact")
    rep.add_argument("run_dir")
    cmp_p = sub.add_parser(
        "compare",
        help="hardware diff of two artifacts, OR cross-box replication verdict with --replicate-with",
    )
    cmp_p.add_argument(
        "run_dirs",
        nargs="+",
        help="box-A artifact(s); exactly 2 dirs and no --replicate-with = single-run hardware diff",
    )
    cmp_p.add_argument(
        "--replicate-with",
        nargs="+",
        default=None,
        dest="replicate_with",
        help="box-B run dirs -> REPLICATED verdict on --metric across both boxes (INVALID dropped)",
    )
    cmp_p.add_argument("--metric", default="kernel_s")
    cmp_p.add_argument("--min-effect-pct", type=float, default=1.0, dest="min_effect_pct")
    cmp_p.add_argument("--bytes", type=float, default=None, dest="bytes_moved")
    cmp_p.add_argument("--flops", type=float, default=0)
    cmp_p.add_argument("--tokens", type=int, default=None)
    cal = sub.add_parser(
        "calibrate",
        help="measure this box's bandwidth wall (STREAM-triad), or the "
        "compute PEAK with --compute (matmul bench / recorded value)",
    )
    cal.add_argument(
        "--python",
        default=None,
        help="torch+CUDA python (default $FLIGHTREC_TORCH_PYTHON / this interpreter)",
    )
    cal.add_argument("--reps", type=int, default=50)
    cal.add_argument(
        "--compute",
        action="store_true",
        help="calibrate the COMPUTE peak (TFLOP/s) instead of the bandwidth wall",
    )
    cal.add_argument(
        "--dtype",
        default="bf16",
        help="dtype for --compute (bf16/fp16 measured; fp8/nvfp4/... need --peak-tflops)",
    )
    cal.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        dest="peak_tflops",
        help="record this compute peak instead of measuring (for quant dtypes)",
    )
    mea = sub.add_parser("measure", help="utilization vector from an artifact (validity-gated)")
    mea.add_argument("run_dir")
    mea.add_argument(
        "--bytes",
        type=float,
        default=None,
        dest="bytes_moved",
        help="analytic bytes moved -> enables %%-of-wall + AI",
    )
    mea.add_argument("--flops", type=float, default=0, help="FLOPs -> arithmetic intensity")
    mea.add_argument("--tokens", type=int, default=None, help="tokens -> tok/s + J/token")
    mea.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        dest="peak_tflops",
        help="compute ceiling (TFLOP/s) for the dtype -> pct_of_peak + 2-axis regime",
    )
    _add_deriver_args(mea, with_config=True)
    agg = sub.add_parser(
        "aggregate", help="N-run distribution (median/IQR/CV + bootstrap CI); drops INVALID"
    )
    agg.add_argument("run_dirs", nargs="+")
    agg.add_argument(
        "--metric",
        default="kernel_s",
        help="utilization-vector field to aggregate (kernel_s, tok_s, energy_j, achieved_gflops, ...)",
    )
    agg.add_argument("--bytes", type=float, default=None, dest="bytes_moved")
    agg.add_argument("--flops", type=float, default=0)
    agg.add_argument("--tokens", type=int, default=None)
    gat = sub.add_parser(
        "gate", help="submission gate: nonzero-exit unless VALID + CI-tight (+ replicated)"
    )
    gat.add_argument("run_dirs", nargs="+")
    gat.add_argument(
        "--replicate-with",
        nargs="+",
        default=None,
        help="run dirs from the other box -> require cross-box replication",
    )
    gat.add_argument("--metric", default="kernel_s")
    gat.add_argument("--min-n", type=int, default=20)
    gat.add_argument("--max-ci-pct", type=float, default=5.0)
    gat.add_argument("--min-effect-pct", type=float, default=1.0)
    gat.add_argument("--bytes", type=float, default=None, dest="bytes_moved")
    gat.add_argument("--flops", type=float, default=0)
    gat.add_argument("--tokens", type=int, default=None)
    cb = sub.add_parser("calibration", help="score predicted-vs-measured (jsonl) -> prune verdict")
    cb.add_argument("jsonl", help="file with one JSON object per line carrying the pred/meas keys")
    cb.add_argument("--pred-key", default="predicted")
    cb.add_argument("--meas-key", default="measured")
    smp = sub.add_parser("sample", help="adaptive-N: run until CI half-width < X%% of median")
    smp.add_argument(
        "--out-prefix",
        required=True,
        dest="out_prefix",
        help="directory prefix; each run stored as <prefix>_NNN",
    )
    smp.add_argument(
        "--cmd",
        nargs=argparse.REMAINDER,
        default=[],
        dest="bench_cmd",
        help="command to benchmark (everything after --cmd)",
    )
    smp.add_argument(
        "--until-ci",
        type=float,
        default=3.0,
        dest="until_ci",
        help="stop when CI half-width < this %% of median (default 3.0)",
    )
    smp.add_argument(
        "--min-n",
        type=int,
        default=5,
        dest="min_n",
        help="minimum runs before checking convergence (default 5)",
    )
    smp.add_argument(
        "--max-n", type=int, default=50, dest="max_n", help="hard cap on total runs (default 50)"
    )
    smp.add_argument(
        "--metric",
        default="kernel_s",
        help="utilization-vector field to converge on (default kernel_s)",
    )
    smp.add_argument(
        "--hz", type=int, default=20, help="FlightRecorder sampling frequency (default 20)"
    )
    smp.add_argument("--bytes", type=float, default=None, dest="bytes_moved")
    smp.add_argument("--flops", type=float, default=0)
    smp.add_argument("--tokens", type=int, default=None)
    smk = sub.add_parser(
        "smoke",
        help="pre-flight bottleneck scan: record a short slice of a heavy job, "
        "emit verdict + ETA BEFORE committing the full run",
    )
    smk.add_argument("--out", required=True, help="artifact dir for the recorded slice")
    smk.add_argument(
        "--marker-re",
        required=True,
        dest="marker_re",
        help="regex; each matching stdout line = one completed unit (layer/step/iter)",
    )
    smk.add_argument(
        "--units", type=int, default=3, help="marker units to observe before killing (default 3)"
    )
    smk.add_argument(
        "--total",
        type=int,
        default=None,
        dest="total_units",
        help="total units in the full run -> enables ETA",
    )
    smk.add_argument(
        "--settle",
        type=int,
        default=1,
        help="leading marker intervals dropped as warm-up (default 1)",
    )
    smk.add_argument("--hz", type=int, default=20, help="recorder sampling frequency (default 20)")
    smk.add_argument(
        "--grace",
        type=float,
        default=5.0,
        help="seconds after SIGTERM before SIGKILL (default 5.0)",
    )
    smk.add_argument("--bytes", type=float, default=None, dest="bytes_moved")
    smk.add_argument("--flops", type=float, default=0)
    smk.add_argument(
        "--tokens-per-unit",
        type=int,
        default=None,
        dest="tokens_per_unit",
        help="tokens produced per unit -> enables tok/s + J/token",
    )
    smk.add_argument("command", nargs=argparse.REMAINDER)
    _add_watch_parser(sub)
    _add_quiesce_parser(sub)
    _add_model_bytes_parser(sub)
    return parser


def _add_deriver_args(parser, with_config):
    """Shared model_bytes deriver flags (measure auto-fill + model-bytes subcommand)."""
    if with_config:
        parser.add_argument(
            "--from-hf-config",
            default=None,
            dest="from_hf_config",
            help="config.json (or model dir) -> auto-derive --bytes + --flops",
        )
    parser.add_argument(
        "--m",
        type=int,
        default=1,
        help="tokens/forward for the deriver (1=AR decode, canvas=diffusion)",
    )
    parser.add_argument(
        "--forwards", type=int, default=None, help="forward-pass count in the measured window"
    )
    parser.add_argument(
        "--dtype", default="bf16", help="weight dtype for the deriver (nvfp4/fp8/bf16/int4/...)"
    )
    parser.add_argument(
        "--expert-mode",
        default="expected",
        dest="expert_mode",
        choices=["expected", "all", "topk"],
        help="MoE expert-activation model (default expected = coupon-collector)",
    )
    parser.add_argument(
        "--dense-ffn",
        action="store_true",
        dest="dense_ffn",
        help="count a per-layer dense FFN alongside experts (hybrid models)",
    )


def _add_model_bytes_parser(sub):
    mbp = sub.add_parser(
        "model-bytes",
        help="derive analytic bytes-moved + FLOPs from an HF config.json "
        "(feed --bytes/--flops without the hand-math)",
    )
    mbp.add_argument("config", help="path to config.json or a model directory")
    _add_deriver_args(mbp, with_config=False)


def _add_quiesce_parser(sub):
    qui = sub.add_parser(
        "quiesce",
        help="pre-flight resting-vitals gate: sample foreign GPU/CPU load, "
        "nonzero-exit if CONTENDED (the box was not at rest to measure on)",
    )
    qui.add_argument(
        "--seconds", type=float, default=1.0, help="baseline sampling window (default 1.0)"
    )
    qui.add_argument(
        "--hz",
        type=float,
        default=5.0,
        help="baseline sampling rate (default 5; GB10 duty refreshes ~6 Hz)",
    )
    qui.add_argument(
        "--busy-floor",
        type=float,
        default=quiesce.GPU_BUSY_FLOOR_PCT,
        dest="busy_floor",
        help="GPU duty %% above which the box is not at rest",
    )
    qui.add_argument(
        "--power-floor",
        type=float,
        default=quiesce.GPU_POWER_FLOOR_W,
        dest="power_floor",
        help="GPU watts above which the box is not at rest",
    )
    qui.add_argument(
        "--load-floor",
        type=float,
        default=quiesce.LOAD_PER_CPU_FLOOR,
        dest="load_floor",
        help="1-min loadavg/core above which CPU is contended",
    )


def _add_watch_parser(sub):
    wat = sub.add_parser(
        "watch",
        help="long-run stall watchdog: alert when a hung job idles the "
        "GPU with no progress for N seconds (wrap a cmd or attach to a log)",
    )
    wat.add_argument(
        "--marker-re",
        required=True,
        dest="marker_re",
        help="regex; each matching stdout/log line = one progress unit",
    )
    wat.add_argument(
        "--logfile",
        default=None,
        help="attach mode: tail this log for markers (instead of wrapping a cmd)",
    )
    wat.add_argument(
        "--pid",
        type=int,
        default=None,
        help="attach mode: watch this pid for liveness / use as kill target",
    )
    wat.add_argument(
        "--stall-s",
        type=float,
        default=600.0,
        dest="stall_s",
        help="seconds of BOTH marker-silence and GPU-idle that mean STALLED (default 600)",
    )
    wat.add_argument(
        "--poll-s",
        type=float,
        default=5.0,
        dest="poll_s",
        help="GPU poll interval seconds (default 5)",
    )
    wat.add_argument(
        "--warmup-s",
        type=float,
        default=120.0,
        dest="warmup_s",
        help="startup grace before stall detection arms (default 120)",
    )
    wat.add_argument(
        "--idle-power-w",
        type=float,
        default=20.0,
        dest="idle_power_w",
        help="GPU power floor below which it counts as idle (default 20)",
    )
    wat.add_argument(
        "--idle-busy-pct",
        type=float,
        default=10.0,
        dest="idle_busy_pct",
        help="GPU duty-cycle floor below which it counts as idle (default 10)",
    )
    wat.add_argument(
        "--on-stall",
        choices=["notify", "kill"],
        default="notify",
        dest="on_stall",
        help="notify (default) or kill (SIGTERM->SIGKILL the job) on stall",
    )
    wat.add_argument(
        "--checkpoint-cmd",
        default=None,
        dest="checkpoint_cmd",
        help="shell hook run on stall (FLIGHTREC_STALL_MSG in env)",
    )
    wat.add_argument(
        "--notify-cmd",
        default=None,
        dest="notify_cmd",
        help="shell hook run on stall (FLIGHTREC_STALL_MSG in env)",
    )
    wat.add_argument(
        "--grace",
        type=float,
        default=10.0,
        help="seconds after SIGTERM before SIGKILL (default 10)",
    )
    wat.add_argument(
        "--record",
        default=None,
        dest="record_dir",
        help="write the poll timeline to <dir>/samples.parquet as stall proof",
    )
    wat.add_argument("--log", default="/tmp/flightrec_watch.log", dest="log_path")
    wat.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="wrap mode: the long job to run and watch (after --)",
    )
    liv = sub.add_parser(
        "live",
        help="live one-screen GPU/CPU monitor with a throttle alarm "
        "(standalone NVML loop; flashes on clock droop under load)",
    )
    liv.add_argument("--hz", type=float, default=2.0, help="refresh rate (default 2 Hz)")
    liv.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    liv.add_argument("--once", action="store_true", help="print one frame and exit")


def _do_record(args):
    cmd = _strip_dashes(args.command)
    tok_s = None
    window = 0 if args.no_quiesce else args.quiesce_window
    with FlightRecorder(args.out, hz=args.hz, quiesce_window=window) as rec:
        with rec.phase("command"):
            if cmd:
                if args.parse_tokens_re:
                    code, tok_s = benchy.run_capturing(cmd, args.parse_tokens_re)
                else:
                    code = subprocess.call(cmd)
            else:
                code = _idle()
    if tok_s is not None:
        _write_throughput(args.out, tok_s)
        print(f"[flightrec] parsed tok/s = {tok_s}")
    print(f"\n[flightrec] artifact -> {args.out}")
    summarize_run(args.out)
    return code


def _do_report(args):
    summarize_run(args.run_dir)
    return 0


def _do_compare(args):
    if args.replicate_with:
        result = replication_over_runs(
            args.run_dirs,
            args.replicate_with,
            metric=args.metric,
            min_effect_pct=args.min_effect_pct,
            bytes_moved=args.bytes_moved,
            flops=args.flops,
            tokens=args.tokens,
        )
        print(json.dumps(result, indent=2))
        print(f"\n[flightrec] {'REPLICATED' if result.get('replicated') else 'NOT REPLICATED'}")
        return 0
    if len(args.run_dirs) != 2:
        print(
            "[flightrec] hardware diff needs exactly 2 run dirs "
            "(or use --replicate-with for a cross-box distribution verdict)"
        )
        return 2
    compare_runs(args.run_dirs[0], args.run_dirs[1])
    return 0


def _do_calibrate(args):
    if args.compute:
        record = calibrate_compute(
            args.dtype, python=args.python, peak_tflops=args.peak_tflops, reps=args.reps
        )
        print(
            f"[flightrec] {record['source']} compute peak {record['dtype']}="
            f"{record['compute_peak_tflops']} TF -> {record['calibration_path']}"
        )
    else:
        record = calibrate(python=args.python, reps=args.reps)
        print(f"[flightrec] measured BW wall -> {record['calibration_path']}")
    print(json.dumps(record, indent=2))
    return 0


def _do_measure(args):
    bytes_moved, flops = _resolve_roofline_inputs(args)
    vector = measure_artifact(
        args.run_dir,
        bytes_moved=bytes_moved,
        flops=flops,
        tokens=args.tokens,
        peak_tflops=_resolve_peak(args),
    )
    print(json.dumps(vector, indent=2))
    return 0


def _resolve_peak(args):
    """Compute peak: explicit --peak-tflops, else the persisted per-box calibration for --dtype."""
    if args.peak_tflops is not None:
        return args.peak_tflops
    return compute_peak_for(args.dtype)


def _resolve_roofline_inputs(args):
    """(bytes, flops): derive from --from-hf-config when given, else pass CLI values through."""
    if not args.from_hf_config:
        return args.bytes_moved, args.flops
    derived = _derive(args.from_hf_config, args)
    # provenance to stderr so `measure` stdout stays pure JSON (pipeable)
    print(
        f"[flightrec] derived: bytes={derived['bytes_moved'] / 1e9:.0f}GB "
        f"flops={derived['flops'] / 1e12:.0f}TF "
        f"(active {derived['active_params_per_token'] / 1e9:.2f}B/tok, "
        f"experts {derived['experts_read_per_layer']}/{derived['total_experts']})",
        file=sys.stderr,
    )
    return derived["bytes_moved"], derived["flops"]


def _derive(config, args):
    """Run the model_bytes deriver from parsed deriver args; SystemExit if --forwards absent."""
    if not args.forwards:
        raise SystemExit("[flightrec] the deriver needs --forwards (and --m for batched/diffusion)")
    return from_config_file(
        config, args.dtype, args.m, args.forwards, args.expert_mode, args.dense_ffn
    )


def _do_model_bytes(args):
    print(json.dumps(_derive(args.config, args), indent=2))
    return 0


def _do_aggregate(args):
    summary = aggregate(
        args.run_dirs,
        metric=args.metric,
        bytes_moved=args.bytes_moved,
        flops=args.flops,
        tokens=args.tokens,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _do_gate(args):
    result = gate(
        args.run_dirs,
        replicate_with=args.replicate_with,
        metric=args.metric,
        min_n=args.min_n,
        max_ci_halfwidth_pct=args.max_ci_pct,
        min_effect_pct=args.min_effect_pct,
        bytes_moved=args.bytes_moved,
        flops=args.flops,
        tokens=args.tokens,
    )
    print(json.dumps(result, indent=2))
    print(f"\n[flightrec] GATE: {'PASS' if result['pass'] else 'FAIL'}")
    return 0 if result["pass"] else 1


def _do_calibration(args):
    pred, meas = _read_pairs(args.jsonl, args.pred_key, args.meas_key)
    if not pred:
        print(f"[flightrec] no rows with both '{args.pred_key}' and '{args.meas_key}'")
        return 1
    print(json.dumps(calibration_report(pred, meas), indent=2))
    return 0


def _do_sample(args):
    cmd = _strip_dashes(args.bench_cmd)
    result = adaptive_sample(
        cmd=cmd,
        out_prefix=args.out_prefix,
        metric=args.metric,
        until_ci_pct=args.until_ci,
        min_n=args.min_n,
        max_n=args.max_n,
        bytes_moved=args.bytes_moved,
        flops=args.flops,
        tokens=args.tokens,
        hz=args.hz,
    )
    print(json.dumps(result, indent=2))
    return 0


def _do_smoke(args):
    cmd = _strip_dashes(args.command)
    if not cmd:
        print(
            "[flightrec] smoke needs a command: flightrec smoke --out ... --marker-re ... -- <cmd>"
        )
        return 2
    card = smoke(
        cmd,
        args.marker_re,
        args.out,
        units=args.units,
        total_units=args.total_units,
        settle=args.settle,
        hz=args.hz,
        grace=args.grace,
        bytes_moved=args.bytes_moved,
        flops=args.flops,
        tokens_per_unit=args.tokens_per_unit,
    )
    print(format_card(card))
    print(f"\n[flightrec] slice artifact -> {args.out}")
    return 0


def _do_watch(args):
    cmd = _strip_dashes(args.command)
    if bool(cmd) == bool(args.logfile):
        print("[flightrec] watch needs EITHER a command (-- <cmd>) OR --logfile, not both/neither")
        return 2
    result = watch(
        cmd=cmd or None,
        marker_re=args.marker_re,
        logfile=args.logfile,
        pid=args.pid,
        stall_s=args.stall_s,
        poll_s=args.poll_s,
        warmup_s=args.warmup_s,
        idle_power_w=args.idle_power_w,
        idle_busy_pct=args.idle_busy_pct,
        on_stall=args.on_stall,
        checkpoint_cmd=args.checkpoint_cmd,
        notify_cmd=args.notify_cmd,
        grace=args.grace,
        log_path=args.log_path,
        record_dir=args.record_dir,
    )
    print(json.dumps(result, indent=2))
    print(f"\n[flightrec] {format_result(result)}")
    return 1 if result["stalled"] else 0


def _do_live(args):
    live(hz=args.hz, duration=args.duration, once=args.once)
    return 0


def _do_quiesce(args):
    result = quiesce.quiescence(
        seconds=args.seconds,
        hz=args.hz,
        busy_floor=args.busy_floor,
        power_floor=args.power_floor,
        load_floor=args.load_floor,
    )
    print(json.dumps(result, indent=2))
    print(f"\n[flightrec] {result['verdict']}: {'; '.join(result['reasons'])}")
    return 0 if result["quiet"] else 1


def _write_throughput(run_dir, tok_s):
    """Persist parsed tok/s into the artifact's header.json under 'throughput'."""
    import os

    header_path = os.path.join(run_dir, "header.json")
    try:
        with open(header_path, "r", encoding="utf-8") as fh:
            header = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        header = {}
    header["throughput"] = {"tok_s": tok_s, "source": "stdout_parse"}
    with open(header_path, "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=2)


def _read_pairs(path, pred_key, meas_key):
    pred, meas = [], []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get(pred_key) is not None and row.get(meas_key) is not None:
                pred.append(row[pred_key])
                meas.append(row[meas_key])
    return pred, meas


def _strip_dashes(command):
    return command[1:] if command and command[0] == "--" else command


def _idle(seconds=5):
    time.sleep(seconds)
    return 0
