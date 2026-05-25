#!/usr/bin/env python3
"""Atlas vs vLLM A/B benchmark — fixed prompt set, OpenAI-compat endpoint.

2026-05-25 fork: adds `return_token_ids: true` and validates PR #71's
invariant `len(choices[0].token_ids) == usage.completion_tokens`
on every response. Throughput is computed from usage.completion_tokens
(unchanged from 2026-05-07 methodology).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median

import urllib.request

PROMPTS = [
    ("short",  "Reply with a single short sentence about cats.",                                                    96),
    ("medium", "Explain in ~150 words how transformers compute attention.",                                        256),
    ("long",   "Write a detailed 500-word summary of the impact of LLMs on software engineering, with examples.",  768),
]


def call(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "return_token_ids": True,  # PR #71 — opt-in, server ignores if unsupported
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        payload = json.loads(r.read())
    elapsed = time.perf_counter() - t0
    usage = payload.get("usage", {})
    completion_tok = usage.get("completion_tokens", 0)
    prompt_tok = usage.get("prompt_tokens", 0)

    # PR #71 invariant check — Σ token_ids == usage.completion_tokens
    choices = payload.get("choices") or [{}]
    token_ids = choices[0].get("token_ids") or []
    token_ids_count = len(token_ids)
    if token_ids_count == 0:
        invariant = "no_token_ids"          # server didn't emit (pre-#71 build or vLLM)
    elif token_ids_count == completion_tok:
        invariant = "ok"
    else:
        invariant = "violation"

    return {
        "elapsed_s": round(elapsed, 3),
        "completion_tokens": completion_tok,
        "prompt_tokens": prompt_tok,
        "token_ids_count": token_ids_count,
        "invariant": invariant,
        "tok_per_s": round(completion_tok / elapsed, 2) if elapsed else 0.0,
        "ttft_ms": usage.get("time_to_first_token_ms"),
        "server_tps": usage.get("response_token/s"),  # Atlas-only
    }


def run(url: str, model: str, runs_per_prompt: int, concurrency: int) -> dict:
    out = {"concurrency": concurrency, "results": [], "by_prompt": {}}
    jobs = []
    for name, prompt, max_tok in PROMPTS:
        for i in range(runs_per_prompt):
            jobs.append((name, prompt, max_tok, i))

    if concurrency == 1:
        for name, prompt, max_tok, i in jobs:
            r = call(url, model, prompt, max_tok)
            r["prompt_name"] = name
            r["iteration"] = i
            out["results"].append(r)
            print(f"  [{name}#{i}] {r['completion_tokens']}tok in {r['elapsed_s']}s = {r['tok_per_s']} tok/s (ttft={r.get('ttft_ms')}ms inv={r['invariant']})", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(call, url, model, prompt, max_tok): (name, i)
                       for name, prompt, max_tok, i in jobs}
            for fut in as_completed(futures):
                name, i = futures[fut]
                try:
                    r = fut.result()
                    r["prompt_name"] = name
                    r["iteration"] = i
                    out["results"].append(r)
                    print(f"  [{name}#{i} c={concurrency}] {r['completion_tokens']}tok / {r['elapsed_s']}s = {r['tok_per_s']} tok/s inv={r['invariant']}", flush=True)
                except Exception as e:
                    print(f"  [{name}#{i}] ERROR: {e}", flush=True)

    # Aggregate per-prompt
    for name, _, _ in PROMPTS:
        recs = [r for r in out["results"] if r.get("prompt_name") == name]
        if not recs: continue
        out["by_prompt"][name] = {
            "n": len(recs),
            "completion_tokens_median": median(r["completion_tokens"] for r in recs),
            "elapsed_s_median": median(r["elapsed_s"] for r in recs),
            "tok_per_s_median": median(r["tok_per_s"] for r in recs),
            "tok_per_s_max": max(r["tok_per_s"] for r in recs),
            "ttft_ms_median": median(r["ttft_ms"] for r in recs if r.get("ttft_ms")) if any(r.get("ttft_ms") for r in recs) else None,
        }

    # Invariant rollup — PR #71 hardware verification
    invs = [r["invariant"] for r in out["results"]]
    out["invariant_summary"] = {
        "ok": invs.count("ok"),
        "violation": invs.count("violation"),
        "no_token_ids": invs.count("no_token_ids"),
        "total": len(invs),
        "violations_detail": [
            {"prompt": r["prompt_name"], "iter": r["iteration"],
             "completion_tokens": r["completion_tokens"],
             "token_ids_count": r["token_ids_count"]}
            for r in out["results"] if r["invariant"] == "violation"
        ],
    }

    out["overall"] = {
        "n": len(out["results"]),
        "tok_per_s_median": median(r["tok_per_s"] for r in out["results"]),
        "tok_per_s_max": max(r["tok_per_s"] for r in out["results"]),
        "wall_time_s": round(sum(r["elapsed_s"] for r in out["results"]), 2),
    }
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    full = {"url": args.url, "model": args.model, "runs_per_prompt": args.runs, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "concurrency_runs": []}
    for c in args.concurrency:
        print(f"\n=== concurrency={c} ===", flush=True)
        result = run(args.url, args.model, args.runs, c)
        full["concurrency_runs"].append(result)

    with open(args.out, "w") as f:
        json.dump(full, f, indent=2)
    print(f"\nresults -> {args.out}")
    for cr in full["concurrency_runs"]:
        c = cr["concurrency"]
        ov = cr["overall"]
        inv = cr["invariant_summary"]
        print(f"  c={c}: median {ov['tok_per_s_median']} tok/s, max {ov['tok_per_s_max']} tok/s, {ov['n']} runs in {ov['wall_time_s']}s wall")
        print(f"        invariant: ok={inv['ok']} violation={inv['violation']} no_token_ids={inv['no_token_ids']} (of {inv['total']})")
