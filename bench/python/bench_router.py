#!/usr/bin/env python3
"""Time the Python router with the same harness as cmd/router-bench.

    python3 bench/python/bench_router.py            # table
    python3 bench/python/bench_router.py --json     # for cmd/bench-runs
    python3 bench/python/bench_router.py --trace    # canonical trace for the cross-check

Standard library only; any CPython 3.8+ works. The interpreter version and
implementation are recorded in the output, because "Python" is not one number:
run it under the version a real deployment would use.
"""

import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router as rt  # noqa: E402

KEY_SPACE = 1 << 16


def time_passes(keys, replicas, iters, repeats):
    """Mirrors timePasses in cmd/router-bench: one warmup, median of repeats."""
    mask = KEY_SPACE - 1

    def one_pass():
        w = rt.Window(rt.new_router(replicas), rt.HOLD)
        step = w.step
        start = time.perf_counter_ns()
        for i in range(iters):
            step(keys[i & mask])
        elapsed = time.perf_counter_ns() - start
        w.drain()
        return elapsed

    one_pass()
    ds = sorted(one_pass() / iters for _ in range(repeats))
    return ds[len(ds) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--trace", action="store_true", help="print the canonical trace and exit")
    ap.add_argument("--trace-n", type=int, default=100_000)
    ap.add_argument("--iters", type=int, default=200_000)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    if args.trace:
        s = rt.trace(rt.SEED, args.trace_n, rt.REPLICAS, rt.HOLD)
        print(json.dumps(s, sort_keys=True, separators=(",", ":")))
        return

    keys = rt.keys(rt.SEED, KEY_SPACE)
    impl = "%s %s" % (platform.python_implementation(), platform.python_version())
    findings = []
    for replicas in (8, 64):
        ns = time_passes(keys, replicas, args.iters, args.repeats)
        findings.append({
            "section": "router",
            "name": "python pick+done, replicas=%d" % replicas,
            "value": ns,
            "unit": "ns/request",
            "note": "%s; median of %d passes x %d requests" % (impl, args.repeats, args.iters),
        })

    if args.json:
        rep = {
            "host": {
                "goos": sys.platform,
                "goarch": platform.machine(),
                "num_cpu": os.cpu_count() or 0,
                "go_version": "",
                "language": impl,
            },
            "findings": findings,
        }
        print(json.dumps(rep, indent=2))
        return

    print("measurement                         value  unit        note")
    for f in findings:
        print("%-34s %7.1f  %-10s  %s" % (f["name"], f["value"], f["unit"], f["note"]))


if __name__ == "__main__":
    main()
