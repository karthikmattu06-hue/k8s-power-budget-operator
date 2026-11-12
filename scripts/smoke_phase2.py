#!/usr/bin/env python3
"""Phase 2 acceptance test: one PowerBudget event end to end, scored pass/fail.

Defaults: 8 rps baseline; 100 W below baseline as a hard cap, 30 W minimum margin, 300 s ramp,
600 s hold; optimiser. Run ``scripts/calibrate.py`` once first (the operator plans with its curve).
The event is saved under ``results/smoke-<time>/``; plot it with ``make report RUN=<dir>``.

Checks
  1. Done, with the plan in place and the target reached before the ramp deadline.
  2. Cap: no 5-s reading during the hold above the cap (enforced in cap mode, reported otherwise).
  3. Interval: every aligned 5-min interval inside the hold averages at or below the target.
  4. Premium p99 TTFT over the event <= 2 s and standard <= 6 s.
  5. Release order (flex readmitted after replicas are Ready), replicas and watts back.
  6. No premium or standard request failed.

Usage: python3 scripts/smoke_phase2.py [--policy optimizer|ladder|uniform] [--mode cap|interval] [--watts 100]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from _kit import cluster_ports, kubectl, verdict
from eventlib import run_event


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", type=float, default=8.0)
    ap.add_argument("--watts", type=float, default=100.0)
    ap.add_argument("--mode", default="cap", choices=["cap", "interval"])
    ap.add_argument("--policy", default="optimizer", choices=["optimizer", "ladder", "uniform"])
    ap.add_argument("--margin", type=float, default=30.0)
    ap.add_argument("--ramp", type=int, default=300)
    ap.add_argument("--hold", type=int, default=600, help=">= 600 s guarantees one full aligned 5-min interval")
    ap.add_argument("--settle", type=int, default=120)
    args = ap.parse_args()

    cm = kubectl("-n", "pbo", "get", "configmap", "pbo-power-model", "-o", "name", check=False).strip()
    print(f"power model: {'calibrated' if cm else 'NOT calibrated (contract prior) -- run make calibrate first'}")
    spec = {"reduction": {"watts": args.watts}, "mode": args.mode, "policy": args.policy,
            "marginWatts": args.margin, "rampSeconds": args.ramp, "holdSeconds": args.hold}  # fmt: skip
    if args.policy == "uniform":
        spec["allowRoot"] = True
    run_id = f"smoke-{time.strftime('%Y%m%d-%H%M%S')}-{args.policy}-{args.mode}"
    with cluster_ports():
        r = run_event(run_id, spec, rps=args.rps, settle_s=args.settle)

    checks = [
        (f"Done; target reached {r['reached_s']} s after start (deadline {args.ramp} s)", r["done"] and r["ramp_met"]),
        (f"cap {r['target_w']:.0f} W: peak {r['cap_peak_w']:.0f} W, {r['cap_above_s']} s above",
         r["cap_above_s"] == 0),
        (f"5-min intervals {r['intervals']}", r["intervals_ok"]),
        (f"p99 TTFT premium {r['premium_p99_s']:.2f} s (<= 2), standard {r['standard_p99_s']:.2f} s (<= 6)",
         r["premium_p99_s"] <= 2.0 and r["standard_p99_s"] <= 6.0),
        ("release order and recovery", r["release_ok"]),
        (f"failed premium/standard requests: {r['failed_protected']:.0f}", r["failed_protected"] == 0),
    ]  # fmt: skip
    ok = True
    for i, (text, passed) in enumerate(checks, 1):
        enforced = not (i == 2 and args.mode != "cap")
        print(f"{i}. {text}: {verdict(passed)}{'' if enforced else ' (reported only)'}")
        ok &= passed or not enforced
    print("   --- report ---")
    print(f"   policy {args.policy}, mode {args.mode}; plan: {json.dumps(r['final_plan'])}")
    print(f"   achieved {r['achieved_w']:.0f} W (requested {args.watts:.0f} W; hold mean {r['hold_mean_w']:.0f} W)")
    print(f"   flex shed {100 * r['flex_shed_frac']:.0f} %; spike {r['spike_w']} W ({r['spike_source']}); "
          f"noise {r['noise_w']} W -> margin {r['margin_w']} W, aim {r['aim_w']} W")  # fmt: skip
    with open(f"results/{run_id}/status.json") as f:
        status = json.load(f)
    for a in status.get("actions") or []:
        where = (
            f"at {a['measuredWatts']} W  {a.get('nodes') or ''}" if "measuredWatts" in a else f"{a.get('spikeWatts')} W"
        )
        print(f"     t={a['at'] - status['startedAt']:5.0f} s  {a['move']}  {where}")
    print(f"   saved: results/{run_id}/  (plot: make report RUN=results/{run_id})")
    print("PHASE 2:", verdict(ok))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
