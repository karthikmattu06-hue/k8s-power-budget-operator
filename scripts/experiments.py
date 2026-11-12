#!/usr/bin/env python3
"""Run a batch of grid events back to back and collect one row per event (for the report).

Each event is a full ``eventlib.run_event`` (baseline -> PowerBudget -> hold -> release -> score),
saved under ``results/<batch>/<run>/``; the batch summary goes to ``results/<batch>/summary.csv``.
Interrupt and re-run with the same ``--batch``: finished runs are skipped.

Default matrix (11 events, ~16 min each, ~3 h; run it overnight):
* policies x modes at -100 W, equal-thirds mix: optimizer / ladder / uniform x cap / interval
* depth sweep, optimizer, cap: -50, -150, -200 W
* tier-mix sweep, optimizer, cap, -100 W: 70/15/15 and 15/15/70 (premium/standard/flex)

Usage:
    python3 scripts/experiments.py [--batch NAME] [--only RUN_ID ...] [--list]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

from _kit import cluster_ports, kubectl
from eventlib import run_event

THIRDS = {"premium": 1 / 3, "standard": 1 / 3, "flex": 1 / 3}


def matrix() -> list[dict[str, Any]]:
    base = {"marginWatts": 30, "rampSeconds": 300, "holdSeconds": 600}
    runs = []
    for policy in ("optimizer", "ladder", "uniform"):
        for mode in ("cap", "interval"):
            spec = {**base, "reduction": {"watts": 100}, "mode": mode, "policy": policy}
            if policy == "uniform":
                spec["allowRoot"] = True
            runs.append({"id": f"{policy}-{mode}-100", "spec": spec, "mix": THIRDS})
    for w in (50, 150, 200):
        runs.append({"id": f"optimizer-cap-{w}", "spec": {**base, "reduction": {"watts": w}, "mode": "cap",
                                                           "policy": "optimizer"}, "mix": THIRDS})  # fmt: skip
    for name, mix in (("prem70", {"premium": 0.7, "standard": 0.15, "flex": 0.15}),
                      ("flex70", {"premium": 0.15, "standard": 0.15, "flex": 0.7})):  # fmt: skip
        spec = {**base, "reduction": {"watts": 100}, "mode": "cap", "policy": "optimizer"}
        runs.append({"id": f"optimizer-cap-100-{name}", "spec": spec, "mix": mix})
    return runs


COLUMNS = [
    "run_id", "policy", "mode", "requested_w", "baseline_w", "target_w", "margin_w", "reached_s", "ramp_met",
    "cap_peak_w", "cap_above_s", "intervals_ok", "hold_mean_w", "achieved_w", "flex_shed_frac",
    "premium_p99_s", "standard_p99_s", "flex_p99_s", "failed_protected", "release_ok",
]  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=time.strftime("batch-%Y%m%d"))
    ap.add_argument("--only", nargs="*", help="run only these run ids")
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    ap.add_argument("--rps", type=float, default=8.0)
    args = ap.parse_args()
    runs = [r for r in matrix() if not args.only or r["id"] in args.only]
    if args.list:
        for r in runs:
            print(r["id"], json.dumps(r["spec"]), r["mix"])
        return 0
    if not kubectl("-n", "pbo", "get", "configmap", "pbo-power-model", "-o", "name", check=False).strip():
        print("No calibration (ConfigMap pbo-power-model). Run: make calibrate")
        return 1
    root = Path("results") / args.batch
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    with cluster_ports():
        for i, r in enumerate(runs, 1):
            done = root / r["id"] / "result.json"
            if done.exists():
                print(f"[{i}/{len(runs)}] {r['id']}: already done, skipping")
                rows.append(json.loads(done.read_text()))
                continue
            print(f"[{i}/{len(runs)}] {r['id']}")
            try:
                rows.append(run_event(r["id"], r["spec"], rps=args.rps, mix=r["mix"], out_root=str(root)))
            except Exception as e:  # keep the batch going; the failure is recorded
                print(f"   FAILED: {e}")
                kubectl("delete", "pb", "smoke-event", "--ignore-not-found", check=False)
                time.sleep(60)
    with (root / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"summary: {root / 'summary.csv'} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
