#!/usr/bin/env python3
"""Measure the power curve the operator plans with, and store it in a ConfigMap.

The optimiser predicts "how many watts would the cluster draw with n replicas at this load?"
from one curve: watts of a GPU node against the requests/s its replica receives. This script
measures that curve on the running cluster, the same way you would characterise real hardware:

    for each load step: hold it, then record (requests/s per replica, mean node watts)

All 5 replicas run during calibration, and the load is spread evenly, so each point is one
replica's share. The result goes to ConfigMap ``pbo/pbo-power-model`` (key ``model.json``),
which the operator reads every tick. Without it the operator falls back to the contract prior
(a straight line from parked to busy), and says so in the event status.

Run it once per cluster, with no PowerBudget active (~7 min):
    python3 scripts/calibrate.py [--rates 2,4,6,8,10] [--hold 75]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from _kit import cluster_ports, deployment, kubectl, scalar, set_rate

ADMITTED = 'sum(rate(pbo_admission_requests_total{{decision="admitted"}}[{w}s]))'
NODE_W = "avg(avg_over_time(DCGM_FI_DEV_POWER_USAGE[{w}s]))"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="2,4,6,8,10", help="total req/s steps (keep the top below ~65 % of capacity)")
    ap.add_argument("--hold", type=int, default=75, help="seconds per step; the last 45 s are measured")
    ap.add_argument("--back-to", type=float, default=8.0, help="rate to leave running afterwards")
    args = ap.parse_args()
    if kubectl("get", "pb", "-o", "name", check=False).strip():
        print("A PowerBudget exists; delete it first (calibration needs all replicas and all flex).")
        return 1
    window = min(45, args.hold - 15)
    points = []
    with cluster_ports():
        desired, ready = deployment("mock-llm")
        if ready != desired:
            print(f"mock-llm not fully ready ({ready}/{desired}); try again shortly")
            return 1
        print(f"calibrating with {ready} replicas: {args.rates} req/s, {args.hold} s each")
        print("  total rps  per replica  node W")
        for rate in (float(r) for r in args.rates.split(",")):
            set_rate(rate)
            time.sleep(args.hold)
            lam = scalar(ADMITTED.format(w=window))
            watts = scalar(NODE_W.format(w=window))
            x = lam / ready
            points.append([round(x, 3), round(watts, 1)])
            print(f"  {lam:9.2f}  {x:11.2f}  {watts:6.1f}")
        set_rate(args.back_to)
    model = {"points": points, "replicas": ready, "measuredAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "pbo-power-model", "namespace": "pbo"},
        "data": {"model.json": json.dumps(model)},
    }
    kubectl("apply", "-f", "-", stdin=json.dumps(cm))
    print(f"wrote ConfigMap pbo/pbo-power-model with {len(points)} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
