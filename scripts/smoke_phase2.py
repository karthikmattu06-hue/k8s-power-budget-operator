#!/usr/bin/env python3
"""Phase 2 acceptance test: a cap-mode PowerBudget, planned by the optimiser, end to end.

Sequence (defaults): 8 rps baseline; apply a PowerBudget for 100 W below baseline as a hard
cap, 30 W margin, 300 s ramp, 600 s hold; sample every 5 s until Done; check recovery.
Run ``scripts/calibrate.py`` once first (the operator plans with its curve).

Checks
  1. Done, with the target reached (settled) before the ramp deadline.
  2. Cap: no 5-s reading of cluster watts during the hold is above the cap.
  3. Interval: every clock-aligned 5-min interval inside the hold averages at or below the cap.
     (Implied by 2; reported because it is how demand-response programmes settle.)
  4. Premium p99 TTFT over the event <= 2 s and standard <= 6 s: protected tiers are protected.
  5. Release order: flex is readmitted only after all replicas are Ready again; then replicas
     and watts are back to baseline.
  6. No premium or standard request failed. Shed flex is counted and reported, not failed.
Reported: plan chosen, model source, learned removal spike, achieved vs requested reduction,
share of flex shed. ``--policy ladder`` runs the fixed-order baseline for comparison.

Usage: python3 scripts/smoke_phase2.py [--policy optimizer|ladder] [--mode cap|interval] [--watts 100]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from _kit import cluster_ports, deployment, kubectl, results, scalar, series, set_rate, verdict

NAME = "smoke-event"
INSTANT_W = "sum(DCGM_FI_DEV_POWER_USAGE)"
SHORT_W = "avg_over_time(sum(DCGM_FI_DEV_POWER_USAGE)[20s:5s])"  # sum first: see adapters.CLUSTER_W
OUTCOMES = "sum by (tier, outcome) (pbo_loadgen_requests_total)"
P99 = 'histogram_quantile(0.99, sum by (le) (increase(pbo_loadgen_ttft_seconds_bucket{{tier="{tier}"}}[{w}s])))'


def pb_status() -> dict[str, Any]:
    out = kubectl("get", "pb", NAME, "-o", "json", check=False)
    return (json.loads(out).get("status") or {}) if out else {}


def outcomes() -> dict[tuple[str, str], float]:
    """Load generator request counts by (tier, outcome) so far."""
    return {(r["metric"]["tier"], r["metric"]["outcome"]): float(r["value"][1]) for r in results(OUTCOMES)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", type=float, default=8.0)
    ap.add_argument("--watts", type=float, default=100.0)
    ap.add_argument("--mode", default="cap", choices=["cap", "interval"])
    ap.add_argument("--policy", default="optimizer", choices=["optimizer", "ladder"])
    ap.add_argument("--margin", type=float, default=30.0)
    ap.add_argument("--ramp", type=int, default=300)
    ap.add_argument("--hold", type=int, default=600, help=">= 600 s guarantees one full aligned 5-min interval")
    ap.add_argument("--settle", type=int, default=120)
    args = ap.parse_args()
    ok = True

    cm = kubectl("-n", "pbo", "get", "configmap", "pbo-power-model", "-o", "name", check=False).strip()
    print(f"power model: {'calibrated' if cm else 'NOT calibrated (contract prior) -- run make calibrate first'}")

    with cluster_ports():
        kubectl("delete", "pb", NAME, "--ignore-not-found", "--wait=true")
        set_rate(args.rps)
        print(f"0. baseline {args.rps:g} rps for {args.settle} s; mock-llm (desired, ready) = {deployment('mock-llm')}")
        time.sleep(args.settle)
        before = outcomes()

        spec = {"reduction": {"watts": args.watts}, "mode": args.mode, "policy": args.policy,
                "marginWatts": args.margin, "rampSeconds": args.ramp, "holdSeconds": args.hold}  # fmt: skip
        manifest = {"apiVersion": "pbo.dev/v1alpha1", "kind": "PowerBudget", "metadata": {"name": NAME}, "spec": spec}
        kubectl("apply", "-f", "-", stdin=json.dumps(manifest))
        t0 = time.time()
        print(f"   applied {NAME}: {json.dumps(spec)}")
        print("   t(s)  phase      20s-avg W  instant W  target W  flex  replicas(d/r)  message")

        samples: list[dict[str, Any]] = []
        st: dict[str, Any] = {}
        printed = -15.0
        deadline = t0 + args.ramp + args.hold + 300
        while time.time() < deadline:
            st = pb_status()
            d, r = deployment("mock-llm")
            s = {"t": time.time(), "phase": st.get("phase", "?"), "flex": st.get("flexAdmit"), "desired": d, "ready": r}
            samples.append(s)
            flex = s["flex"] if s["flex"] is not None else "-"
            if s["t"] - t0 - printed >= 15 or st.get("phase") == "Done":
                printed = s["t"] - t0
                print(f"   {printed:5.0f}  {s['phase']:<9}  {scalar(SHORT_W):9.0f}  {scalar(INSTANT_W):9.0f}  "
                      f"{st.get('targetWatts') or float('nan'):8.0f}  {flex:>4}  "
                      f"{d}/{r:<12}  {str(st.get('message', ''))[:70]}")  # fmt: skip
            if st.get("phase") == "Done":
                break
            time.sleep(5)

        started, target = st.get("startedAt") or t0, float(st.get("targetWatts") or float("nan"))
        hold_until = float(st.get("holdUntil") or time.time())
        hold_start = hold_until - args.hold

        # 1. reached in time
        reached = st.get("reachedAt")
        t_reach = (reached - started) if reached else float("nan")
        c1 = st.get("phase") == "Done" and st.get("rampMet") is True
        print(f"1. Done; target reached {t_reach:.0f} s after start (deadline {args.ramp} s): {verdict(c1)}")
        ok &= c1

        # 2. cap at 5-s resolution over the hold
        hold = series(INSTANT_W, hold_start, hold_until)
        peak = max((v for _, v in hold), default=float("nan"))
        above = sum(5 for _, v in hold if v > target)
        c2 = bool(hold) and peak <= target
        print(f"2. cap {target:.0f} W: peak {peak:.0f} W over {len(hold)} readings, {above} s above: {verdict(c2)}")
        if args.mode == "cap":
            ok &= c2

        # 3. aligned 5-min intervals fully inside the hold
        first = (int(hold_start) // 300 + 1) * 300
        intervals = []
        for a in range(first, int(hold_until) - 299, 300):
            vals = [v for t, v in hold if a <= t < a + 300]
            if vals:
                intervals.append((a, sum(vals) / len(vals)))
        c3 = bool(intervals) and all(avg <= target for _, avg in intervals)
        desc = ", ".join(f"{time.strftime('%H:%M', time.localtime(a))} {avg:.0f} W" for a, avg in intervals)
        print(f"3. 5-min intervals: {desc or 'none fully inside the hold'}: {verdict(c3)}")
        ok &= c3

        # 4. protected tiers' latency over the whole event
        span = int(hold_until - started)
        p_prem = scalar(P99.format(tier="premium", w=span))
        p_std = scalar(P99.format(tier="standard", w=span))
        c4 = p_prem <= 2.0 and p_std <= 6.0
        print(
            f"4. p99 TTFT over the event: premium {p_prem:.2f} s (<= 2), standard {p_std:.2f} s (<= 6): {verdict(c4)}"
        )
        ok &= c4

        # 5. release order and recovery
        rel = [s for s in samples if s["phase"] == "Releasing"]
        early = [s for s in rel if (s["flex"] or 0) >= 1.0 and s["ready"] < s["desired"]]
        orig = int((st.get("originalReplicas") or {}).get("pbo/mock-llm", 5))
        t_rel = time.time()
        while time.time() - t_rel < 120 and deployment("mock-llm") != (orig, orig):
            time.sleep(5)
        time.sleep(40)
        w_after, base = scalar(SHORT_W), float(st.get("baselineWatts") or 0)
        c5 = not early and deployment("mock-llm") == (orig, orig) and w_after >= 0.95 * base
        print(f"5. release: flex readmitted after replicas Ready ({'yes' if not early else 'NO'}); "
              f"{orig}/{orig} back; {w_after:.0f} W vs baseline {base:.0f} W: {verdict(c5)}")  # fmt: skip
        ok &= c5

        # 6. no protected request failed
        after = outcomes()
        delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
        failed = sum(v for (tier, out), v in delta.items() if tier != "flex" and out != "ok")
        flex_shed = sum(v for (tier, out), v in delta.items() if tier == "flex" and out != "ok")
        flex_all = sum(v for (tier, _), v in delta.items() if tier == "flex")
        c6 = failed == 0
        print(f"6. failed premium/standard requests: {failed:.0f}: {verdict(c6)}")
        ok &= c6

        mean_hold = sum(v for _, v in hold) / len(hold) if hold else float("nan")
        print("   --- report ---")
        print(f"   policy {args.policy}, mode {args.mode}; plan: {json.dumps(st.get('plan', {}))}")
        print(f"   achieved reduction {base - mean_hold:.0f} W (requested {args.watts:.0f} W; cap {target:.0f} W, "
              f"hold mean {mean_hold:.0f} W)")  # fmt: skip
        print(f"   flex shed {flex_shed:.0f} of {flex_all:.0f} ({100 * flex_shed / max(flex_all, 1):.0f} %); "
              f"learned removal spike {st.get('spikeWatts')} W ({st.get('spikeSource')})")  # fmt: skip
        print(f"   reading noise {st.get('noiseWatts')} W (std dev) -> margin {st.get('effectiveMarginWatts')} W; "
              f"aim {st.get('planTargetWatts')} W")  # fmt: skip
        for a in st.get("actions") or []:
            where = (
                f"at {a['measuredWatts']} W  {a.get('node') or ''}"
                if "measuredWatts" in a
                else f"{a.get('spikeWatts')} W"
            )
            print(f"     t={a['at'] - started:5.0f} s  {a['move']}  {where}")
        kubectl("delete", "pb", NAME, "--wait=true")

    print("PHASE 2:", verdict(ok))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
