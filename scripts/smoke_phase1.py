#!/usr/bin/env python3
"""Phase 1 acceptance test on a running kind cluster ("done when", checked by machine).

Checks, in order:
  1. Every emulated GPU node publishes DCGM_FI_DEV_POWER_USAGE to Prometheus.
  2. Load ladder below saturation (default 4 -> 10 -> 13 rps, i.e. ~25 / 60 / 80 % of the
     ~16 rps the 5 replicas can serve): mean watts rise at every step, and p99 TTFT and
     TPOT are higher at the top step than at the bottom one.
  3. Every node carries traffic at the baseline rate (no replica left idle by load balancing).
  4. At the baseline rate (default 10 rps) the SLOs hold: premium TTFT p99 < 2 s, standard
     < 6 s, TPOT p99 < 50 ms. This is the operating point Phase 2 starts from.
  5. Removing the serving pod from one node (cordon + delete) drops that node to parked watts.

Capacity, for reference: a full batch of 16 runs at 30 ms per output token, so an average
request (~576 prompt + 160 output tokens) takes ~4.9 s; 16 / 4.9 = ~3.3 rps per replica,
~16 rps for 5. Above that the queue grows without bound and every SLO breaks.

Needs: kubectl pointed at the kind cluster, `make deploy` done. Stdlib only.
Usage: python3 scripts/smoke_phase1.py [--steps 4,10,13] [--baseline 10] [--hold 120]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager

PROM = "http://localhost:9090"
LOADGEN = "http://localhost:9101"
PARKED_W = 42.5  # 0.17 x 250 W, from the contract
SLO_TTFT = {"premium": 2.0, "standard": 6.0}  # seconds, p99 (contract tiers, assumed)
SLO_TPOT = 0.050  # seconds, p99, premium and standard


@contextmanager
def port_forward(ns: str, target: str, port: int) -> Iterator[None]:
    """Run `kubectl port-forward` in the background for the duration of the block."""
    p = subprocess.Popen(
        ["kubectl", "-n", ns, "port-forward", target, f"{port}:{port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)
    try:
        yield
    finally:
        p.terminate()


def query(promql: str, key: str = "node") -> dict[str, float]:
    """Instant query -> {value of label `key`: number}. Series without that label are keyed 'all'."""
    url = f"{PROM}/api/v1/query?" + urllib.parse.urlencode({"query": promql})
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.load(r)["data"]["result"]
    return {s["metric"].get(key, "all"): float(s["value"][1]) for s in data}


def set_rate(rps: float) -> None:
    req = urllib.request.Request(f"{LOADGEN}/rate", data=json.dumps({"rps": rps}).encode(), method="POST")
    urllib.request.urlopen(req, timeout=5).read()


def kubectl(*args: str) -> str:
    return subprocess.run(["kubectl", *args], check=True, capture_output=True, text=True).stdout


def fmt(values: dict[str, float], unit: str, scale: float = 1.0) -> str:
    return ", ".join(f"{k.replace('pbo-', '')}={v * scale:.2f}{unit}" for k, v in sorted(values.items()))


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# All windows are 1 minute; each step holds >= 2 minutes so the window sees only that step.
WATTS = "avg by (node) (avg_over_time(DCGM_FI_DEV_POWER_USAGE[1m]))"
SERVER_P99 = "histogram_quantile(0.99, sum by (le{by}) (rate(vllm:{m}_bucket[1m])))"
CLIENT_TTFT_P99 = "histogram_quantile(0.99, sum by (le, tier) (rate(pbo_loadgen_ttft_seconds_bucket[1m])))"


def measure(rps: float) -> dict[str, dict[str, float]]:
    """Snapshot of watts, server-side p99s and client-side per-tier p99 TTFT."""
    m = {
        "watts": query(WATTS),
        "ttft_node": query(SERVER_P99.format(by=", node", m="time_to_first_token_seconds")),
        "ttft": query(SERVER_P99.format(by="", m="time_to_first_token_seconds")),
        "tpot": query(SERVER_P99.format(by="", m="time_per_output_token_seconds")),
        "tier_ttft": query(CLIENT_TTFT_P99, key="tier"),
    }
    print(f"[{rps:g} rps]")
    print(f"  watts:          {fmt(m['watts'], ' W')}")
    print(f"  p99 TTFT/node:  {fmt(m['ttft_node'], ' s')}")
    print(f"  p99 TTFT/tier:  {fmt(m['tier_ttft'], ' s')}  (client side)")
    print(f"  cluster p99:    TTFT={m['ttft'].get('all', float('nan')):.2f} s, "
          f"TPOT={m['tpot'].get('all', float('nan')) * 1000:.1f} ms")  # fmt: skip
    return m


def mean(d: dict[str, float]) -> float:
    return sum(d.values()) / max(len(d), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="4,10,13", help="comma-separated rps ladder, ascending")
    ap.add_argument("--baseline", type=float, default=10.0, help="rate at which SLOs must hold (one of --steps)")
    ap.add_argument("--hold", type=int, default=120, help="seconds at each rate (>= 2 x the 1 m window)")
    args = ap.parse_args()
    steps = [float(x) for x in args.steps.split(",")]
    if args.baseline not in steps:
        ap.error("--baseline must be one of --steps")
    ok = True

    with port_forward("monitoring", "svc/kps-prometheus", 9090), port_forward("pbo", "deploy/loadgen", 9101):
        # 1. every node reports watts
        nodes = [n.split("/")[1] for n in kubectl("get", "nodes", "-l", "pbo.dev/gpu=emulated", "-o", "name").split()]
        deadline = time.time() + 120
        while len(query("DCGM_FI_DEV_POWER_USAGE")) < len(nodes) and time.time() < deadline:
            time.sleep(5)
        seen = query("DCGM_FI_DEV_POWER_USAGE")
        print(f"1. watts series: {len(seen)} of {len(nodes)} nodes: {verdict(len(seen) == len(nodes))}")
        ok &= len(seen) == len(nodes)

        # 2. load ladder
        print(f"2. load ladder: {' -> '.join(f'{s:g}' for s in steps)} rps, {args.hold} s each")
        results: dict[float, dict[str, dict[str, float]]] = {}
        for rps in steps:
            set_rate(rps)
            time.sleep(args.hold)
            results[rps] = measure(rps)
        w = [mean(results[s]["watts"]) for s in steps]
        lo, hi = results[steps[0]], results[steps[-1]]
        rising = all(w[i + 1] > w[i] * 1.05 for i in range(len(w) - 1))  # watts up >= 5 % per step
        ttft_up = hi["ttft"].get("all", 0) > lo["ttft"].get("all", 0)
        tpot_up = hi["tpot"].get("all", 0) > lo["tpot"].get("all", 0)
        ladder_ok = rising and ttft_up and tpot_up
        print(f"   mean watts {' -> '.join(f'{x:.0f}' for x in w)}; p99 TTFT and TPOT higher at top: "
              f"{verdict(ladder_ok)}")  # fmt: skip
        ok &= ladder_ok

        # 3. balance at baseline: every node well above parked
        base = results[args.baseline]
        idle = [n for n, v in base["watts"].items() if v < PARKED_W + 10]
        spread = max(base["watts"].values()) - min(base["watts"].values())
        bal_ok = not idle and len(base["watts"]) == len(nodes)
        print(f"3. balance at {args.baseline:g} rps: spread {spread:.0f} W, idle nodes {idle or 'none'}: "
              f"{verdict(bal_ok)}")  # fmt: skip
        ok &= bal_ok

        # 4. SLOs at baseline
        checks = [(f"{t} TTFT p99", base["tier_ttft"].get(t, float("inf")), lim) for t, lim in SLO_TTFT.items()]
        checks.append(("TPOT p99", base["tpot"].get("all", float("inf")), SLO_TPOT))
        slo_ok = all(v < lim for _, v, lim in checks)
        print(f"4. SLOs at {args.baseline:g} rps: "
              + ", ".join(f"{name} {v:.3g} < {lim:g}" for name, v, lim in checks)
              + f": {verdict(slo_ok)}")  # fmt: skip
        ok &= slo_ok

        # 5. park one node
        set_rate(args.baseline)
        victim = nodes[-1]
        pod = kubectl(
            "-n", "pbo", "get", "pods", "-l", "app=mock-llm",
            "--field-selector", f"spec.nodeName={victim}", "-o", "name",
        ).strip()  # fmt: skip
        print(f"5. cordon {victim}, delete {pod}")
        kubectl("cordon", victim)
        try:
            kubectl("-n", "pbo", "delete", pod, "--wait=true")
            time.sleep(45)  # stale timeout 3 s + scrape 15 s + margin
            now = query(f'DCGM_FI_DEV_POWER_USAGE{{node="{victim}"}}').get(victim, float("nan"))
            park_ok = abs(now - PARKED_W) < 1.0
            print(f"   {victim} now {now:.1f} W (parked = {PARKED_W} W): {verdict(park_ok)}")
            ok &= park_ok
        finally:
            kubectl("uncordon", victim)  # the pending replica returns to the node

    print("PHASE 1:", verdict(ok))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
