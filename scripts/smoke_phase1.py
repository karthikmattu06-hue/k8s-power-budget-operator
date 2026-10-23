#!/usr/bin/env python3
"""Phase 1 acceptance test on a running kind cluster ("done when", checked by machine).

Checks, in order:
  1. Every emulated GPU node publishes DCGM_FI_DEV_POWER_USAGE to Prometheus.
  2. Stepping the load up raises per-node watts and p99 TTFT / TPOT (low rate, then high rate).
  3. Removing the serving pod from one node (cordon + delete) drops that node to parked watts.

Needs: kubectl pointed at the kind cluster, `make deploy` done. Stdlib only.
Usage: python3 scripts/smoke_phase1.py [--low 4] [--high 17] [--hold 120]
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


def query(promql: str) -> dict[str, float]:
    """Instant query -> {node: value}. Series without a node label are keyed 'all'."""
    url = f"{PROM}/api/v1/query?" + urllib.parse.urlencode({"query": promql})
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.load(r)["data"]["result"]
    return {s["metric"].get("node", "all"): float(s["value"][1]) for s in data}


def set_rate(rps: float) -> None:
    req = urllib.request.Request(f"{LOADGEN}/rate", data=json.dumps({"rps": rps}).encode(), method="POST")
    urllib.request.urlopen(req, timeout=5).read()


def kubectl(*args: str) -> str:
    return subprocess.run(["kubectl", *args], check=True, capture_output=True, text=True).stdout


def show(title: str, values: dict[str, float], unit: str) -> None:
    print(f"  {title}: " + ", ".join(f"{k.replace('pbo-', '')}={v:.2f}{unit}" for k, v in sorted(values.items())))


WATTS = "avg by (node) (avg_over_time(DCGM_FI_DEV_POWER_USAGE[1m]))"
P99 = "histogram_quantile(0.99, sum by (le{by}) (rate(vllm:{m}_bucket[1m])))"


def measure(label: str) -> tuple[float, float, float]:
    w = query(WATTS)
    ttft_node = query(P99.format(by=", node", m="time_to_first_token_seconds"))
    ttft = query(P99.format(by="", m="time_to_first_token_seconds")).get("all", float("nan"))
    tpot = query(P99.format(by="", m="time_per_output_token_seconds")).get("all", float("nan"))
    print(f"[{label}]")
    show("watts", w, " W")
    show("p99 TTFT", ttft_node, " s")
    print(f"  cluster p99 TTFT={ttft:.2f} s, p99 TPOT={tpot * 1000:.1f} ms")
    return sum(w.values()) / max(len(w), 1), ttft, tpot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", type=float, default=4.0)
    ap.add_argument("--high", type=float, default=17.0)
    ap.add_argument("--hold", type=int, default=120, help="seconds at each rate (>= 2 x rate window)")
    args = ap.parse_args()
    ok = True

    with port_forward("monitoring", "svc/kps-prometheus", 9090), port_forward("pbo", "deploy/loadgen", 9101):
        # 1. every node reports watts
        nodes = [n for n in kubectl("get", "nodes", "-l", "pbo.dev/gpu=emulated", "-o", "name").split()]
        nodes = [n.split("/")[1] for n in nodes]
        deadline = time.time() + 120
        while len(query("DCGM_FI_DEV_POWER_USAGE")) < len(nodes) and time.time() < deadline:
            time.sleep(5)
        seen = query("DCGM_FI_DEV_POWER_USAGE")
        print(f"1. watts series: {len(seen)} of {len(nodes)} nodes")
        ok &= len(seen) == len(nodes)

        # 2. load step
        print(f"2. load step: {args.low} rps then {args.high} rps, {args.hold} s each")
        set_rate(args.low)
        time.sleep(args.hold)
        w_lo, ttft_lo, tpot_lo = measure(f"{args.low} rps")
        set_rate(args.high)
        time.sleep(args.hold)
        w_hi, ttft_hi, tpot_hi = measure(f"{args.high} rps")
        step_ok = w_hi > w_lo * 1.2 and ttft_hi > ttft_lo and tpot_hi > tpot_lo
        print(
            f"   mean watts {w_lo:.0f} -> {w_hi:.0f}; p99 TTFT {ttft_lo:.2f} -> {ttft_hi:.2f} s: "
            f"{'PASS' if step_ok else 'FAIL'}"
        )
        ok &= step_ok

        # 3. park one node
        victim = nodes[-1]
        pod = kubectl(
            "-n",
            "pbo",
            "get",
            "pods",
            "-l",
            "app=mock-llm",
            "--field-selector",
            f"spec.nodeName={victim}",
            "-o",
            "name",
        ).strip()
        print(f"3. cordon {victim}, delete {pod}")
        kubectl("cordon", victim)
        try:
            kubectl("-n", "pbo", "delete", pod, "--wait=true")
            time.sleep(45)  # stale timeout 3 s + scrape 15 s + 1 m average needs ~ a minute to settle fully
            now = query(f'DCGM_FI_DEV_POWER_USAGE{{node="{victim}"}}').get(victim, float("nan"))
            park_ok = abs(now - PARKED_W) < 1.0
            print(f"   {victim} now {now:.1f} W (parked = {PARKED_W} W): {'PASS' if park_ok else 'FAIL'}")
            ok &= park_ok
        finally:
            kubectl("uncordon", victim)  # the pending replica returns to the node
        set_rate(10)

    print("PHASE 1:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
