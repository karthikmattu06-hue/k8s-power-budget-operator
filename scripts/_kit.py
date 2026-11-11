"""Shared helpers for the cluster scripts (stdlib only, so they run with any python3)."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

PROM = "http://localhost:9090"
LOADGEN = "http://localhost:9101"


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


@contextmanager
def cluster_ports() -> Iterator[None]:
    with port_forward("monitoring", "svc/kps-prometheus", 9090), port_forward("pbo", "deploy/loadgen", 9101):
        yield


def results(promql: str) -> list[dict[str, Any]]:
    """Raw instant-query result: [{"metric": {...labels}, "value": [t, "v"]}, ...]."""
    url = f"{PROM}/api/v1/query?" + urllib.parse.urlencode({"query": promql})
    with urllib.request.urlopen(url, timeout=10) as r:
        data: list[dict[str, Any]] = json.load(r)["data"]["result"]
    return data


def query(promql: str, key: str = "node") -> dict[str, float]:
    """Instant query -> {label value: number}; series without the label are keyed 'all'."""
    return {s["metric"].get(key, "all"): float(s["value"][1]) for s in results(promql)}


def scalar(promql: str) -> float:
    return query(promql).get("all", float("nan"))


def series(promql: str, start: float, end: float, step: float = 5.0) -> list[tuple[float, float]]:
    """Range query for a single series -> [(t, value)]."""
    params = {"query": promql, "start": f"{start:.0f}", "end": f"{end:.0f}", "step": f"{step:g}"}
    url = f"{PROM}/api/v1/query_range?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as r:
        res = json.load(r)["data"]["result"]
    return [(float(t), float(v)) for t, v in res[0]["values"]] if res else []


def set_rate(rps: float) -> None:
    req = urllib.request.Request(f"{LOADGEN}/rate", data=json.dumps({"rps": rps}).encode(), method="POST")
    urllib.request.urlopen(req, timeout=5).read()


def kubectl(*args: str, stdin: str | None = None, check: bool = True) -> str:
    return subprocess.run(["kubectl", *args], check=check, capture_output=True, text=True, input=stdin).stdout


def deployment(name: str, ns: str = "pbo") -> tuple[int, int]:
    """(desired, ready) replicas."""
    d: dict[str, Any] = json.loads(kubectl("-n", ns, "get", "deploy", name, "-o", "json"))
    return int(d["spec"].get("replicas", 0)), int(d["status"].get("readyReplicas", 0) or 0)


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"
