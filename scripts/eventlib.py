"""Run one grid event on the cluster, score it, and save everything needed to plot it later.

Used by ``smoke_phase2.py`` (one event, pass/fail) and ``experiments.py`` (a batch). Stdlib only.

One event = settle at a baseline load -> apply a PowerBudget -> sample until Done -> wait for
recovery -> score -> save to ``results/<run-id>/``:

* ``spec.json``        what was asked (PowerBudget spec, load, tier mix)
* ``status.json``      the PowerBudget's final status (plan, actions, learned values)
* ``timeseries.csv``   5-s samples over the event, pulled from Prometheus at the end
* ``result.json``      the scores below

Scores
* reached_s           settled at/below target this many seconds after start (None: never)
* ramp_met            reached before the ramp deadline
* cap_peak_w          highest single 5-s reading during the hold; cap_above_s seconds above target
* intervals           aligned 5-min averages fully inside the hold (start time, watts)
* p99 TTFT per tier   over the whole event, client side
* release_ok          flex readmitted only after replicas were Ready; replicas and watts back
* failed_protected    premium + standard requests that failed during the event
* achieved_w          baseline minus mean hold watts; flex_shed_frac share of flex not served
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from _kit import deployment, kubectl, results, scalar, series, set_rate

INSTANT_W = "sum(DCGM_FI_DEV_POWER_USAGE)"
SHORT_W = "avg_over_time(sum(DCGM_FI_DEV_POWER_USAGE)[20s:5s])"  # sum first, then average
OUTCOMES = "sum by (tier, outcome) (pbo_loadgen_requests_total)"
P99_EVENT = 'histogram_quantile(0.99, sum by (le) (increase(pbo_loadgen_ttft_seconds_bucket{{tier="{tier}"}}[{w}s])))'
TIER_P99 = 'histogram_quantile(0.99, sum by (le) (rate(pbo_loadgen_ttft_seconds_bucket{{tier="{tier}"}}[1m])))'
# Time series saved for the figure (name -> PromQL). One value per 5 s step.
SERIES = {
    "watts": INSTANT_W,
    "watts_20s": SHORT_W,
    "flex_admit": "max(pbo_admission_flex_admit)",
    "replicas_ready": 'sum(kube_deployment_status_replicas_ready{deployment="mock-llm"})',
    "clock": "avg(pbo_clock_ratio)",
    **{f"{tier}_p99": TIER_P99.format(tier=tier) for tier in ("premium", "standard", "flex")},
    "rps_admitted": 'sum(rate(pbo_admission_requests_total{decision="admitted"}[1m]))',
    "rps_offered": "sum(rate(pbo_admission_arrivals_total[1m]))",
}


def outcomes() -> dict[tuple[str, str], float]:
    return {(r["metric"]["tier"], r["metric"]["outcome"]): float(r["value"][1]) for r in results(OUTCOMES)}


def pb_status(name: str) -> dict[str, Any]:
    out = kubectl("get", "pb", name, "-o", "json", check=False)
    status: dict[str, Any] = (json.loads(out).get("status") or {}) if out else {}
    return status


def set_mix(mix: dict[str, float]) -> None:
    import urllib.request

    from _kit import LOADGEN

    req = urllib.request.Request(f"{LOADGEN}/mix", data=json.dumps(mix).encode(), method="POST")
    urllib.request.urlopen(req, timeout=5).read()


def run_event(
    run_id: str,
    spec: dict[str, Any],
    rps: float = 8.0,
    mix: dict[str, float] | None = None,
    settle_s: int = 120,
    out_root: str = "results",
    name: str = "smoke-event",
    verbose: bool = True,
) -> dict[str, Any]:
    out = Path(out_root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    mix = mix or {"premium": 1 / 3, "standard": 1 / 3, "flex": 1 / 3}
    (out / "spec.json").write_text(json.dumps({"spec": spec, "rps": rps, "mix": mix}, indent=2))

    kubectl("delete", "pb", name, "--ignore-not-found", "--wait=true")
    set_rate(rps)
    set_mix(mix)
    if verbose:
        print(f"[{run_id}] baseline {rps:g} rps, mix {mix}, settle {settle_s} s; mock-llm {deployment('mock-llm')}")
    time.sleep(settle_s)
    before = outcomes()

    manifest = {"apiVersion": "pbo.dev/v1alpha1", "kind": "PowerBudget", "metadata": {"name": name}, "spec": spec}
    kubectl("apply", "-f", "-", stdin=json.dumps(manifest))
    t0 = time.time()
    if verbose:
        print(f"   applied {json.dumps(spec)}")
        print("   t(s)  phase      20s-avg W  instant W  target W  flex  clock  replicas(d/r)  message")

    samples: list[dict[str, Any]] = []
    st: dict[str, Any] = {}
    printed = -15.0
    deadline = t0 + float(spec.get("rampSeconds", 600)) + float(spec.get("holdSeconds", 900)) + 300
    while time.time() < deadline:
        st = pb_status(name)
        d, r = deployment("mock-llm")
        s = {"t": time.time(), "phase": st.get("phase", "?"), "flex": st.get("flexAdmit"), "desired": d, "ready": r}
        samples.append(s)
        if verbose and (s["t"] - t0 - printed >= 15 or st.get("phase") == "Done"):
            printed = s["t"] - t0
            flex = s["flex"] if s["flex"] is not None else "-"
            print(f"   {printed:5.0f}  {s['phase']:<9}  {scalar(SHORT_W):9.0f}  {scalar(INSTANT_W):9.0f}  "
                  f"{st.get('targetWatts') or float('nan'):8.0f}  {flex:>4}  {st.get('clockRatio', 1):>5}  "
                  f"{d}/{r:<12}  {str(st.get('message', ''))[:60]}")  # fmt: skip
        if st.get("phase") == "Done":
            break
        time.sleep(5)

    # -- recovery
    orig = int((st.get("originalReplicas") or {}).get("pbo/mock-llm", 5))
    t_rel = time.time()
    while time.time() - t_rel < 120 and deployment("mock-llm") != (orig, orig):
        time.sleep(5)
    time.sleep(40)
    end = time.time()

    # -- score
    started = float(st.get("startedAt") or t0)
    target = float(st.get("targetWatts") or float("nan"))
    hold_s = float(spec.get("holdSeconds", 900))
    hold_until = float(st.get("holdUntil") or end)
    hold_start = hold_until - hold_s
    hold = series(INSTANT_W, hold_start, hold_until)
    peak = max((v for _, v in hold), default=float("nan"))
    first = (int(hold_start) // 300 + 1) * 300
    intervals = []
    for a in range(first, int(hold_until) - 299, 300):
        vals = [v for t, v in hold if a <= t < a + 300]
        if vals:
            intervals.append([a, round(sum(vals) / len(vals), 1)])
    span = int(hold_until - started)
    p99 = {t: scalar(P99_EVENT.format(tier=t, w=span)) for t in ("premium", "standard", "flex")}
    rel = [s for s in samples if s["phase"] == "Releasing"]
    early = [s for s in rel if (s["flex"] or 0) >= 1.0 and s["ready"] < s["desired"]]
    base = float(st.get("baselineWatts") or float("nan"))
    w_after = scalar(SHORT_W)
    after = outcomes()
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    failed = sum(v for (tier, o), v in delta.items() if tier != "flex" and o != "ok")
    flex_all = sum(v for (tier, _), v in delta.items() if tier == "flex")
    flex_shed = sum(v for (tier, o), v in delta.items() if tier == "flex" and o != "ok")
    mean_hold = sum(v for _, v in hold) / len(hold) if hold else float("nan")
    reached = st.get("reachedAt")
    result = {
        "run_id": run_id,
        "policy": spec.get("policy", "optimizer"),
        "mode": spec.get("mode", "interval"),
        "requested_w": spec.get("reduction", {}).get("watts"),
        "rps": rps,
        "mix": mix,
        "baseline_w": base,
        "target_w": target,
        "aim_w": st.get("planTargetWatts"),
        "margin_w": st.get("effectiveMarginWatts"),
        "noise_w": st.get("noiseWatts"),
        "reached_s": round(reached - started, 1) if reached else None,
        "ramp_met": st.get("rampMet") is True,
        "done": st.get("phase") == "Done",
        "cap_peak_w": round(peak, 1),
        "cap_above_s": 5 * sum(1 for _, v in hold if v > target),
        "intervals": intervals,
        "intervals_ok": bool(intervals) and all(v <= target for _, v in intervals),
        "hold_mean_w": round(mean_hold, 1),
        "achieved_w": round(base - mean_hold, 1),
        "premium_p99_s": round(p99["premium"], 3),
        "standard_p99_s": round(p99["standard"], 3),
        "flex_p99_s": round(p99["flex"], 3),
        "flex_shed_frac": round(flex_shed / flex_all, 3) if flex_all else 0.0,
        "failed_protected": failed,
        "release_ok": not early and deployment("mock-llm") == (orig, orig) and w_after >= 0.95 * base,
        "final_plan": st.get("plan"),
        "spike_w": st.get("spikeWatts"),
        "spike_source": st.get("spikeSource"),
        "t0": t0,
        "hold_start": hold_start,
        "hold_until": hold_until,
    }
    (out / "status.json").write_text(json.dumps(st, indent=2))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    save_timeseries(
        out / "timeseries.csv", t0 - 60, end, extra={"target_w": target, "aim_w": st.get("planTargetWatts")}
    )
    kubectl("delete", "pb", name, "--wait=true", check=False)
    return result


def save_timeseries(path: Path, start: float, end: float, extra: dict[str, Any]) -> None:
    cols: dict[str, dict[float, float]] = {}
    for key, q in SERIES.items():
        try:
            cols[key] = dict(series(q, start, end))
        except Exception:
            cols[key] = {}
    times = sorted({t for c in cols.values() for t in c})
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", *SERIES, *extra])
        for t in times:
            w.writerow([round(t, 1), *(cols[k].get(t, "") for k in SERIES), *extra.values()])
