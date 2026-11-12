#!/usr/bin/env python3
"""Plot one saved event: ``python scripts/event_report.py results/<run>`` -> ``results/<run>/event.png``.

Three panels sharing a time axis (seconds from the PowerBudget being applied):
  1. Cluster watts: single 5-s readings and the 20-s average, with baseline, target (cap) and the
     planner's aim; the hold is shaded.
  2. The knobs: ready replicas, flex admit fraction, GPU clock ratio.
  3. p99 TTFT per tier (client side, 1-min window) against the premium (2 s) and standard (6 s) SLOs.
Needs matplotlib (``pip install -e ".[study]"``); no cluster needed -- it reads the saved files.
With several run directories, also writes ``compare.png`` (watts of each run on one axis).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run: Path) -> tuple[dict[str, list[float]], dict[str, object]]:
    rows = list(csv.DictReader((run / "timeseries.csv").open()))
    res = json.loads((run / "result.json").read_text())
    t0 = float(res["t0"])
    cols: dict[str, list[float]] = {k: [] for k in rows[0]} if rows else {}
    for r in rows:
        for k, v in r.items():
            cols[k].append(float(v) if v not in ("", "NaN", "nan") else float("nan"))
    cols["s"] = [t - t0 for t in cols.get("t", [])]
    return cols, res


def plot_one(run: Path) -> Path:
    c, r = load(run)
    t0 = float(r["t0"])  # type: ignore[arg-type]
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.4, 1.6]})
    hs, he = float(r["hold_start"]) - t0, float(r["hold_until"]) - t0  # type: ignore[arg-type]
    for a in ax:
        a.axvspan(hs, he, color="0.92", zorder=0)
        a.axvline(0, color="0.5", lw=0.8, ls=":")
    a = ax[0]
    a.plot(c["s"], c["watts"], lw=0.8, color="tab:blue", alpha=0.6, label="5-s reading")
    a.plot(c["s"], c["watts_20s"], lw=1.6, color="tab:blue", label="20-s average")
    a.axhline(float(r["baseline_w"]), color="0.4", lw=1, ls="--", label="baseline")  # type: ignore[arg-type]
    a.axhline(float(r["target_w"]), color="tab:red", lw=1.4, label=f"target ({r['mode']})")  # type: ignore[arg-type]
    if r.get("aim_w") is not None:
        a.axhline(float(r["aim_w"]), color="tab:red", lw=1, ls=":", label="planner aim")  # type: ignore[arg-type]
    a.set_ylabel("cluster W")
    a.legend(loc="upper right", fontsize=8, ncol=3)
    a.set_title(f"{r['run_id']}: {r['policy']}, {r['mode']}, -{r['requested_w']} W requested, "
                f"{r['achieved_w']} W achieved, flex shed {100 * float(r['flex_shed_frac']):.0f} %",  # type: ignore[arg-type]
                fontsize=10)  # fmt: skip
    a = ax[1]
    a.plot(c["s"], c["replicas_ready"], color="tab:green", label="replicas ready", drawstyle="steps-post")
    a.set_ylabel("replicas")
    b = a.twinx()
    b.plot(c["s"], c["flex_admit"], color="tab:orange", label="flex admit", drawstyle="steps-post")
    b.plot(c["s"], c["clock"], color="tab:purple", label="GPU clock", drawstyle="steps-post")
    b.set_ylim(-0.05, 1.05)
    b.set_ylabel("fraction")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, ncol=3)
    a = ax[2]
    for tier, col in (("premium", "tab:red"), ("standard", "tab:blue"), ("flex", "tab:gray")):
        a.plot(c["s"], c[f"{tier}_p99"], color=col, label=f"{tier} p99 TTFT")
    a.axhline(2.0, color="tab:red", lw=0.8, ls="--")
    a.axhline(6.0, color="tab:blue", lw=0.8, ls="--")
    a.set_yscale("log")
    a.set_ylabel("seconds")
    a.set_xlabel("seconds since the PowerBudget was applied (grey: hold)")
    a.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    path = run / "event.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def compare(runs: list[Path], out: Path) -> Path:
    fig, a = plt.subplots(figsize=(9, 4))
    for run in runs:
        c, r = load(run)
        rel = [w - float(r["baseline_w"]) for w in c["watts_20s"]]  # type: ignore[arg-type]
        a.plot(c["s"], rel, lw=1.4, label=f"{r['policy']}/{r['mode']}")
    a.axhline(0, color="0.5", lw=0.8)
    a.set_xlabel("seconds since the PowerBudget was applied")
    a.set_ylabel("W relative to baseline (20-s avg)")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    runs = [Path(p) for p in sys.argv[1:]]
    if not runs:
        print(__doc__)
        return 1
    for run in runs:
        print(plot_one(run))
    if len(runs) > 1:
        print(compare(runs, runs[0].parent / "compare.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
