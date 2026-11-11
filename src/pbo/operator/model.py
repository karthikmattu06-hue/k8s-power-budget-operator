"""Power predictor: how many watts would the cluster draw with ``n`` replicas serving ``rate`` req/s?

The optimiser needs to compare plans *before* acting ("4 replicas with a quarter of flex, or 3
replicas with all of it?"). It does that with one measured curve per serving replica:

    node_w(x) = watts of a GPU node whose replica receives x requests/s

and assumes the load balancer spreads requests evenly (it does: one connection per request),
so with ``n`` replicas on ``nodes`` GPU nodes:

    cluster_w(n, rate) = n * node_w(rate / n) + (nodes - n) * parked_w + bias_w

Where the curve comes from, in order of preference:

1. **Calibration** -- ``scripts/calibrate.py`` steps the load and records (x, node watts) points
   into the ConfigMap ``pbo-power-model``. Measured on *this* cluster, so it includes the
   redistribution penalty Phase 1 found (fuller batches decode slower and draw more).
2. **Contract prior** -- two points from the lever-curve contract: idle (``parked``) and fully
   busy (``busy_monolithic``) at the replica's capacity, joined by a straight line. Crude, and
   labelled as such in the event's status.

``bias_w`` is the feedback term: after each action has settled, the operator compares what it
measured with what this model predicted for the current state and keeps the difference
(smoothed). The model's *shape* ranks plans; the bias keeps its *level* honest.

The curve is never the emulator's own formula -- that would make prediction trivially perfect
and tell us nothing about how this would behave on real GPUs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any


@dataclass(frozen=True)
class PowerModel:
    points: tuple[tuple[float, float], ...]  # (requests/s per replica, node watts), sorted by x
    parked_w: float  # a GPU node with no replica
    max_w: float  # a node can never draw more than this (fully busy)
    bias_w: float = 0.0
    source: str = "contract prior"

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("a power curve needs at least two points")

    @staticmethod
    def from_points(
        points: Sequence[Sequence[float]], parked_w: float, max_w: float, source: str = "calibration"
    ) -> PowerModel:
        pts = sorted((float(x), float(w)) for x, w in points)
        if pts[0][0] > 0:  # an idle replica draws about what a parked node does
            pts.insert(0, (0.0, parked_w))
        return PowerModel(tuple(pts), parked_w, max_w, 0.0, source)

    @staticmethod
    def contract_prior(parked_w: float, busy_w: float, capacity_rps: float) -> PowerModel:
        return PowerModel(((0.0, parked_w), (capacity_rps, busy_w)), parked_w, busy_w, 0.0, "contract prior")

    @staticmethod
    def from_configmap(data: Mapping[str, Any], parked_w: float, max_w: float) -> PowerModel:
        """``data`` is the parsed ``model.json`` written by scripts/calibrate.py."""
        return PowerModel.from_points(
            data["points"], parked_w, max_w, source=f"calibration {data.get('measuredAt', '')}"
        )

    def with_bias(self, bias_w: float) -> PowerModel:
        return replace(self, bias_w=bias_w)

    def node_w(self, x: float) -> float:
        """Watts of one node whose replica gets ``x`` req/s: linear between points, straight-line
        extrapolation past the last one (load we never calibrated), never above ``max_w``."""
        pts = self.points
        x = max(x, 0.0)
        for (x0, w0), (x1, w1) in pairwise(pts):
            if x <= x1:
                return w0 + (w1 - w0) * (x - x0) / (x1 - x0) if x1 > x0 else w1
        (x0, w0), (x1, w1) = pts[-2], pts[-1]
        slope = (w1 - w0) / (x1 - x0) if x1 > x0 else 0.0
        return min(w1 + slope * (x - x1), self.max_w)

    def cluster_w(self, replicas: int, rate: float, nodes: int) -> float:
        """Predicted cluster watts (including the bias term)."""
        active = max(replicas, 0)
        per = rate / active if active else 0.0
        return active * self.node_w(per) + max(nodes - active, 0) * self.parked_w + self.bias_w
