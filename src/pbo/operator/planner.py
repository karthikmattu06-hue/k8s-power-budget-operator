"""Choosing what to do: the optimiser, and the fixed-order ladder kept as a baseline to beat.

The decision, stated as an optimisation problem
-----------------------------------------------
Two knobs in Phase 2:

* ``flex_admit`` -- share of flex requests let through (1, 0.75, 0.5, 0.25, 0). Only flex
  consented to be shed (ADR-0001), so premium and standard are never touched.
* ``replicas`` -- serving replicas of the consenting pool; each one removed parks a GPU.

Pick the end state (replicas, flex_admit) that

    minimises   flex requests shed per second  +  0.05 x replicas removed
    subject to  predicted cluster watts            <= target           (the power budget)
                load / (replicas x capacity)        <= utilisation limit (premium SLO proxy)
                replicas >= minReplicas;  no knob moves back up during an event

Why these terms
* Shedding flex *loses work*, so it is the thing to minimise. Parking a GPU loses nothing as long
  as the remaining replicas keep up -- which is what the utilisation constraint guards (Phase 1:
  premium p99 TTFT held at ~50 % utilisation and got marginal near 60 %). The small replica term
  only breaks ties toward fewer disruptions.
* The search is exhaustive: 5 admit levels x at most a handful of replica counts. No solver
  needed; every candidate is checked, so the answer is the true optimum *of the model*.

Order of execution falls out instead of being written down: shedding flex lowers utilisation and
removing replicas raises it, so doing the admission change first keeps every intermediate state
inside the utilisation limit whenever the end state is. The operator applies the whole plan in one
tick (admission and replicas together; the admission change takes effect within seconds, the
replicas drain over ~10 s), then measures, corrects the model and plans again (receding horizon):
feedforward for the big jump, feedback for the trim. The plan is never trusted blindly.

If no state meets the target, the plan is the deepest cut that keeps premium safe, and the gap
is reported as a shortfall: budgets bind on flex, they are advisory on premium and standard.
"""

from __future__ import annotations

from dataclasses import dataclass

from pbo.operator.model import PowerModel

ADMIT_LEVELS = (1.0, 0.75, 0.5, 0.25, 0.0)
REPLICA_COST = 0.05  # in "flex req/s shed" units: a tie-breaker, not a real price


@dataclass(frozen=True)
class Knobs:
    replicas: int
    flex_admit: float


@dataclass(frozen=True)
class Load:
    """Offered load at the admission proxy, req/s (what arrives, before any shedding)."""

    protected_rps: float  # premium + standard: always admitted
    flex_rps: float

    def admitted(self, flex_admit: float) -> float:
        return self.protected_rps + flex_admit * self.flex_rps


@dataclass(frozen=True)
class Constraints:
    target_w: float
    min_replicas: int
    util_limit: float  # e.g. 0.6
    capacity_rps: float  # one replica's sustainable throughput, req/s
    nodes: int  # GPU nodes the pool can occupy (parked ones still draw parked_w)
    allow_replica_change: bool = True  # False in cap mode when a removal spike would cross the cap


@dataclass(frozen=True)
class Plan:
    end: Knobs
    predicted_w: float
    utilisation: float
    flex_shed_rps: float
    feasible: bool  # meets the power target
    next_move: str  # "admission", "replica", or "none"
    reason: str

    def next_knobs(self, current: Knobs) -> Knobs:
        """The single move to make now (admission first, then one replica)."""
        if self.next_move == "admission":
            return Knobs(current.replicas, self.end.flex_admit)
        if self.next_move == "replica":
            return Knobs(current.replicas - 1, current.flex_admit)
        return current


def utilisation(knobs: Knobs, load: Load, capacity_rps: float) -> float:
    return load.admitted(knobs.flex_admit) / (max(knobs.replicas, 1) * capacity_rps)


def _first_move(cur: Knobs, end: Knobs) -> str:
    if end.flex_admit < cur.flex_admit:
        return "admission"
    if end.replicas < cur.replicas:
        return "replica"
    return "none"


def optimise(cur: Knobs, load: Load, model: PowerModel, c: Constraints) -> Plan:
    """Exhaustive search over end states reachable from ``cur`` without moving any knob up."""
    util_now = utilisation(cur, load, c.capacity_rps)
    # Never require premium to be *safer* than it is now; only forbid making it worse past the limit.
    util_cap = max(c.util_limit, util_now)
    lowest = c.min_replicas if c.allow_replica_change else cur.replicas
    best: Plan | None = None
    fallback: Plan | None = None
    for n in range(cur.replicas, max(lowest, 0) - 1, -1):
        for f in (lvl for lvl in ADMIT_LEVELS if lvl <= cur.flex_admit + 1e-9):
            k = Knobs(n, f)
            u = utilisation(k, load, c.capacity_rps)
            if u > util_cap + 1e-9:
                continue  # would push premium past the limit
            w = model.cluster_w(n, load.admitted(f), c.nodes)
            shed = (1.0 - f) * load.flex_rps
            cost = shed + REPLICA_COST * (cur.replicas - n)
            plan = Plan(k, w, u, shed, w <= c.target_w, _first_move(cur, k), "")
            if plan.feasible:
                # cheapest; on a tie, the one closer to the target (less overshoot)
                if best is None or (cost, -w) < (_cost(best, cur), -best.predicted_w):
                    best = plan
            elif fallback is None or w < fallback.predicted_w:
                fallback = plan  # deepest cut that keeps premium safe
    if best is not None:
        why = (
            f"cheapest plan meeting {c.target_w:.0f} W: {best.end.replicas} replicas, flex admit "
            f"{best.end.flex_admit:.2f} -> {best.predicted_w:.0f} W predicted, util {best.utilisation:.2f}"
        )
        return Plan(best.end, best.predicted_w, best.utilisation, best.flex_shed_rps, True, best.next_move, why)
    assert fallback is not None  # the current state itself is always a candidate
    why = (
        f"no plan meets {c.target_w:.0f} W within the utilisation limit; deepest safe cut: "
        f"{fallback.end.replicas} replicas, flex admit {fallback.end.flex_admit:.2f} -> {fallback.predicted_w:.0f} W"
    )
    return Plan(fallback.end, fallback.predicted_w, fallback.utilisation, fallback.flex_shed_rps, False,
                fallback.next_move, why)  # fmt: skip


def _cost(p: Plan, cur: Knobs) -> float:
    return p.flex_shed_rps + REPLICA_COST * (cur.replicas - p.end.replicas)


def ladder(cur: Knobs, load: Load, model: PowerModel, c: Constraints) -> Plan:
    """Baseline policy with a fixed order: shed all flex first, then remove replicas one by one.

    It ignores the power model for *choosing* (only reports a prediction) and ignores the
    utilisation limit -- which is exactly what a hand-written rule list does.
    """
    if cur.flex_admit > 0:
        end = Knobs(cur.replicas, 0.0)
        move = "admission"
    elif c.allow_replica_change and cur.replicas > c.min_replicas:
        end = Knobs(cur.replicas - 1, 0.0)
        move = "replica"
    else:
        end, move = cur, "none"
    w = model.cluster_w(end.replicas, load.admitted(end.flex_admit), c.nodes)
    u = utilisation(end, load, c.capacity_rps)
    return Plan(end, w, u, (1 - end.flex_admit) * load.flex_rps, w <= c.target_w, move, f"ladder step: {move}")
