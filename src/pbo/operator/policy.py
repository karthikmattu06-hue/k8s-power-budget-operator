"""The event state machine: pure functions, no Kubernetes, no clocks, no network.

``main.py`` gathers an :class:`Observation` every tick (5 s), calls :func:`step`, carries out the
returned actions and stores the returned :class:`State` in the PowerBudget's ``.status``.
Everything decided lives here or in ``planner.py``, so it is all unit-testable with a fake clock.

Phases
------
::

    Pending -> Ramping -> Holding -> Releasing -> Done

* **Pending -> Ramping.** First valid power reading becomes the baseline. The budget is
  ``target = baseline - reduction``; the planner aims at ``target - margin`` so noise and small
  model errors do not cross the line.
* **Ramping.** Plan, make one move, wait until it has *settled*, measure, plan again. "Settled"
  = the lever's ``speed_s`` from the contract (admission 8 s, replica 90 s) plus the sensing lag
  (scrape + averaging window). No decision is made on a reading taken mid-transition -- the
  first Phase 2 run declared victory on a dip that was not real.
* **Holding** starts when a settled reading is at or below target, or when the ramp deadline
  passes. It lasts ``hold_s``. The loop keeps correcting if the power drifts up.
* **Releasing.** Replicas are restored first; flex is readmitted only once they are Ready.
  Readmitting flex onto the reduced pool first would overload it.

Two ways an obligation can be scored (``mode``)
-----------------------------------------------
* ``interval`` -- demand-response settlement: each metered interval's *average* must be at or
  below target (5 min here; markets use 15 or 60, and every aligned 5-min pass implies those).
* ``cap`` -- a hard limit: no reading may exceed the target during the hold. Removing a replica
  can raise power briefly (the draining pod still serves while the others take its load; small,
  0-5 W measured), so in cap mode a replica move is allowed during the hold only if ``reading + spike estimate``
  stays under the cap, and during the ramp only if it will have settled before the deadline.
  Admission changes have no spike and are always allowed. The spike estimate is learned from
  the operator's own past removals.

Feedback
--------
After every settled tick the gap between measured and predicted watts updates ``bias_w`` (half
old, half new). The planner's model is shape; the bias keeps its level honest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pbo.operator.levers import LeverBook
from pbo.operator.model import PowerModel
from pbo.operator.planner import Constraints, Knobs, Load, Plan, ladder, optimise

PENDING, RAMPING, HOLDING, RELEASING, DONE = "Pending", "Ramping", "Holding", "Releasing", "Done"
ACTIVE = (RAMPING, HOLDING, RELEASING)
DEFER_MAX_S = 600.0  # flex time-to-last-token SLO: nothing may be held longer (ADR-0001)
RELEASE_TIMEOUT_S = 120.0
DEFAULT_SPIKE_W = 50.0  # prior for the power rise after a replica removal, until one is measured


# -- inputs -------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """Validated copy of the PowerBudget ``.spec``."""

    reduction_watts: float | None = None
    reduction_fraction: float | None = None
    mode: str = "interval"  # or "cap"
    interval_s: float = 300.0
    margin_w: float = 30.0
    noise_sigmas: float = 3.5  # cap mode: margin is at least this many std devs of single readings
    ramp_s: float = 600.0
    hold_s: float = 900.0
    min_replicas: int = 1
    policy: str = "optimizer"  # or "ladder"
    flex_action: str = "reject"  # or "defer"
    util_limit: float = 0.6
    capacity_rps: float = 3.3
    sensing_lag_s: float = 30.0

    @staticmethod
    def from_k8s(spec: Mapping[str, Any]) -> Spec:
        red = spec.get("reduction", {}) or {}
        s = Spec(
            reduction_watts=_opt_float(red.get("watts")),
            reduction_fraction=_opt_float(red.get("fraction")),
            mode=str(spec.get("mode", "interval")),
            interval_s=float(spec.get("intervalSeconds", 300)),
            margin_w=float(spec.get("marginWatts", 30)),
            noise_sigmas=float(spec.get("noiseSigmas", 3.5)),
            ramp_s=float(spec.get("rampSeconds", 600)),
            hold_s=float(spec.get("holdSeconds", 900)),
            min_replicas=int(spec.get("minReplicas", 1)),
            policy=str(spec.get("policy", "optimizer")),
            flex_action=str(spec.get("flexAction", "reject")),
            util_limit=float(spec.get("utilizationLimit", 0.6)),
            capacity_rps=float(spec.get("replicaCapacityRps", 3.3)),
            sensing_lag_s=float(spec.get("sensingLagSeconds", 30)),
        )
        if (s.reduction_watts is None) == (s.reduction_fraction is None):
            raise ValueError("spec.reduction needs exactly one of 'watts' or 'fraction'")
        if s.reduction_watts is not None and s.reduction_watts <= 0:
            raise ValueError("reduction.watts must be > 0")
        if s.reduction_fraction is not None and not 0 < s.reduction_fraction < 1:
            raise ValueError("reduction.fraction must be in (0, 1)")
        if s.mode not in ("interval", "cap"):
            raise ValueError("mode must be 'interval' or 'cap'")
        if s.policy not in ("optimizer", "ladder"):
            raise ValueError("policy must be 'optimizer' or 'ladder'")
        if s.flex_action not in ("reject", "defer"):
            raise ValueError("flexAction must be 'reject' or 'defer'")
        if s.flex_action == "defer" and s.ramp_s + s.hold_s > DEFER_MAX_S:
            raise ValueError(f"defer holds flex for up to ramp + hold; that must be <= {DEFER_MAX_S:.0f} s")
        if s.min_replicas < 0 or s.ramp_s <= 0 or s.hold_s < 0 or s.margin_w < 0 or s.noise_sigmas < 0:
            raise ValueError("minReplicas, holdSeconds, marginWatts must be >= 0 and rampSeconds > 0")
        if not 0 < s.util_limit <= 1 or s.capacity_rps <= 0:
            raise ValueError("utilizationLimit must be in (0, 1] and replicaCapacityRps > 0")
        return s

    def target(self, baseline_w: float) -> float:
        cut = self.reduction_watts if self.reduction_watts is not None else baseline_w * (self.reduction_fraction or 0)
        return baseline_w - cut


def _opt_float(x: Any) -> float | None:
    return None if x is None else float(x)


@dataclass(frozen=True)
class Replica:
    pod: str
    node: str
    fill: float | None  # batch fill on its node, 0..1; None if unknown


@dataclass(frozen=True)
class Pool:
    """A consenting Deployment."""

    key: str  # "namespace/name"
    replicas: int  # desired
    ready: int = 0
    pods: tuple[Replica, ...] = ()


@dataclass(frozen=True)
class Observation:
    now: float  # epoch seconds
    cluster_w: float | None  # short average (decisions)
    instant_w: float | None  # latest scrape (cap compliance, spike learning)
    nodes: int  # GPU nodes reporting power
    load: Load | None  # offered load at the admission proxy; None until it has data
    pools: tuple[Pool, ...] = ()
    noise_w: float | None = None  # std dev of single cluster readings over the last 2 min


# -- state carried in .status ----------------------------------------------------


@dataclass(frozen=True)
class State:
    phase: str = PENDING
    started_at: float | None = None
    baseline_w: float | None = None
    target_w: float | None = None  # the budget (cap in cap mode)
    plan_target_w: float | None = None  # target - margin: what the planner aims for
    measured_w: float | None = None
    instant_w: float | None = None
    reached_at: float | None = None
    ramp_met: bool | None = None
    hold_until: float | None = None
    release_started_at: float | None = None
    flex_admit: float = 1.0
    replicas: int | None = None
    original: Mapping[str, int] = field(default_factory=dict)
    last_move: str | None = None
    last_move_at: float | None = None
    pre_move_w: float | None = None
    post_move_peak_w: float | None = None
    spike_w: float = DEFAULT_SPIKE_W
    spike_source: str = "default"
    spike_pending: bool = False  # a replica move is settling; its peak is being watched
    noise_w: float | None = None  # std dev of single readings at event start
    margin_eff_w: float | None = None  # margin actually used: max(marginWatts, noiseSigmas x noise)
    bias_w: float = 0.0
    peak_hold_w: float | None = None
    above_target_s: float = 0.0
    last_tick_at: float | None = None
    shortfall_w: float = 0.0
    plan: Mapping[str, Any] = field(default_factory=dict)
    actions: tuple[Mapping[str, Any], ...] = ()
    message: str = ""

    def to_k8s(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "startedAt": self.started_at,
            "baselineWatts": _r(self.baseline_w),
            "targetWatts": _r(self.target_w),
            "planTargetWatts": _r(self.plan_target_w),
            "measuredWatts": _r(self.measured_w),
            "instantWatts": _r(self.instant_w),
            "reachedAt": self.reached_at,
            "rampMet": self.ramp_met,
            "holdUntil": self.hold_until,
            "releaseStartedAt": self.release_started_at,
            "flexAdmit": self.flex_admit,
            "replicas": self.replicas,
            "originalReplicas": dict(self.original),
            "lastMove": self.last_move,
            "lastMoveAt": self.last_move_at,
            "preMoveWatts": _r(self.pre_move_w),
            "postMovePeakWatts": _r(self.post_move_peak_w),
            "spikeWatts": _r(self.spike_w),
            "spikeSource": self.spike_source,
            "spikePending": self.spike_pending,
            "noiseWatts": _r(self.noise_w),
            "effectiveMarginWatts": _r(self.margin_eff_w),
            "biasWatts": _r(self.bias_w),
            "peakHoldWatts": _r(self.peak_hold_w),
            "aboveTargetSeconds": round(self.above_target_s, 1),
            "lastTickAt": self.last_tick_at,
            "shortfallWatts": _r(self.shortfall_w),
            "plan": dict(self.plan),
            "actions": [dict(a) for a in self.actions],
            "message": self.message,
        }

    @staticmethod
    def from_k8s(status: Mapping[str, Any] | None) -> State:
        s = status or {}
        return State(
            phase=s.get("phase") or PENDING,
            started_at=s.get("startedAt"),
            baseline_w=s.get("baselineWatts"),
            target_w=s.get("targetWatts"),
            plan_target_w=s.get("planTargetWatts"),
            measured_w=s.get("measuredWatts"),
            instant_w=s.get("instantWatts"),
            reached_at=s.get("reachedAt"),
            ramp_met=s.get("rampMet"),
            hold_until=s.get("holdUntil"),
            release_started_at=s.get("releaseStartedAt"),
            flex_admit=float(s["flexAdmit"]) if s.get("flexAdmit") is not None else 1.0,
            replicas=s.get("replicas"),
            original=dict(s.get("originalReplicas") or {}),
            last_move=s.get("lastMove"),
            last_move_at=s.get("lastMoveAt"),
            pre_move_w=s.get("preMoveWatts"),
            post_move_peak_w=s.get("postMovePeakWatts"),
            spike_w=float(s["spikeWatts"]) if s.get("spikeWatts") is not None else DEFAULT_SPIKE_W,
            spike_source=s.get("spikeSource") or "default",
            spike_pending=bool(s.get("spikePending")),
            noise_w=s.get("noiseWatts"),
            margin_eff_w=s.get("effectiveMarginWatts"),
            bias_w=float(s.get("biasWatts") or 0.0),
            peak_hold_w=s.get("peakHoldWatts"),
            above_target_s=float(s.get("aboveTargetSeconds") or 0.0),
            last_tick_at=s.get("lastTickAt"),
            shortfall_w=float(s.get("shortfallWatts") or 0.0),
            plan=dict(s.get("plan") or {}),
            actions=tuple(s.get("actions") or ()),
            message=s.get("message") or "",
        )


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 1)


# -- outputs -----------------------------------------------------------------------


@dataclass(frozen=True)
class SetAdmission:
    flex_admit: float
    mode: str  # reject | defer


@dataclass(frozen=True)
class ScaleDown:
    pool: str
    to_replicas: int
    victim: str | None
    victim_node: str | None


@dataclass(frozen=True)
class Restore:
    replicas: Mapping[str, int]


Action = SetAdmission | ScaleDown | Restore


# -- helpers -----------------------------------------------------------------------


def choose_victim(pool: Pool) -> Replica | None:
    """The pod on the node with the emptiest batch: fewest in-flight requests disturbed."""
    if not pool.pods:
        return None
    return min(pool.pods, key=lambda r: (r.fill is None, r.fill if r.fill is not None else 0.0, r.pod))


def settled(st: State, book: LeverBook, spec: Spec, now: float) -> bool:
    if st.last_move is None or st.last_move_at is None:
        return True
    return now - st.last_move_at >= book.speed_s.get(st.last_move, 60.0) + spec.sensing_lag_s


def admission_level(st: State) -> float | None:
    """The flex admit level main.py must keep renewing at the proxy (None = leave it alone)."""
    return st.flex_admit if st.phase in ACTIVE else None


def restore_on_delete(st: State) -> list[Action]:
    if st.phase in ACTIVE and st.original:
        return [Restore(dict(st.original)), SetAdmission(1.0, "reject")]
    return []


# -- the tick ----------------------------------------------------------------------


def step(spec: Spec, book: LeverBook, model: PowerModel, st: State, obs: Observation) -> tuple[State, list[Action]]:
    if st.phase == DONE:
        return st, []
    if obs.cluster_w is None:
        return replace(st, message="waiting for a cluster power reading"), []
    pool = max(obs.pools, key=lambda p: (p.replicas, p.key)) if obs.pools else None
    dt = obs.now - st.last_tick_at if st.last_tick_at is not None else 0.0
    st = replace(st, measured_w=obs.cluster_w, instant_w=obs.instant_w, last_tick_at=obs.now)

    if st.phase == PENDING:
        if pool is None:
            return replace(st, message="no consenting Deployment in scope; nothing to do"), []
        target = spec.target(obs.cluster_w)
        # A cap is judged on single readings, which scatter more than the 20 s average the planner
        # sees; the margin must cover that scatter. Measured once, at the start, and then frozen
        # (a spread measured mid-event would include the event's own steps and move the aim).
        margin = spec.margin_w
        if spec.mode == "cap" and obs.noise_w is not None:
            margin = max(margin, spec.noise_sigmas * obs.noise_w)
        st = replace(
            st,
            phase=RAMPING,
            started_at=obs.now,
            baseline_w=obs.cluster_w,
            target_w=target,
            plan_target_w=target - margin,
            noise_w=obs.noise_w,
            margin_eff_w=margin,
            original={p.key: p.replicas for p in obs.pools},
            replicas=pool.replicas,
            message=f"baseline {obs.cluster_w:.0f} W; target {target:.0f} W ({spec.mode}), aiming at "
            f"{target - margin:.0f} W (margin {margin:.0f} W)",
        )
    assert st.started_at is not None and st.target_w is not None and st.plan_target_w is not None
    target_w, aim_w, watts = st.target_w, st.plan_target_w, obs.cluster_w
    deadline = st.started_at + spec.ramp_s
    is_settled = settled(st, book, spec, obs.now)

    # Learn the removal spike while a replica move is settling; fold it in once settled.
    # Reference = the 20 s average just before the move (a single reading is too noisy: the first
    # kind run took a high one as the reference and "learned" a 0 W spike). The estimate keeps the
    # largest spike ever seen and never drops below the default: for a hard cap, err high.
    if st.spike_pending and obs.instant_w is not None:
        st = replace(st, post_move_peak_w=max(st.post_move_peak_w or obs.instant_w, obs.instant_w))
        if is_settled and st.pre_move_w is not None and st.post_move_peak_w is not None:
            seen = max(st.post_move_peak_w - st.pre_move_w, 0.0)
            st = replace(
                st,
                spike_w=max(st.spike_w, seen),
                spike_source="measured" if seen >= st.spike_w else st.spike_source,
                spike_pending=False,
                actions=(*st.actions, {"at": obs.now, "move": "spike measured", "spikeWatts": round(seen, 1)}),
            )

    # Feedback: correct the model's level against what was actually measured.
    knobs = Knobs(st.replicas if st.replicas is not None else (pool.replicas if pool else 0), st.flex_admit)
    if is_settled and obs.load is not None and st.phase in (RAMPING, HOLDING):
        raw = replace(model, bias_w=0.0).cluster_w(knobs.replicas, obs.load.admitted(knobs.flex_admit), obs.nodes)
        st = replace(st, bias_w=0.5 * st.bias_w + 0.5 * (watts - raw))

    # -- phase transitions --------------------------------------------------------
    if st.phase == RAMPING:
        if is_settled and watts <= target_w:
            st = replace(st, phase=HOLDING, reached_at=obs.now, ramp_met=obs.now <= deadline,
                         hold_until=obs.now + spec.hold_s)  # fmt: skip
        elif obs.now > deadline:
            st = replace(st, phase=HOLDING, ramp_met=False, hold_until=obs.now + spec.hold_s)

    if st.phase == HOLDING:
        if st.reached_at is None and is_settled and watts <= target_w:
            st = replace(st, reached_at=obs.now)
        if obs.instant_w is not None:
            st = replace(st, peak_hold_w=max(st.peak_hold_w or obs.instant_w, obs.instant_w))
            if obs.instant_w > target_w:
                st = replace(st, above_target_s=st.above_target_s + dt)
        if st.hold_until is not None and obs.now >= st.hold_until:
            st = replace(st, phase=RELEASING, release_started_at=obs.now,
                         message="hold complete: restoring replicas, then readmitting flex")  # fmt: skip
            return st, [Restore(dict(st.original))]

    if st.phase == RELEASING:
        ready = all(p.ready >= st.original.get(p.key, 0) for p in obs.pools)
        timed_out = obs.now - (st.release_started_at or obs.now) >= RELEASE_TIMEOUT_S
        if ready or timed_out:
            note = "replicas Ready" if ready else f"replicas not Ready after {RELEASE_TIMEOUT_S:.0f} s"
            st = replace(st, phase=DONE, flex_admit=1.0, message=f"{note}; flex readmitted; event done")
            return st, [SetAdmission(1.0, spec.flex_action)]
        return replace(st, message="waiting for restored replicas to be Ready"), []

    # -- act (Ramping or Holding) -------------------------------------------------
    if not is_settled:
        return replace(st, message=f"{watts:.0f} W; waiting for the last move ({st.last_move}) to settle"), []
    if watts <= aim_w:
        return replace(st, shortfall_w=0.0, message=f"{watts:.0f} W <= aim {aim_w:.0f} W"), []
    if obs.load is None or pool is None:
        return replace(st, message="no load reading from the admission proxy yet; not acting blind"), []

    # Cap mode: a replica removal must not push a reading over the cap.
    allow_replica = "replica" in book.allowed
    if spec.mode == "cap" and obs.instant_w is not None:
        if st.phase == HOLDING:
            allow_replica &= obs.instant_w + st.spike_w <= target_w
        else:  # ramping: allowed if its transient will be over before the hold can start
            allow_replica &= obs.now + book.speed_s["replica"] + spec.sensing_lag_s <= deadline
    c = Constraints(aim_w, spec.min_replicas, spec.util_limit, spec.capacity_rps, obs.nodes, allow_replica)
    planner = optimise if spec.policy == "optimizer" else ladder
    plan: Plan = planner(knobs, obs.load, model.with_bias(st.bias_w), c)
    if "admission" not in book.allowed and plan.next_move == "admission":
        plan = replace(plan, next_move="none", reason="admission lever not authorised by the contract")
    summary = {
        "replicas": plan.end.replicas,
        "flexAdmit": plan.end.flex_admit,
        "predictedWatts": round(plan.predicted_w, 1),
        "utilisation": round(plan.utilisation, 3),
        "feasible": plan.feasible,
        "model": model.source,
        "reason": plan.reason,
    }
    st = replace(st, plan=summary, shortfall_w=0.0 if plan.feasible else max(plan.predicted_w - aim_w, 0.0))
    nxt = plan.next_knobs(knobs)
    record = {"at": obs.now, "measuredWatts": round(watts, 1), "policy": spec.policy}
    if plan.next_move == "admission":
        st = replace(st, flex_admit=nxt.flex_admit, last_move="admission", last_move_at=obs.now,
                     actions=(*st.actions, {**record, "move": f"flex admit {knobs.flex_admit:g}->{nxt.flex_admit:g}"}),
                     message=plan.reason)  # fmt: skip
        return st, [SetAdmission(nxt.flex_admit, spec.flex_action)]
    if plan.next_move == "replica":
        victim = choose_victim(pool)
        st = replace(
            st,
            replicas=nxt.replicas,
            last_move="replica",
            last_move_at=obs.now,
            pre_move_w=watts,  # the 20 s average, not one reading
            post_move_peak_w=None,
            spike_pending=True,
            actions=(*st.actions, {**record, "move": f"{pool.key} {knobs.replicas}->{nxt.replicas}",
                                   "node": victim.node if victim else None}),
            message=plan.reason,
        )  # fmt: skip
        return st, [ScaleDown(pool.key, nxt.replicas, victim.pod if victim else None, victim.node if victim else None)]
    why = plan.reason
    if not allow_replica and spec.mode == "cap":
        why += f"; replica moves blocked in cap mode (reading + spike {st.spike_w:.0f} W would cross the cap)"
    return replace(st, message=why), []


def make_model(points: Sequence[Sequence[float]] | None, book: LeverBook, capacity_rps: float) -> PowerModel:
    """Calibrated curve if available, else the contract prior."""
    if points:
        return PowerModel.from_points(points, book.parked_w, book.busy_w, source="calibration")
    return PowerModel.contract_prior(book.parked_w, book.busy_w, capacity_rps)
