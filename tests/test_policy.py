"""The event state machine (pbo.operator.policy) with a fake clock."""

from __future__ import annotations

import pytest

from pbo.operator.levers import default_book
from pbo.operator.model import PowerModel
from pbo.operator.planner import Load
from pbo.operator.policy import (
    DONE,
    HOLDING,
    PENDING,
    RAMPING,
    RELEASING,
    Observation,
    Pool,
    Replica,
    Restore,
    ScaleDown,
    SetAdmission,
    Spec,
    State,
    admission_level,
    restore_on_delete,
    step,
)

BOOK = default_book()  # admission settles in 10 + 30 s, replicas in 60 + 30 s
MODEL = PowerModel.from_points(
    [(0.4, 150), (0.8, 165), (1.2, 172), (1.6, 178), (2.0, 185), (2.6, 195)], parked_w=42.5, max_w=210
)
LOAD = Load(5.33, 2.67)
SPEC = Spec(reduction_watts=100, margin_w=25, ramp_s=600, hold_s=300, min_replicas=1)
CAP = Spec(reduction_watts=100, margin_w=25, ramp_s=600, hold_s=300, mode="cap")


def pool(n: int, ready: int | None = None) -> Pool:
    pods = tuple(Replica(f"p{i}", f"worker{i}", 0.5 - 0.1 * i) for i in range(n))
    return Pool("pbo/mock-llm", n, n if ready is None else ready, pods)


def obs(t: float, w: float | None, n: int = 5, inst: float | None = None, ready: int | None = None) -> Observation:
    return Observation(t, w, inst if inst is not None else w, 5, LOAD, (pool(n, ready),))


def run(spec: Spec, st: State, o: Observation) -> tuple[State, list]:  # type: ignore[type-arg]
    return step(spec, BOOK, MODEL, st, o)


# -- spec ---------------------------------------------------------------------------


def test_spec_validation() -> None:
    assert Spec.from_k8s({"reduction": {"fraction": 0.1}}).target(800) == pytest.approx(720)
    for bad in (
        {"reduction": {"watts": 1, "fraction": 0.1}},
        {"reduction": {"watts": 1}, "mode": "soft"},
        {"reduction": {"watts": 1}, "policy": "random"},
        {"reduction": {"watts": 1}, "flexAction": "defer", "rampSeconds": 300, "holdSeconds": 900},  # > 600 s
    ):
        with pytest.raises(ValueError):
            Spec.from_k8s(bad)


def test_status_round_trips() -> None:
    st = State(phase=HOLDING, flex_admit=0.5, original={"pbo/mock-llm": 5}, spike_pending=True)
    back = State.from_k8s(st.to_k8s())
    assert back.flex_admit == 0.5 and back.original == {"pbo/mock-llm": 5} and back.spike_pending
    assert State.from_k8s(None).phase == PENDING


# -- ramp ---------------------------------------------------------------------------


def test_first_tick_sets_target_and_sheds_flex_first() -> None:
    st, acts = run(SPEC, State(), obs(0, 890))
    assert st.phase == RAMPING and st.target_w == 790 and st.plan_target_w == 765
    assert acts == [SetAdmission(0.5, "reject")]  # optimiser: 4 replicas + half flex; admission first
    assert st.plan["replicas"] == 4 and admission_level(st) == 0.5


def test_waits_until_settled_then_removes_the_emptiest_replica() -> None:
    st, _ = run(SPEC, State(), obs(0, 890))
    st, acts = run(SPEC, st, obs(20, 860))  # admission settles at 10 + 30 = 40 s
    assert acts == [] and "settle" in st.message
    st, acts = run(SPEC, st, obs(40, 860))
    assert acts == [ScaleDown("pbo/mock-llm", 4, "p4", "worker4")]  # p4 has the lowest fill
    assert st.spike_pending and st.pre_move_w == 860


def test_no_victory_on_an_unsettled_dip() -> None:
    st, _ = run(SPEC, State(), obs(0, 890))
    st, _ = run(SPEC, st, obs(5, 700))  # a transient dip before the admission change has settled
    assert st.phase == RAMPING
    st, _ = run(SPEC, st, obs(45, 760))  # settled and at/below target (790)
    assert st.phase == HOLDING and st.ramp_met


def test_spike_is_learned_against_the_average_not_one_reading() -> None:
    """Regression (first cap run): a high single reading at the move hid the spike (learned 0 W)."""
    st, _ = run(SPEC, State(), obs(0, 890))
    st, _ = run(SPEC, st, obs(40, 800, inst=900))  # replica removed; this one reading is high
    assert st.pre_move_w == 800  # reference is the 20 s average
    st, _ = run(SPEC, st, obs(60, 840, n=4, inst=880))  # drain transient
    st, _ = run(SPEC, st, obs(130, 760, n=4))  # settled (40 + 60 + 30)
    assert st.spike_w == pytest.approx(80) and st.spike_source == "measured" and not st.spike_pending
    assert any(a.get("move") == "spike measured" for a in st.actions)


def test_spike_estimate_never_drops_below_default_or_previous() -> None:
    st, _ = run(SPEC, State(), obs(0, 890))
    st, _ = run(SPEC, st, obs(40, 860))
    st, _ = run(SPEC, st, obs(60, 870, n=4, inst=880))  # only +20 W seen
    st, _ = run(SPEC, st, obs(130, 760, n=4))
    assert st.spike_w == 50 and st.spike_source == "default"  # conservative for a cap


def test_cap_margin_covers_reading_noise() -> None:
    o = Observation(0, 890, 890, 5, LOAD, (pool(5),), noise_w=20)
    st, _ = run(CAP, State(), o)
    assert st.margin_eff_w == pytest.approx(70) and st.plan_target_w == pytest.approx(790 - 70)  # 3.5 x 20
    st, _ = run(SPEC, State(), o)  # interval mode keeps the plain margin
    assert st.margin_eff_w == 25


def test_feedback_bias_tracks_model_error() -> None:
    st, _ = run(SPEC, State(), obs(0, 890))
    model_now = MODEL.cluster_w(5, 8.0, 5)  # 890 predicted at the first tick
    assert st.bias_w == pytest.approx(0.5 * (890 - model_now))


# -- cap mode -----------------------------------------------------------------------


def test_cap_mode_blocks_spiky_moves_in_hold() -> None:
    st = State(phase=HOLDING, started_at=0, target_w=790, plan_target_w=765, baseline_w=890,
               hold_until=900, flex_admit=1.0, replicas=5, original={"pbo/mock-llm": 5})  # fmt: skip
    st, acts = run(CAP, st, obs(300, 780, inst=780))  # 780 + 50 W spike estimate > 790 cap
    assert all(not isinstance(a, ScaleDown) for a in acts)
    assert acts and isinstance(acts[0], SetAdmission)  # admission is spike-free: still allowed


def test_cap_mode_counts_time_above_the_cap() -> None:
    st = State(phase=HOLDING, started_at=0, target_w=790, plan_target_w=765, baseline_w=890, hold_until=900,
               flex_admit=0.0, replicas=4, original={"pbo/mock-llm": 5}, last_tick_at=300)  # fmt: skip
    st, _ = run(CAP, st, obs(305, 760, n=4, inst=800))
    assert st.above_target_s == pytest.approx(5) and st.peak_hold_w == 800


# -- release ------------------------------------------------------------------------


def test_release_restores_replicas_before_readmitting_flex() -> None:
    st = State(phase=HOLDING, started_at=0, target_w=790, plan_target_w=765, baseline_w=890, hold_until=400,
               flex_admit=0.5, replicas=4, original={"pbo/mock-llm": 5}, reached_at=50)  # fmt: skip
    st, acts = run(SPEC, st, obs(400, 760, n=4))
    assert st.phase == RELEASING and acts == [Restore({"pbo/mock-llm": 5})]
    assert admission_level(st) == 0.5  # flex still shed while replicas come back
    st, acts = run(SPEC, st, obs(410, 800, n=5, ready=4))
    assert acts == [] and st.phase == RELEASING
    st, acts = run(SPEC, st, obs(420, 820, n=5, ready=5))
    assert st.phase == DONE and acts == [SetAdmission(1.0, "reject")]


def test_delete_mid_event_restores_everything() -> None:
    st, _ = run(SPEC, State(), obs(0, 890))
    assert restore_on_delete(st) == [Restore({"pbo/mock-llm": 5}), SetAdmission(1.0, "reject")]
    assert restore_on_delete(State()) == []


# -- closed loop on a toy cluster -----------------------------------------------------


def test_loop_reaches_and_holds_on_a_toy_cluster() -> None:
    """Plant = the same curve plus a +20 W error the operator does not know about."""
    st, replicas, admit, t = State(), 5, 1.0, 0.0
    while st.phase not in (HOLDING, DONE) and t < 600:
        w = MODEL.cluster_w(replicas, LOAD.admitted(admit), 5) + 20
        st, acts = run(SPEC, st, obs(t, w, n=replicas))
        for a in acts:
            if isinstance(a, SetAdmission):
                admit = a.flex_admit
            elif isinstance(a, ScaleDown):
                replicas = a.to_replicas
        t += 5
    assert st.phase == HOLDING and st.ramp_met
    assert MODEL.cluster_w(replicas, LOAD.admitted(admit), 5) + 20 <= st.target_w
    assert admit > 0  # did not throw away all flex to get there
