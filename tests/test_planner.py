"""Power model, optimiser, ladder baseline and contract adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from pbo.curves import load_curves
from pbo.operator.levers import default_book, from_contract
from pbo.operator.model import PowerModel
from pbo.operator.planner import Constraints, Knobs, Load, ladder, optimise, utilisation

CONTRACT = Path(__file__).resolve().parents[1] / "curves" / "lever-curves-v0.3.json"

# A concave node curve like the one the emulator produces: most power is drawn as soon as a
# replica is busy at all; more load adds a little.
POINTS = [(0.4, 150), (0.8, 165), (1.2, 172), (1.6, 178), (2.0, 185), (2.6, 195)]
MODEL = PowerModel.from_points(POINTS, parked_w=42.5, max_w=210)
LOAD = Load(protected_rps=5.33, flex_rps=2.67)  # 8 rps, equal thirds
NODES = 5


def cons(target: float, allow: bool = True, min_replicas: int = 1) -> Constraints:
    return Constraints(target, min_replicas, 0.6, 3.3, NODES, allow)


# -- model ----------------------------------------------------------------------------


def test_model_interpolates_extrapolates_and_parks() -> None:
    assert MODEL.node_w(0) == 42.5  # idle replica ~ parked (point added automatically)
    assert MODEL.node_w(1.0) == pytest.approx(168.5)
    assert MODEL.node_w(5.0) == 210  # extrapolation clamped at max
    assert MODEL.cluster_w(4, 8.0, NODES) == pytest.approx(4 * 185 + 42.5)
    assert MODEL.with_bias(-10).cluster_w(4, 8.0, NODES) == pytest.approx(4 * 185 + 42.5 - 10)


def test_contract_prior_is_a_straight_line() -> None:
    m = PowerModel.contract_prior(42.5, 210, 3.3)
    assert m.node_w(1.65) == pytest.approx((42.5 + 210) / 2)
    assert m.source == "contract prior"


# -- optimiser ------------------------------------------------------------------------


def test_optimiser_finds_the_cheap_mix() -> None:
    """Baseline 5 replicas = 890 W. Aim 765 W. Cheapest: 4 replicas + half the flex."""
    plan = optimise(Knobs(5, 1.0), LOAD, MODEL, cons(765))
    assert plan.feasible and plan.end == Knobs(4, 0.5)
    assert plan.next_move == "admission"  # admission first: spike-free and lowers utilisation
    assert plan.predicted_w <= 765 and plan.utilisation <= 0.6


def test_optimiser_never_breaks_premium_to_save_flex() -> None:
    """3 replicas with all flex would meet power but sit at 81 % utilisation: not allowed."""
    plan = optimise(Knobs(5, 1.0), LOAD, MODEL, cons(700))
    assert plan.feasible
    assert utilisation(plan.end, LOAD, 3.3) <= 0.6
    assert plan.end.replicas == 3 and plan.end.flex_admit == 0.0


def test_infeasible_target_gives_deepest_safe_cut() -> None:
    plan = optimise(Knobs(5, 1.0), LOAD, MODEL, cons(300))
    assert not plan.feasible and "no plan" in plan.reason
    assert utilisation(plan.end, LOAD, 3.3) <= 0.6


def test_cap_mode_without_replica_moves_uses_admission_only() -> None:
    plan = optimise(Knobs(5, 1.0), LOAD, MODEL, cons(860, allow=False))
    assert plan.end.replicas == 5 and plan.end.flex_admit < 1.0


def test_knobs_never_move_up_during_an_event() -> None:
    plan = optimise(Knobs(4, 0.25), LOAD, MODEL, cons(500))
    assert plan.end.replicas <= 4 and plan.end.flex_admit <= 0.25


# -- ladder baseline --------------------------------------------------------------------


def test_ladder_sheds_all_flex_then_replicas() -> None:
    p1 = ladder(Knobs(5, 1.0), LOAD, MODEL, cons(765))
    assert p1.next_move == "admission" and p1.end.flex_admit == 0.0
    p2 = ladder(Knobs(5, 0.0), LOAD, MODEL, cons(765))
    assert p2.next_move == "replica" and p2.end.replicas == 4


# -- contract adapter -------------------------------------------------------------------


def test_contract_feeds_speeds_power_and_slos() -> None:
    b = from_contract(load_curves(CONTRACT))
    assert b.speed_s == {"admission": 8.0, "replica": 90.0}
    assert b.parked_w == pytest.approx(42.5) and b.busy_w == pytest.approx(210)
    assert b.slo["premium_ttft_p99_s"] == 2.0 and b.slo["flex_ttlt_s"] == 600.0
    assert b.allowed == {"admission", "replica"}
    assert "borrowed from prefill" in b.sources["speed_admission"]  # labelled, not hidden
    assert "combined" in b.combined_levers_note


def test_default_book_is_labelled() -> None:
    assert default_book().sources == {"all": "default"}
