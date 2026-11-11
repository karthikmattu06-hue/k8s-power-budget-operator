"""Unit tests for the Phase 1 latency and power models."""

from pathlib import Path

import pytest

from pbo.curves import load_curves
from pbo.emulator.power import PowerParams, gpu_watts
from pbo.mockserve.model import (
    BusyTracker,
    ServingParams,
    decode_clock_factor,
    prefill_seconds,
    tpot_seconds,
)

CONTRACT = Path(__file__).resolve().parents[1] / "curves" / "lever-curves-v0.3.json"
P = ServingParams()


# -- latency model -----------------------------------------------------------


def test_prefill_scales_inversely_with_clock() -> None:
    assert prefill_seconds(4000, 1.0, P) == pytest.approx(1.0)
    assert prefill_seconds(4000, 0.5, P) == pytest.approx(2.0)


def test_decode_is_flat_above_knee_and_slows_below() -> None:
    assert decode_clock_factor(1.0, P) == 1.0
    assert decode_clock_factor(0.6, P) == 1.0
    assert decode_clock_factor(0.3, P) == pytest.approx(2.0)


def test_tpot_grows_with_batch_and_caps_at_full() -> None:
    empty, full = tpot_seconds(0, 1.0, P), tpot_seconds(P.max_batch, 1.0, P)
    assert empty == pytest.approx(0.020)
    assert full == pytest.approx(0.030)
    assert tpot_seconds(10 * P.max_batch, 1.0, P) == pytest.approx(full)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_clock_outside_range_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        prefill_seconds(10, bad, P)


def test_busy_tracker_reports_duty_and_fill() -> None:
    t = BusyTracker(max_batch=4, now=0.0)
    t.change(+2, now=0.0)  # 2 of 4 slots for the first half second
    t.change(+2, now=0.5)  # 4 of 4 until 1.0
    t.change(-4, now=1.0)  # idle from 1.0 to 2.0
    occ = t.sample(now=2.0)
    assert occ.duty == pytest.approx(0.5)  # something ran for 1 s of 2 s
    assert occ.fill == pytest.approx(0.375)  # (0.5x2 + 0.5x4) slot-seconds / (2 s x 4 slots)
    assert t.sample(now=3.0) == (0.0, 0.0)  # new window, idle throughout


# -- power model -------------------------------------------------------------


def test_power_states_come_from_contract() -> None:
    p = PowerParams.from_curvebook(load_curves(CONTRACT))
    assert p.parked_frac == pytest.approx(0.17)
    assert p.busy_frac_by_role == {"prefill": 0.92, "decode": 0.79, "mono": 0.84}


def test_power_endpoints() -> None:
    p = PowerParams()  # w = 0.6
    assert gpu_watts(p, None, 1.0, 0.9, 1.0) == pytest.approx(42.5)  # no pod -> parked
    assert gpu_watts(p, "mono", 0.0, 0.0, 1.0) == pytest.approx(42.5)  # idle pod -> parked
    assert gpu_watts(p, "mono", 1.0, 1.0, 1.0) == pytest.approx(210.0)  # contract: ~210 W mixed
    assert gpu_watts(p, "prefill", 1.0, 1.0, 1.0) == pytest.approx(230.0)
    # Running all the time with a near-empty batch already draws 60 % of the dynamic power.
    assert gpu_watts(p, "mono", 1.0, 0.0, 1.0) == pytest.approx(42.5 + 0.6 * 167.5)


def test_consolidation_saves_power_only_when_power_is_concave() -> None:
    """The same load on 4 replicas instead of 5 must draw less -- unless power is linear in fill.

    10 rps on 5 replicas: each busy all the time with fill 0.5. Remove one: the other 4 carry
    the same work at fill 0.625 and the fifth GPU parks.
    """

    def cluster(p: PowerParams, replicas: int, fill: float) -> float:
        return replicas * gpu_watts(p, "mono", 1.0, fill, 1.0) + (5 - replicas) * p.parked_w

    concave, linear = PowerParams(), PowerParams(batch_independent_frac=0.0)
    saved = cluster(concave, 5, 0.5) - cluster(concave, 4, 0.625)
    assert saved == pytest.approx(0.6 * 167.5)  # the parked GPU's batch-independent power
    assert cluster(linear, 5, 0.5) == pytest.approx(cluster(linear, 4, 0.625))  # the old flaw: zero saved


def test_power_drops_with_clock_cubed() -> None:
    p = PowerParams()
    full = gpu_watts(p, "mono", 1.0, 1.0, 1.0) - p.parked_w
    half = gpu_watts(p, "mono", 1.0, 1.0, 0.5) - p.parked_w
    assert half == pytest.approx(full / 8)


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        gpu_watts(PowerParams(), "training", 1.0, 0.5, 1.0)
    with pytest.raises(ValueError):
        PowerParams(batch_independent_frac=1.5)
