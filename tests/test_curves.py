"""Tests for the lever-curve loader.

Each test pins one promise the operator relies on. If a test here fails after
a contract update, the contract changed in a way the operator must review.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pbo.curves import ContractError, Num, load_curves

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "curves" / "lever-curves-v0.3.json"
OVERLAY = ROOT / "curves" / "published-overlay.json"


@pytest.fixture(scope="module")
def book():
    return load_curves(CONTRACT, OVERLAY)


def _write(tmp_path: Path, name: str, doc: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


# --- contract header -------------------------------------------------------


def test_loads_frozen_contract(book):
    assert (book.schema_version, book.revision) == ("0.3", 2)


def test_refuses_other_revision(tmp_path):
    doc = json.loads(CONTRACT.read_text())
    doc["revision"] = 3
    with pytest.raises(ContractError, match="rev"):
        load_curves(_write(tmp_path, "c.json", doc))


# --- the review fixes we depend on (M1, M2) --------------------------------


def test_admission_depth_is_at_stated_mix(book):
    """M1: admission depth is quoted at the entry's own tier mix (~0.21), not 0.62."""
    [entry] = book.find("admission_by_tier", "prefill", {"shed": ["flex"], "mode": "reject"})
    depth = entry.num("depth_frac")
    assert depth.value == pytest.approx(0.21)
    assert depth.status == "estimated"


def test_composition_exclusive_pairs(book):
    """M2: drain excludes everything; power cap excludes clock lock; others combine."""
    assert book.exclusive("drain_to_parked", "admission_by_tier")
    assert book.exclusive("gpu_power_cap", "gpu_clock_lock")
    assert book.exclusive("gpu_clock_lock", "gpu_power_cap")
    assert not book.exclusive("admission_by_tier", "replica_count")


def test_three_tiers_equal_thirds(book):
    assert book.tier_names() == ["premium", "standard", "flex"]
    assert sum(book.default_mix().values()) == pytest.approx(1.0)


# --- overlay rules ---------------------------------------------------------


def test_overlay_adds_published_entry_and_ranks_it_after_contract(book):
    """The 'any'-pool contract entry and the decode-only published one both apply
    to decode; the contract entry must come first."""
    hits = book.find("replica_count", "decode", {"replicas_removed": 1})
    assert [h.origin for h in hits] == ["measured_jade", "published"]
    assert hits[1].from_overlay
    assert hits[1].num("release", "recovery_s") == Num(78.0, "measured", hits[1].num("release", "recovery_s").note)


def test_overlay_never_enters_a_median_as_p99(book):
    published = book.find("replica_count", "decode")[-1]
    assert not published.num("cost", "all", "d_ttft_p99_ms").known


def test_overlay_must_cite(tmp_path):
    ov = json.loads(OVERLAY.read_text())
    del ov["hardware"]["yan2026_h200_sglang"]["citation"]
    with pytest.raises(ContractError, match="citation"):
        load_curves(CONTRACT, _write(tmp_path, "o.json", ov))


def test_overlay_cannot_reuse_contract_hardware(tmp_path):
    ov = json.loads(OVERLAY.read_text())
    ov["hardware"]["sporc_a100_pcie_e01"] = {"origin": "published", "citation": "x"}
    with pytest.raises(ContractError, match="collides"):
        load_curves(CONTRACT, _write(tmp_path, "o.json", ov))


# --- interpolation over load (M4) --------------------------------------------


def _two_load_overlay(tmp_path: Path) -> Path:
    """A synthetic published lever measured at two loads, to test interpolation."""
    ov = json.loads(OVERLAY.read_text())
    base = {
        "name": "test_lever",
        "pool": "decode",
        "authority": "operator",
        "setting": {"k": 1},
        "source_hardware": {"ref": "yan2026_h200_sglang"},
    }
    ov["levers"] = [
        {**base, "at_load": {"pool_busy_frac": 0.4}, "depth_frac": {"value": 0.2, "status": "measured"}},
        {**base, "at_load": {"pool_busy_frac": 0.8}, "depth_frac": {"value": 0.6, "status": "estimated"}},
    ]
    return _write(tmp_path, "o.json", ov)


def test_interpolates_between_loads(tmp_path):
    b = load_curves(CONTRACT, _two_load_overlay(tmp_path))
    mid = b.depth_at("test_lever", "decode", {"k": 1}, 0.6)
    assert mid.value == pytest.approx(0.4)
    assert mid.status == "estimated"  # worse of measured and estimated


def test_clamps_outside_measured_range(tmp_path):
    b = load_curves(CONTRACT, _two_load_overlay(tmp_path))
    assert b.depth_at("test_lever", "decode", {"k": 1}, 0.95).value == pytest.approx(0.6)
    assert b.depth_at("test_lever", "decode", {"k": 1}, 0.10).value == pytest.approx(0.2)


def test_single_point_returned_as_is(book):
    d = book.depth_at("drain_to_parked", "prefill", {"shed": ["premium", "standard", "flex"], "mode": "reject"}, 0.5)
    assert d.value == pytest.approx(0.76)


def test_unknown_lever_is_na_not_error(book):
    assert not book.depth_at("no_such_lever", "decode", {}, 0.5).known
