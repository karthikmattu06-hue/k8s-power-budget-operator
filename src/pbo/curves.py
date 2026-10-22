"""Load the lever-curve contract (schema 0.3 rev 2) and answer the questions the
operator needs answered: "what levers does this pool have, how many watts does
each shed, how fast, and what does it cost each tier?"

Why this module exists
----------------------
The operator never measures GPUs itself. It reads a JSON *datasheet* of levers.
Each entry describes one lever (e.g. "reject flex-tier requests on the prefill
pool") with five rows:

    depth_frac  how much power it sheds, as a fraction of one GPU's nominal power
    speed_s     how long until 90 % of that drop has happened
    duration_s  how long it can be held before a served tier breaks its SLO
    cost        latency added (or work lost) per tier while it is held
    release     what happens when it is undone (recovery time, rebound)

Two files feed this module:

1. ``curves/lever-curves-v0.3.json`` -- the *contract*, produced by a companion
   measurement project and copied here unchanged. Its structure is frozen at
   schema 0.3, revision 2. We refuse to load anything else, so a structural
   change on the producer side can never silently change what the operator does.
2. ``curves/published-overlay.json`` -- *our* file. It adds entries taken from
   published papers (origin = "published", with a citation) so the simulator
   has numbers before the contract's measured entries exist. The overlay may
   only *add* entries; it can never edit or replace a contract entry.

Every number in both files is wrapped as ``{"value", "status", "note"}`` where
status is one of measured / estimated / assumed / n/a. We keep the status with
the number all the way to the report, so an estimate is never quoted as data.

Only the Python standard library is used here, so this module can be imported
by the operator, the mock serving pod, the emulator and the tests alike.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

# The one contract version this code understands. Bump only after reviewing
# the producer's change (see docs/adr/0003-lever-curve-contract.md).
EXPECTED_SCHEMA_VERSION = "0.3"
EXPECTED_REVISION = 2

VALID_STATUS = frozenset({"measured", "estimated", "assumed", "n/a"})

# When several entries describe the same lever on the same pool, prefer them in
# this order. Lower rank wins. "measured_jade" is the producer's origin label.
_ORIGIN_RANK = {"measured_jade": 0, "published": 1}
_STATUS_RANK = {"measured": 0, "estimated": 1, "assumed": 2, "n/a": 3}


class ContractError(ValueError):
    """The contract or overlay file is not something we are allowed to load."""


# ---------------------------------------------------------------------------
# Small value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Num:
    """One number from the datasheet, carried together with its honesty label.

    ``value`` is None when the field is empty (status "n/a"). Code that does
    arithmetic should check ``known`` first instead of assuming a float.
    """

    value: float | None
    status: str
    note: str = ""

    @property
    def known(self) -> bool:
        """True when there is a usable number (anything except n/a / None)."""
        return self.value is not None and self.status != "n/a"

    @staticmethod
    def from_json(obj: Any) -> Num:
        """Turn ``{"value": .., "status": .., "note": ..}`` into a Num.

        A missing field becomes Num(None, "n/a") rather than an exception,
        because many entries are legitimately still empty.
        """
        if obj is None:
            return Num(None, "n/a")
        if not isinstance(obj, Mapping) or "status" not in obj:
            raise ContractError(f"expected a {{value, status}} object, got {obj!r}")
        status = obj["status"]
        if status not in VALID_STATUS:
            raise ContractError(f"unknown status {status!r} (allowed: {sorted(VALID_STATUS)})")
        raw = obj.get("value")
        value = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        return Num(value, status, str(obj.get("note", "")))


def _worse_status(a: str, b: str) -> str:
    """When two numbers are combined, the result is only as good as the worse one."""
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


# ---------------------------------------------------------------------------
# One lever entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeverEntry:
    """One row-set of the datasheet: a lever, on a pool, at one setting and load.

    ``raw`` keeps the original JSON so fields we do not model yet are not lost.
    Use :meth:`num` to read any numeric field by path, e.g.
    ``entry.num("release", "recovery_s")``.
    """

    name: str
    pool: str  # "prefill" | "decode" | "any"
    authority: str  # "operator" | "root"
    setting: Mapping[str, Any]
    at_load: Mapping[str, Any]
    source_ref: str | None
    origin: str  # from the hardware table: "measured_jade" | "published" | "unknown"
    from_overlay: bool
    raw: Mapping[str, Any] = field(repr=False)

    def num(self, *path: str) -> Num:
        """Read a numeric field by path. Missing anywhere along the path -> n/a."""
        node: Any = self.raw
        for part in path:
            if not isinstance(node, Mapping) or part not in node:
                return Num(None, "n/a")
            node = node[part]
        return Num.from_json(node)

    @property
    def busy_frac(self) -> float | None:
        """The pool load this entry was measured at (the interpolation key, M4)."""
        v = self.at_load.get("pool_busy_frac")
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    @property
    def setting_key(self) -> str:
        """A stable string for the setting, so equal settings compare equal."""
        return json.dumps(self.setting, sort_keys=True)

    def applies_to(self, pool: str) -> bool:
        """A lever listed for pool "any" applies to prefill and decode alike."""
        return self.pool in (pool, "any")

    def _rank(self) -> tuple[int, int]:
        """Sort key: contract-measured first, then contract-estimated, then published."""
        origin_rank = _ORIGIN_RANK.get(self.origin, 2)
        return (origin_rank, _STATUS_RANK[self.num("depth_frac").status])


# ---------------------------------------------------------------------------
# The whole book of curves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurveBook:
    """Everything the operator reads from the datasheet, after validation."""

    schema_version: str
    revision: int
    tiers: Mapping[str, Any]
    hardware: Mapping[str, Mapping[str, Any]]
    composition: Mapping[str, Any]
    entries: tuple[LeverEntry, ...]

    # -- lookup -------------------------------------------------------------

    def find(self, name: str, pool: str, setting: Mapping[str, Any] | None = None) -> list[LeverEntry]:
        """All entries for a lever on a pool, best source first.

        ``setting=None`` returns every setting; otherwise only exact matches.
        """
        key = None if setting is None else json.dumps(setting, sort_keys=True)
        hits = [
            e for e in self.entries if e.name == name and e.applies_to(pool) and (key is None or e.setting_key == key)
        ]
        return sorted(hits, key=LeverEntry._rank)

    def depth_at(self, name: str, pool: str, setting: Mapping[str, Any], busy_frac: float) -> Num:
        """Depth of a lever at the pool's *current* load (schema M4).

        Rule, in order:
        * take the best-ranked source that has a known depth;
        * among its entries, if two or more carry ``pool_busy_frac``, interpolate
          linearly between the two loads that bracket ``busy_frac`` (clamping
          outside the measured range -- we never extrapolate);
        * otherwise return the single point as-is.
        The status of the answer is the worse of the points used.
        """
        candidates = [e for e in self.find(name, pool, setting) if e.num("depth_frac").known]
        if not candidates:
            return Num(None, "n/a", f"no known depth for {name}/{pool}/{dict(setting)}")
        best_rank = candidates[0]._rank()[0]
        same_source = [e for e in candidates if e._rank()[0] == best_rank]
        loaded = sorted((e for e in same_source if e.busy_frac is not None), key=lambda e: e.busy_frac or 0.0)
        if len(loaded) < 2:
            return candidates[0].num("depth_frac")
        return _interpolate(loaded, busy_frac)

    # -- composition (schema M2) ---------------------------------------------

    def exclusive(self, lever_a: str, lever_b: str) -> bool:
        """True if the two levers may not be used together on one GPU.

        The contract lists exclusive pairs; "*" means "with anything".
        """
        for pair in self.composition.get("exclusive_on_one_gpu", []):
            x, y = pair
            for p, q in ((x, y), (y, x)):
                if p == lever_a and q in ("*", lever_b):
                    return True
        return False

    def tier_names(self) -> list[str]:
        """Tier names in the contract, e.g. ['premium', 'standard', 'flex']."""
        return [k for k, v in self.tiers.items() if isinstance(v, Mapping) and k != "default_mix"]

    def default_mix(self) -> dict[str, float]:
        """Default share of requests per tier (equal thirds in rev 2)."""
        mix = self.tiers.get("default_mix", {})
        return {k: float(v) for k, v in mix.items() if isinstance(v, (int, float))}


def _interpolate(points: list[LeverEntry], x: float) -> Num:
    """Piecewise-linear depth over pool load, clamped at both ends."""
    xs = [p.busy_frac for p in points]
    assert all(v is not None for v in xs)
    lo, hi = points[0], points[-1]
    if x <= (lo.busy_frac or 0.0):
        n = lo.num("depth_frac")
        return Num(n.value, n.status, f"clamped to lowest measured load {lo.busy_frac}")
    if x >= (hi.busy_frac or 0.0):
        n = hi.num("depth_frac")
        return Num(n.value, n.status, f"clamped to highest measured load {hi.busy_frac}")
    for left, right in pairwise(points):
        x0, x1 = left.busy_frac or 0.0, right.busy_frac or 0.0
        if x0 <= x <= x1:
            y0, y1 = left.num("depth_frac"), right.num("depth_frac")
            assert y0.value is not None and y1.value is not None
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return Num(y0.value + t * (y1.value - y0.value), _worse_status(y0.status, y1.status), "interpolated")
    raise AssertionError("unreachable: x is inside the measured range")


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def _check_contract_header(doc: Mapping[str, Any], where: str) -> None:
    """Refuse anything that is not exactly schema 0.3 revision 2."""
    version, revision = doc.get("schema_version"), doc.get("revision")
    if version != EXPECTED_SCHEMA_VERSION or revision != EXPECTED_REVISION:
        raise ContractError(
            f"{where}: expected schema {EXPECTED_SCHEMA_VERSION} rev {EXPECTED_REVISION}, "
            f"got {version!r} rev {revision!r}. Review the change before bumping the constants."
        )


def _parse_entries(
    levers: Iterable[Mapping[str, Any]], hardware: Mapping[str, Mapping[str, Any]], from_overlay: bool
) -> list[LeverEntry]:
    out = []
    for raw in levers:
        ref = (raw.get("source_hardware") or {}).get("ref")
        origin = str(hardware.get(ref, {}).get("origin", "unknown")) if ref else "unknown"
        entry = LeverEntry(
            name=str(raw["name"]),
            pool=str(raw["pool"]),
            authority=str(raw.get("authority", "operator")),
            setting=dict(raw.get("setting") or {}),
            at_load=dict(raw.get("at_load") or {}),
            source_ref=ref,
            origin=origin,
            from_overlay=from_overlay,
            raw=raw,
        )
        entry.num("depth_frac")  # validates the status label early
        out.append(entry)
    return out


def _check_overlay(overlay: Mapping[str, Any], contract_hw: Mapping[str, Any]) -> None:
    """The overlay may only add published entries, each traceable to a paper."""
    target = overlay.get("overlay_for", {})
    if target.get("schema_version") != EXPECTED_SCHEMA_VERSION or target.get("revision") != EXPECTED_REVISION:
        raise ContractError(f"overlay targets {target!r}, not schema {EXPECTED_SCHEMA_VERSION} rev {EXPECTED_REVISION}")
    hw = overlay.get("hardware", {})
    for key, rec in hw.items():
        if key in contract_hw:
            raise ContractError(f"overlay hardware key {key!r} collides with the contract")
        if rec.get("origin") != "published" or not rec.get("citation"):
            raise ContractError(f"overlay hardware {key!r} must have origin='published' and a citation")
    for lever in overlay.get("levers", []):
        ref = (lever.get("source_hardware") or {}).get("ref")
        if ref not in hw:
            raise ContractError(f"overlay lever {lever.get('name')!r} must point at overlay hardware, got {ref!r}")


def load_curves(contract_path: str | Path, overlay_path: str | Path | None = None) -> CurveBook:
    """Load the frozen contract, optionally add our published overlay, validate both."""
    contract = json.loads(Path(contract_path).read_text())
    _check_contract_header(contract, str(contract_path))
    hardware: dict[str, Mapping[str, Any]] = dict(contract.get("hardware", {}))
    entries = _parse_entries(contract.get("levers", []), hardware, from_overlay=False)

    if overlay_path is not None:
        overlay = json.loads(Path(overlay_path).read_text())
        _check_overlay(overlay, hardware)
        hardware.update(overlay.get("hardware", {}))
        entries += _parse_entries(overlay.get("levers", []), hardware, from_overlay=True)

    return CurveBook(
        schema_version=contract["schema_version"],
        revision=int(contract["revision"]),
        tiers=contract.get("tiers", {}),
        hardware=hardware,
        composition=contract.get("composition", {}),
        entries=tuple(entries),
    )
