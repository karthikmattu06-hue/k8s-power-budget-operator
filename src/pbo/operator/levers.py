"""Adapter between the lever-curve contract and the optimiser.

The optimiser never reads the contract JSON. It reads a :class:`LeverBook`, built here from a
:class:`pbo.curves.CurveBook`. When the contract changes (new measurements, new pools), only
the loader and this file are touched.

Every number carries its provenance (``contract:measured``, ``contract:estimated``,
``contract:borrowed(...)`` or ``default``) and the event status lists them, so a report can say
which decisions rested on measurements and which on guesses.

What the contract provides today (schema 0.3 rev 2, 19 Oct copy) and where it goes:

=====================  ==========================================================
contract field         used for
=====================  ==========================================================
states (parked/busy)   power-model prior and parked watts of a GPU without a replica
hardware.nominal_w     scales those fractions to watts
lever speed_s          settle time: how long after a move before judging its effect
lever duration_s       upper bound on holding a lever (defer: 600 s)
lever authority        only ``operator`` levers are used; ``root`` (power cap, clock) need opt-in
composition            "one lever per pool per event": recorded; see ``combined_levers_note``
tiers                  SLO thresholds reported in the event status
=====================  ==========================================================

Gaps, kept here so they are not forgotten (raise with the producer when its data settles):
* state rows carry their load only in the note text, not in an ``at_load`` field;
* combined depth of two levers on one pool is not measured ("one lever per pool per event"),
  while the optimiser combines admission and replicas -- we rely on measured feedback for that;
* admission entries exist for the prefill pool only; a monolithic pool borrows them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pbo.curves import CurveBook, Num

DEFAULT_SPEED_S = {"admission": 10.0, "replica": 60.0, "clock": 10.0}
DEFAULT_PARKED_FRAC, DEFAULT_BUSY_FRAC, DEFAULT_NOMINAL_W = 0.17, 0.84, 250.0
DEFAULT_DEFER_MAX_S = 600.0


@dataclass(frozen=True)
class LeverBook:
    nominal_w: float
    parked_frac: float
    busy_frac: float  # mixed (monolithic) traffic
    speed_s: dict[str, float]  # lever -> seconds until its effect is complete
    defer_max_s: float
    allowed: frozenset[str]  # levers the operator may use on its own authority
    slo: dict[str, float]  # e.g. {"premium_ttft_p99_s": 2.0, ...}
    sources: dict[str, str] = field(default_factory=dict)
    root_levers: frozenset[str] = frozenset()  # need root (hardware owner) and spec.allowRoot
    combined_levers_note: str = ""

    @property
    def parked_w(self) -> float:
        return self.nominal_w * self.parked_frac

    @property
    def busy_w(self) -> float:
        return self.nominal_w * self.busy_frac


def _label(n: Num, borrowed: str = "") -> str:
    return f"contract:{n.status}" + (f" (borrowed from {borrowed})" if borrowed else "")


def default_book() -> LeverBook:
    """Used when no contract is available (unit tests, or a broken contract file)."""
    return LeverBook(
        nominal_w=DEFAULT_NOMINAL_W,
        parked_frac=DEFAULT_PARKED_FRAC,
        busy_frac=DEFAULT_BUSY_FRAC,
        speed_s=dict(DEFAULT_SPEED_S),
        defer_max_s=DEFAULT_DEFER_MAX_S,
        allowed=frozenset({"admission", "replica"}),
        root_levers=frozenset({"clock"}),
        slo={"premium_ttft_p99_s": 2.0, "standard_ttft_p99_s": 6.0, "tpot_p99_s": 0.05, "flex_ttlt_s": 600.0},
        sources={"all": "default"},
    )


def from_contract(book: CurveBook, pool: str = "mono") -> LeverBook:
    src: dict[str, str] = {}

    # -- power states -> watts
    parked = book.state_frac("parked", "any")
    busy = book.state_frac("busy_monolithic", "any") if pool == "mono" else book.state_frac("busy", pool)
    parked_frac = float(parked.value) if parked.known and parked.value is not None else DEFAULT_PARKED_FRAC
    busy_frac = float(busy.value) if busy.known and busy.value is not None else DEFAULT_BUSY_FRAC
    src["parked_frac"] = _label(parked) if parked.known else "default"
    src["busy_frac"] = _label(busy) if busy.known else "default"
    refs = [s.get("source_hardware", {}).get("ref") for s in book.states if s.get("state") == "parked"]
    hw = book.hardware.get(refs[0]) if refs and refs[0] else None
    nominal = float(hw["nominal_w"]) if hw and hw.get("nominal_w") else DEFAULT_NOMINAL_W
    src["nominal_w"] = f"contract hardware {refs[0]}" if hw else "default"

    # -- speeds (settle times). Admission has prefill-pool entries only; a mono pool borrows them.
    speed = dict(DEFAULT_SPEED_S)
    adm_pool = pool if book.find("admission_by_tier", pool) else "prefill"
    for lever, (name, p) in {"admission": ("admission_by_tier", adm_pool), "replica": ("replica_count", pool)}.items():
        entries = [e for e in book.find(name, p) if e.num("speed_s").known]
        if entries:
            n = entries[0].num("speed_s")
            speed[lever] = float(n.value or speed[lever])
            src[f"speed_{lever}"] = _label(n, borrowed=p if p != pool else "")
        else:
            src[f"speed_{lever}"] = "default"

    # -- authority: only levers the operator may use on its own
    allowed = set()
    for lever, name in {"admission": "admission_by_tier", "replica": "replica_count"}.items():
        entries = book.find(name, adm_pool if lever == "admission" else pool)
        if entries and all(e.authority == "operator" for e in entries):
            allowed.add(lever)
    src["authority"] = "contract"
    # Clock locking is a root lever in the contract (it changes the hardware, not the workload).
    # Its depth and speed are n/a there; the emulator's clock^3 model stands in (assumed).
    root = set()
    clock = book.find("gpu_clock_lock", "decode") or book.find("gpu_clock_lock", "prefill")
    if clock and all(e.authority == "root" for e in clock):
        root.add("clock")
    src["speed_clock"] = "default (contract n/a)"

    # -- SLOs from the tier table (assumed values in rev 2), in seconds
    slo = default_book().slo
    for tier, field_name, key in (
        ("premium", "ttft_p99_slo_ms", "premium_ttft_p99_s"),
        ("standard", "ttft_p99_slo_ms", "standard_ttft_p99_s"),
        ("premium", "tpot_p99_slo_ms", "tpot_p99_s"),
        ("flex", "ttlt_slo_ms", "flex_ttlt_s"),
    ):
        t = book.tiers.get(tier)
        val = t.get(field_name) if isinstance(t, dict) else None
        if isinstance(val, dict) and val.get("value") is not None:
            slo[key] = float(val["value"]) / 1000.0
    src["slo"] = "contract:assumed"

    note = str(book.composition.get("rule", "")) if book.composition else ""
    return LeverBook(
        nominal_w=nominal,
        parked_frac=parked_frac,
        busy_frac=busy_frac,
        speed_s=speed,
        defer_max_s=DEFAULT_DEFER_MAX_S,
        allowed=frozenset(allowed),
        root_levers=frozenset(root),
        slo=slo,
        sources=src,
        combined_levers_note=(
            f"contract composition: '{note}'. The optimiser combines admission and replicas on one pool; "
            "their combined effect is not in the contract, so it is checked against measured watts."
        ),
    )
