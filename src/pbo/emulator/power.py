"""Power model of one emulated GPU (pure functions, no I/O).

The emulated nodes have no GPU, so the emulator *computes* a wattage from what
the serving pod reports (how busy it is, at what clock) and publishes it under
the same metric name a real NVIDIA DCGM exporter uses. The operator cannot tell
the difference, which is the point: its control loop is tested end to end.

Model, per GPU:

    P = nominal * ( parked + (busy_role - parked) * busy_frac * clock_ratio**3 )

* ``parked`` -- idle power, as a fraction of nominal (0.17 in the contract: an A100
  sitting idle draws ~43 W of 250 W).
* ``busy_role`` -- power when fully busy, which depends on what the GPU is doing:
  prefill 0.92, decode 0.79, mixed ("mono") 0.84 (contract ``states``).
* ``busy_frac`` -- 0..1, how full the batch was (reported by the serving pod).
* ``clock_ratio**3`` -- dynamic power scales roughly with frequency x voltage^2, and
  voltage tracks frequency, so cubic. This is a textbook approximation (assumed), used
  only so the clock lever has a plausible effect; the contract's measured depth for
  clock locking replaces it once available.

A node with no serving pod (or one that stopped reporting) draws parked power.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pbo.curves import CurveBook

# Fallback numbers = the contract's values at 0.3 rev 2, used when no CurveBook is given (tests).
_DEFAULT_BUSY = {"prefill": 0.92, "decode": 0.79, "mono": 0.84}


@dataclass(frozen=True)
class PowerParams:
    nominal_w: float = 250.0  # emulated A100-PCIe board power
    parked_frac: float = 0.17
    busy_frac_by_role: Mapping[str, float] = field(default_factory=lambda: dict(_DEFAULT_BUSY))
    max_sm_clock_mhz: float = 1410.0  # A100 boost clock; only used for the SM-clock metric

    @property
    def parked_w(self) -> float:
        return self.nominal_w * self.parked_frac

    @classmethod
    def from_curvebook(cls, book: CurveBook, nominal_w: float = 250.0) -> PowerParams:
        """Read parked and busy fractions from the lever-curve contract's ``states``."""

        def frac(state: str, pool: str, fallback: float) -> float:
            n = book.state_frac(state, pool)
            return float(n.value) if n.known and n.value is not None else fallback

        return cls(
            nominal_w=nominal_w,
            parked_frac=frac("parked", "any", 0.17),
            busy_frac_by_role={
                "prefill": frac("busy", "prefill", _DEFAULT_BUSY["prefill"]),
                "decode": frac("busy", "decode", _DEFAULT_BUSY["decode"]),
                # The contract calls the mixed state "busy_monolithic".
                "mono": frac("busy_monolithic", "any", _DEFAULT_BUSY["mono"]),
            },
        )


def gpu_watts(p: PowerParams, role: str | None, busy_frac: float, clock_ratio: float) -> float:
    """Board power in watts. ``role=None`` means no serving pod: parked."""
    if role is None:
        return p.parked_w
    if role not in p.busy_frac_by_role:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(p.busy_frac_by_role)}")
    busy = min(max(busy_frac, 0.0), 1.0)
    clock = min(max(clock_ratio, 0.0), 1.0)
    dynamic = (p.busy_frac_by_role[role] - p.parked_frac) * busy * clock**3
    return p.nominal_w * (p.parked_frac + dynamic)
