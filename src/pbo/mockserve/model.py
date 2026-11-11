"""Latency model of an LLM serving replica (pure functions, no I/O).

The mock serving pod uses this to decide how long to "think" before it answers,
so that latency responds to the same things a real vLLM replica responds to:

* **Prefill** (reading the prompt) is compute-bound: its time grows with prompt
  length and shrinks with GPU clock. Halve the clock, double the prefill time.
* **Decode** (writing tokens one by one) is memory-bandwidth-bound: lowering the
  clock barely matters until it falls below a "knee" (~0.6 of max, per PowerSlider),
  after which it slows in proportion. Decode also slows a little as the batch fills,
  because every running request shares each forward pass.
* **Batch slots**: a replica runs at most ``max_batch`` requests at once; the rest
  wait in a queue. Waiting is where TTFT blows up under overload.

All numbers are illustrative defaults for a mid-size model on one data-centre GPU.
They are not measurements; the point of the emulated cluster is the *shape* of the
response (queueing, clock sensitivity), which the operator's control loop reacts to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

ROLES = ("prefill", "decode", "mono")


@dataclass(frozen=True)
class ServingParams:
    """Knobs of the latency model. Defaults give ~4 s per request at half load."""

    max_batch: int = 16  # concurrent requests per replica (vLLM's max_num_seqs, scaled down)
    prefill_tok_per_s: float = 4000.0  # prompt tokens per second at full clock
    tpot_base_s: float = 0.020  # time per output token, empty batch, full clock
    tpot_batch_slope: float = 0.5  # +50 % TPOT when the batch is full
    decode_clock_knee: float = 0.6  # decode is clock-insensitive above this ratio

    def __post_init__(self) -> None:
        if self.max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        if not 0.0 < self.decode_clock_knee <= 1.0:
            raise ValueError("decode_clock_knee must be in (0, 1]")


def _check_clock(clock_ratio: float) -> None:
    # clock_ratio is current SM clock / max SM clock. 0 would mean "infinitely slow".
    if not 0.0 < clock_ratio <= 1.0:
        raise ValueError(f"clock_ratio must be in (0, 1], got {clock_ratio}")


def prefill_seconds(prompt_tokens: int, clock_ratio: float, p: ServingParams) -> float:
    """Time to process the prompt. Compute-bound, so inversely proportional to clock."""
    _check_clock(clock_ratio)
    return max(prompt_tokens, 0) / (p.prefill_tok_per_s * clock_ratio)


def decode_clock_factor(clock_ratio: float, p: ServingParams) -> float:
    """Slow-down of decode due to clock: 1.0 above the knee, knee/clock below it.

    Example with knee 0.6: clock 0.8 -> 1.0 (no change); clock 0.3 -> 2.0 (twice as slow).
    """
    _check_clock(clock_ratio)
    return 1.0 if clock_ratio >= p.decode_clock_knee else p.decode_clock_knee / clock_ratio


def tpot_seconds(active: int, clock_ratio: float, p: ServingParams) -> float:
    """Time per output token for one request when ``active`` requests share the batch."""
    fill = min(max(active, 0), p.max_batch) / p.max_batch
    return p.tpot_base_s * (1.0 + p.tpot_batch_slope * fill) * decode_clock_factor(clock_ratio, p)


class Occupancy(NamedTuple):
    """How the GPU was used over one sampling window (both in [0, 1])."""

    duty: float  # fraction of time at least one request was running (what DCGM calls GPU utilisation)
    fill: float  # time-averaged share of batch slots in use (8 of 16 slots all window -> 0.5)


class BusyTracker:
    """Measures :class:`Occupancy` since the last ``sample()``.

    The power emulator needs two numbers, not one:

    * **duty** -- was the GPU running *anything*? Decode is memory-bound: every step reads all
      the model weights from GPU memory whether the batch holds 1 request or 16, so most of
      the power is spent just by running steps at all.
    * **fill** -- how full was the batch? Extra requests add their own KV-cache reads and
      compute, so power still rises with fill, just less steeply.

    Instantaneous counts jump around, so both are time integrals divided by elapsed time.
    Time is passed in (``now``) rather than read from a clock so tests are exact.
    """

    def __init__(self, max_batch: int, now: float) -> None:
        self._max_batch = max_batch
        self._active = 0
        self._last_change = now  # when ``_active`` last changed
        self._window_start = now  # start of the current averaging window
        self._slot_area = 0.0  # integral of active-slots x seconds in this window
        self._run_time = 0.0  # seconds in this window with >= 1 request active

    @property
    def active(self) -> int:
        return self._active

    def _advance(self, now: float) -> None:
        dt = max(now - self._last_change, 0.0)
        self._slot_area += self._active * dt
        if self._active > 0:
            self._run_time += dt
        self._last_change = now

    def change(self, delta: int, now: float) -> None:
        """Record a request starting (+1) or finishing (-1) at time ``now``."""
        self._advance(now)
        self._active = max(self._active + delta, 0)

    def sample(self, now: float) -> Occupancy:
        """Occupancy since the previous sample; starts a new window."""
        self._advance(now)
        elapsed = now - self._window_start
        if elapsed <= 0:  # zero-length window: report the current state
            duty, fill = float(self._active > 0), self._active / self._max_batch
        else:
            duty, fill = self._run_time / elapsed, self._slot_area / (elapsed * self._max_batch)
        self._window_start = now
        self._slot_area = self._run_time = 0.0
        return Occupancy(duty=min(max(duty, 0.0), 1.0), fill=min(max(fill, 0.0), 1.0))
