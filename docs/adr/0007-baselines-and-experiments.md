# ADR-0007: Two baselines and the experiment matrix

Date: 2026-11-11 · Status: accepted

## Context

The claim to test is that *allocation* -- choosing which knobs move, guided by a power model
and the tier constraints -- beats simple rules. That needs baselines run on the same cluster,
the same load and the same event, and a record of each event detailed enough to plot.

## Decision

**Policies** (`spec.policy`):
* `optimizer` -- ADR-0006: model-based search over flex admission and replicas, applied whole
  (feedforward) and trimmed by feedback.
* `ladder` -- a fixed rule list: shed all flex, then remove replicas one at a time while above the
  aim. Ignores the model and the utilisation limit.
* `uniform` -- Stage 0's naive response: lower **every** GPU's clock by 0.1 of max per settle
  until under the aim (floor 0.4). No tiers, no consent, no choice of GPU. Clock is a *root* lever
  (it changes hardware, not workload), so it requires `allowRoot: true`: only an operator that
  owns the hardware may use it.

**Caveat on `uniform`.** On kind the clock's effect is the emulator's assumed model (dynamic power
∝ clock³; decode insensitive above 0.6 of max clock, prefill slows ∝ 1/clock). The contract has no
measured clock depth (n/a). Kind results for `uniform` are labelled *assumed-physics*; Phase 4
measures a real clock lock on the T4 through a privileged per-node agent.

**Admission granularity**: 0.1 steps (was 0.25), so the optimiser sheds less flex than the
coarse steps forced.

**Cap noise estimate**: the std dev of single readings *around the 20-s average* (was: of the
readings themselves, which counts slow load drift as noise and doubled the margin).

**Event record** (`scripts/eventlib.py`): every event saves `spec.json`, `status.json`,
`result.json` (scores) and `timeseries.csv` (5-s Prometheus samples: watts, 20-s average, flex
admit, ready replicas, clock, p99 TTFT per tier, offered/admitted rate) under `results/`.
`scripts/event_report.py` plots one event (watts / knobs / latency panels) or compares several.

**Experiment matrix** (`scripts/experiments.py`, ~3 h, resumable):
* 3 policies × {cap, interval} at −100 W, equal-thirds mix (6 runs);
* depth sweep, optimiser, cap: −50, −150, −200 W (3 runs; −100 W above);
* tier-mix sweep, optimiser, cap, −100 W: premium/standard/flex 70/15/15 and 15/15/70 (2 runs).
Each run: 8 rps baseline, 300 s ramp, 600 s hold (one full aligned 5-min interval).
Per run: time to target, cap peak and seconds above, interval compliance, achieved vs requested
reduction, share of flex shed, p99 TTFT per tier, failed protected requests, release order.

## Consequences

* One run per cell: differences smaller than run-to-run noise (watts ±20 W, p99 on ~200 requests
  per minute) are not claims. The report states that and repeats a cell where a result matters.
* Prefill/decode pools and Pub/Sub event intake stay out of scope unless time allows.
