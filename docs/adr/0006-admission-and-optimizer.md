# ADR-0006: Admission proxy with a lease, and a model-based optimiser instead of fixed rules

Date: 2026-11-11 · Status: accepted

## Context

* Under concave GPU power (ADR-0004), shedding requests saves little on its own -- every GPU
  still runs a batch. Rejecting all flex at 8 rps saves ~45 W. Its value is that it frees
  capacity so GPUs can be *parked* without pushing premium past its latency limit.
* A hard-coded order ("all flex first, then replicas") overshoots and sheds more flex than
  needed. Emerald AI's published field trial (Phoenix, arXiv:2507.00909) chooses actions by
  solving for power reduction subject to per-job SLA tiers, with a simulator predicting power
  and performance -- a model, constraints and an objective, not a rule list.
* The lever-curve contract describes levers per GPU and per pool; its numbers are mostly
  estimates and will change. The operator must use them without being rewritten when they do.

## Decision

**Admission proxy** (`pbo.admission.proxy`), in front of the serving Service.
* Premium and standard always pass. Flex passes with probability `flex_admit` (1 … 0); the
  rest is rejected (503, `Retry-After`) or deferred (held until readmitted, at most 600 s).
* The operator sets `flex_admit` over HTTP every tick. Each call renews a **30-s lease**; if it
  lapses, the proxy readmits everyone. A dead controller cannot leave customers shut out.
* One upstream connection per request, so the Service balances requests evenly.

**Optimiser** (`pbo.operator.planner`), re-run every settled tick (receding horizon):
* Knobs: `flex_admit` ∈ {1, 0.75, 0.5, 0.25, 0}; replicas ∈ [minReplicas, current]. Knobs only
  move down during an event (no flapping).
* Minimise flex req/s shed + 0.05 × replicas removed (a tie-breaker: parking a GPU loses no
  work), subject to predicted watts ≤ target − margin and admitted load ÷ (replicas × capacity)
  ≤ utilisation limit (0.6: premium p99 held at ~50 %, marginal near 60 %).
* Exhaustive search (≤ 30 states); infeasible targets get the deepest premium-safe cut and a
  reported shortfall (budgets bind on flex, advise on premium/standard).
* The whole plan is applied in one tick (feedforward), then trimmed after each settle (feedback).
  Shedding flex lowers utilisation and removing replicas raises it, so any end state within the
  limit is safe to reach directly. The second cap run, moving one knob per settle period, could
  not finish its plan inside a 300 s ramp and crossed the cap from a half-finished state.

**Power model** (`pbo.operator.model`): one curve, node watts vs requests/s per replica,
measured on the cluster by `scripts/calibrate.py` (ConfigMap `pbo-power-model`); fallback is a
contract prior (parked → busy). A bias term, updated from measured minus predicted watts after
every settled tick, keeps the level honest. The emulator's own formula is never used.

**Contract adapter** (`pbo.operator.levers`): the only code that reads the contract. It supplies
settle times (`speed_s`), parked/busy power and nominal watts, the authority filter, SLO
thresholds and defer limits, each tagged with its provenance in the event status. Known gaps
(load only in state-row notes; combined depth of two levers not measured; no monolithic-pool
admission entries) are listed there and covered by measured feedback until the producer fills them.

**Baseline.** `policy: ladder` keeps the fixed-order rule for comparison in the evaluation.

## Consequences

* The proxy is now on every request's path (one extra hop, one replica). Several proxy replicas
  would need the setting fanned out to each.
* Plans are only as good as the calibration; feedback corrects level errors, not shape errors.
  Re-run calibration when the serving setup changes.
* Allocation across several pools (prefill/decode) and finer admission (per-tier rates) extend
  the same search in Phase 3.
