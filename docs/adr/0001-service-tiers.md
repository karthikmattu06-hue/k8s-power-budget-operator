# ADR-0001: Three service tiers, per-request, with consent-based authority

Date: 2026-10-19 · Status: accepted

## Context

The proposal (15 Oct) used two tiers, premium and flex, and the Stage 0 study modelled a
node as hosting one or the other. The lever-curve contract (schema 0.3 rev 2) defines three
tiers with assumed SLOs taken from QoServe (Goel et al., ASPLOS '26), and tags tiers per
request rather than per node. Results from this project should be comparable with numbers
measured against that contract.

## Decision

* Tiers are `premium`, `standard`, `flex`, carried per request in the header
  **`X-Service-Tier`** (header name is local to this project; tier names and meanings are shared).
* SLOs (all `assumed`): premium TTFT p99 2 s, standard TTFT p99 6 s, TPOT p99 50 ms for both;
  flex time-to-last-token 600 s.
* Default request mix is equal thirds; sensitivity runs at 70/15/15 and 15/15/70.
* **Authority follows consent.** Only flex has agreed in advance to be deferred or rejected
  during an event. Per-node budgets are **binding** on flex capacity and **advisory** on
  standard and premium. When an advisory node does not shed its share, the shortfall is
  reported, never forced. Quality-degrading knobs (quantization, reasoning effort) are never used.
* `reject` = shed work is lost (no rebound). `defer` = held until release, then admitted at
  once (the rebound). A defer can last at most 600 s, the flex TTLT SLO.

## Consequences

* Nodes now differ in tier through their request *mix*, not a single label; the emulator gets
  a per-node mix parameter.
* Stage 0 numbers (two tiers) are kept as the proposal's preliminary result; the allocator is
  re-run on three tiers once it is ported.
* Events longer than 10 minutes cannot be covered by deferral alone; the allocator must switch
  to reject or another lever before the 600 s cap.
