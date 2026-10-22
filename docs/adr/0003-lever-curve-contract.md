# ADR-0003: Consume a frozen lever-curve contract; add published numbers in our own overlay

Date: 2026-10-19 · Status: accepted

## Context

The operator needs, for every lever on every pool: how much power it sheds, how fast, for how
long, at what latency cost per tier, and what happens on release. These numbers are measured
by a companion measurement project and published as a JSON contract, schema 0.3 revision 2,
frozen on 19 Oct 2026 after a two-sided review. No code is shared between the two projects;
only the file format and its meanings.

## Decision

* `curves/lever-curves-v0.3.json` is an unchanged copy of the contract
  (sha256 `34e025a90d3d05e33dc9d52e798f984065eb45e81a3ab63cdf61076ea1558013` at 19 Oct).
  New values are pulled by replacing the file; the loader refuses any schema version or
  revision other than 0.3 rev 2.
* `curves/published-overlay.json` is ours. It may only **add** entries, each pointing at a
  hardware record with `origin: "published"` and a citation. It never edits a contract entry.
  A paper's statistic is entered as reported: a median is not entered in a p99 field.
* When several entries describe the same lever on the same pool, the loader prefers
  contract-measured, then contract-estimated, then published.
* Depth at the current pool load is interpolated linearly over `at_load.pool_busy_frac` and
  clamped to the measured range; it is never extrapolated.
* Every number keeps its `status` (measured / estimated / assumed / n/a) into the reports.

## Consequences

* A structural change on the producer side fails loudly at load time instead of silently
  changing allocation decisions.
* Until measured entries arrive (first batch expected 29 Oct, one load per entry), most depths
  are estimates; every figure states which.
