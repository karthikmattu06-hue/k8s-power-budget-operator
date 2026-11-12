# ADR-0005: PowerBudget resource, two kinds of obligation, closed-loop operator

Date: 2026-11-11 · Status: accepted (revised the same day after the first kind run)

## Context

Phase 2 turns a grid event into actions on the cluster. The Phase 1 and first Phase 2 runs
established:

* Removing one of five replicas at constant load saved the cluster 72 W while that GPU alone
  dropped ~135 W: the remaining replicas absorb its work. Per-GPU depth over-predicts.
* Removing a replica raises cluster power slightly for a few seconds before it falls: the
  draining pod finishes its in-flight requests while the others take its new load. Measured on
  single 5-s readings this is small (0–5 W in the 11 Nov runs). The much larger jumps first
  seen in the 20–30-s averages (880 W against an 848 W baseline) were a **measurement artifact**:
  the emulator's power series carry a `pod` label, so after a pod is removed the old and new
  series overlap inside the window and `sum(avg_over_time(...))` counts that node twice.
  Aggregate first, then average: `avg_over_time(sum(...)[20s:5s])`.
* Prometheus sees a change only after the emulator's stale timeout, a scrape and an averaging
  window. A reading taken mid-transition can show a dip that is not real (the first run
  declared "target reached" on one).
* Obligations come in two kinds. Market demand response settles on metered **interval
  averages** (PJM hourly; ERCOT 15-min settlement, 5-min dispatch), and every interval must
  comply. A **hard cap** (a physical limit behind a flexible interconnection) must not be
  exceeded at any reading.

## Decision

* **Resource.** `PowerBudget` (`pbo.dev/v1alpha1`), cluster-scoped. Spec: a reduction relative
  to the baseline measured at event start (watts or fraction); `mode` = `interval` (default,
  `intervalSeconds` 300) or `cap`; `marginWatts` (planner aims this far below the target);
  `rampSeconds` (default 600); `holdSeconds`; `minReplicas`; `policy` = `optimizer` | `ladder`;
  `flexAction` = `reject` | `defer`; `utilizationLimit`; `replicaCapacityRps`; `scope`.
  Status holds the whole state machine, so the operator process is stateless.
* **Phases.** Pending → Ramping → Holding → Releasing → Done. Holding starts when the latest plan
  is fully in place and a *settled* reading is at or below target, or at the ramp deadline.
* **Settling.** No decision -- not even "target reached" -- is made until the last move has
  settled: the lever's `speed_s` from the lever-curve contract (admission 8 s, replica 90 s)
  plus the sensing lag (5-s scrape + 20-s averaging, 30 s by default).
* **5-minute scoring for `interval`.** The shortest settlement interval in use; if every aligned
  5-min average complies, every 15-min and hourly one does too.
* **Cap mode.** A replica move is allowed during the hold only if the latest reading plus the
  removal spike estimate stays under the cap, and during the ramp only if it will have settled
  before the deadline. Admission changes have no spike and are always allowed. The spike is
  learned from the operator's own removals: peak single reading after the move minus the 20-s
  average before it. 50 W until the first removal is measured, then the largest measured
  (0–8 W in the 11 Nov runs; the noise margin already covers single-reading scatter).
* **Cap margin covers sensor noise.** A cap is judged on single 5-s readings, which scatter
  around the 20-s average the planner sees (std dev 22–32 W in the 11 Nov runs). At event start
  the operator reads the std dev of single readings over the past 2 min and uses
  `margin = max(marginWatts, noiseSigmas × std dev)` (default 3.5), frozen for the event.
  The first cap run, with a fixed 30 W margin and a spike wrongly learned as 0 W against a single
  high reading, crossed the cap by 10 W for 10 s after a removal made during the hold.
* **Consent and authority.** Only Deployments labelled `pbo.dev/sheddable=true`, never below
  `minReplicas`; only flex requests are shed; only levers the contract marks `operator`.
  When the target cannot be met within these, the gap is reported (`shortfallWatts`).
* **Which pod goes.** `controller.kubernetes.io/pod-deletion-cost: -1000` on the pod on the
  node with the emptiest batch, then scale down by one.
* **Release.** Restore replicas first; readmit flex only when they are all Ready (or after
  120 s). Readmitting flex onto the reduced pool would overload it. Deleting the PowerBudget
  mid-event restores everything (finalizer).
* **Graceful drain.** Serving pods: 5 s `preStop`, then finish in-flight requests for up to 30 s.
* **Sensing.** Operator ticks every 5 s; Prometheus scrapes the emulator, proxy and replicas
  every 5 s. Baseline load for tests: 8 rps (~50 % of capacity).

How the moves are chosen is ADR-0006.

## Consequences

* Replicas alone are too coarse and spiky for a cap; cap compliance depends on the admission
  lever (ADR-0006).
* A hard cap on a noisy signal costs headroom: the margin grows with the noise, so the cut
  overshoots more and more flex is shed than under interval scoring of the same event.
* Reaching a target takes one settle period for the whole plan (~120 s when replicas move), plus
  one per corrective trim.
* `kubectl delete pb` waits for the operator (finalizer); if the operator is down it hangs until
  the operator returns or the finalizer is removed by hand.
* The operator's 5-s view cannot see shorter spikes; a real cap would be checked against the
  utility's own telemetry rate.
