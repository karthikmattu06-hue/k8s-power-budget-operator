# ADR-0004: Emulated GPU nodes -- per-node power DaemonSet, vLLM/DCGM metric names, monolithic pods first

Date: 2026-10-22 · Status: accepted

## Context

Phases 1–3 run on kind with no GPU, and most nodes stay emulated on GKE. The operator must
see the same signals from an emulated node as from the real T4 node, or the control loop
tested locally is not the one that runs on GKE. Three choices shape that.

## Decision

1. **Power is emulated per node by a DaemonSet**, not a sidecar in the serving pod. This
   mirrors NVIDIA's DCGM exporter, which is itself a DaemonSet. The serving pod reports its
   busy fraction, clock and role every second through a node-local Service
   (`internalTrafficPolicy: Local`); the emulator turns that into watts (revised 10 Nov, below):
   `P = nominal × (parked + (busy_role − parked) × duty × (w + (1 − w) × fill) × clock³)`,
   with parked and busy fractions read from the contract's `states` (0.17; prefill 0.92,
   decode 0.79, mixed 0.84)
   and nominal 250 W. A node whose pod is gone or silent for 3 s reports parked watts, which
   a sidecar could not do: it would vanish with the pod.
2. **Metric names copy the real exporters.** Serving metrics use vLLM's names
   (`vllm:time_to_first_token_seconds`, `vllm:time_per_output_token_seconds`,
   `vllm:num_requests_running`/`waiting`, ...); power uses DCGM's (`DCGM_FI_DEV_POWER_USAGE`,
   `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_SM_CLOCK`, `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION`) with
   DCGM's labels. Queries and dashboards then work unchanged against vLLM + DCGM on the T4.
   Extras that real exporters lack carry a `pbo_` prefix (`pbo_emulated_gpu`,
   `pbo_requests_by_tier_total`, `pbo_clock_ratio`).
3. **Monolithic pods first; prefill/decode split later.** Phase 1 deploys `role=mono`
   replicas. The code already accepts `role=prefill|decode` and the emulator uses the
   role's busy power; the split goes live in Phase 3 with the admission proxy routing between
   pools.

The latency model (`pbo.mockserve.model`) is illustrative, not measured: batch of 16 slots,
prefill 4000 tok/s scaled by clock, TPOT 20 ms rising 50 % with a full batch, decode
insensitive to clock above 0.6 of max. Its job is the shape of the response (queueing under
load, clock sensitivity), not absolute numbers.

## Consequences

* Replica changes, admission and drain are real Kubernetes actions on emulated nodes; only
  watts and clock are modelled. The emulated cluster tests loop dynamics, not GPU physics.
* Every reading carries `pbo_emulated_gpu=1`, so reports can separate emulated from measured.
* The clock³ term is an assumed approximation; it is replaced by the contract's measured
  depth for clock locking when that arrives.
* The emulator models one GPU per node, matching one replica per node in Phase 1.

## Revision, 2026-11-10: power is concave in load

The first version reported one number, `busy_frac` (time-averaged batch fill), and made power
linear in it. The first kind runs showed the consequence: moving the same load onto fewer
replicas left total power unchanged (the removed node's savings reappear as fuller batches
elsewhere), so the replica lever had ~zero depth at constant load. Real GPUs are not like that:
decode is memory-bound and every step streams all model weights whatever the batch size, so a
GPU running any batch draws much of its busy power. The contract agrees: removing a replica has
depth 0.63 of that GPU's power.

Now the serving pod reports two numbers each second: **duty** (share of time at least one
request ran; also what DCGM's `GPU_UTIL` means) and **fill** (batch occupancy). The share of
dynamic power that does not depend on batch size, `w`, defaults to **0.6 (assumed)** and is a
flag on the emulator (`--batch-independent-frac`). At `w = 0.6`, removing one of five replicas
at 10 rps saves about 100 W (~11 % of the cluster) while the remaining four absorb the work.
`w` is calibrated once measured power at 2–3 loads per lever entry arrives; the acceptance test
checks that consolidation saves > 5 %.
