# ADR-0002: Build on a local kind cluster first, GKE second

Date: 2026-10-19 · Status: accepted

## Context

The cloud budget is the $50 course credit. GPU quota on a new billing account may be refused
or take time. Most of the system (CRD, operator, admission proxy, mock serving pod, power
emulator, Prometheus) does not need a GPU.

## Decision

* Phases 1–3 run on **kind** (Kubernetes in Docker) on the developer's Mac: 1 control-plane
  node and 5 workers standing in for 1 real + 4 emulated nodes. Cost $0.
* The same manifests are applied to GKE in Phase 4. Anything that differs between kind and GKE
  (GPU node pool, DCGM, Managed Prometheus) is isolated behind a kustomize overlay.
* GKE clusters are created by Terraform and destroyed at the end of every session.

## Consequences

* Debugging happens off the meter; GKE time is spent on integration and experiments only.
* The T4 quota request can be pending for weeks without blocking work.
* kind does not reproduce GKE's real pod start-up times, image pulls or Managed Prometheus
  scrape intervals; timing results come from GKE runs only.
