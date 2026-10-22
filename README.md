# k8s-power-budget-operator

A Kubernetes operator that takes one power target for a cluster of LLM inference
services and decides **which nodes, pools and services shed, in what order**, so the
cluster reaches the target within a grid deployment window without breaking premium
latency SLOs.

Course project, Modern Cloud Computing, Fall 2026 (RIT). Proposal: `docs/proposal.pdf`.

## How it works

```
grid event ──► PowerBudget (CRD) ──► operator ──► per-node budgets ──► node levers
   (Pub/Sub)     target, window,      merit order:     binding on flex,     admission (reject/defer),
                 premium p99          cheapest SLO     advisory on          drain, replicas,
                                      cost per watt    standard/premium     cap/clock (root only)
      ▲                                                                            │
      └──── back off ◄── Grafana SLO alert ◄── Prometheus ◄── DCGM + vLLM metrics ◄┘
```

* **Tiers:** `premium`, `standard`, `flex` (request header `X-Service-Tier`). Default mix equal
  thirds. See [ADR-0001](docs/adr/0001-service-tiers.md).
* **Lever curves:** the operator reads a datasheet of levers (depth, speed, duration, cost,
  release) from `curves/`. See [ADR-0003](docs/adr/0003-lever-curve-contract.md).
* **Testbed:** one real T4 GPU node on GKE plus emulated CPU nodes whose scaling, admission and
  drain are real Kubernetes actions; only watts are modelled. Development starts on a local
  `kind` cluster. See [ADR-0002](docs/adr/0002-local-first-development.md).

## Layout

| Path | What |
|---|---|
| `src/pbo/curves.py` | Loads and validates the lever-curve contract and the published overlay |
| `curves/` | `lever-curves-v0.3.json` (contract, schema 0.3 rev 2, unchanged copy) and `published-overlay.json` (ours) |
| `studies/stage0/` | Stage 0 allocation study: uniform vs greedy vs LP vs MIP on a 12-node model |
| `docs/adr/` | Architecture decision records |
| `docs/setup.md` | Mac and GCP setup, step by step |

## Develop

```bash
make install   # venv + editable install with dev tools
make check     # ruff format --check, ruff check, mypy, pytest
make stage0    # rerun the Stage 0 study and figures
```

## Status

Phase 0 (scaffold, contract loader, quota request). Roadmap in [docs/setup.md](docs/setup.md#roadmap).
