# Setup and roadmap

Everything below runs in your **Mac's own terminal** (ADR-0002). Each step says what it does
and why, so you can stop at any point and know where you are.

## 1. Tools on the Mac (once, ~20 min)

```bash
# Homebrew, if you don't have it: https://brew.sh
brew install python@3.12 kind kubectl helm gh
brew tap hashicorp/tap && brew install hashicorp/tap/terraform   # Terraform left Homebrew core in 2023
brew install --cask google-cloud-sdk                              # Docker: Docker Desktop (already installed) or OrbStack
```

* **kind** runs a whole Kubernetes cluster as Docker containers on your laptop. It needs a
  Docker engine; OrbStack provides one (Docker Desktop works too). Open OrbStack once after
  installing so the engine starts.
* **kubectl** is the Kubernetes command line; **helm** installs packaged apps (we use it for
  Prometheus); **gh** is the GitHub CLI.

Check: `kind version && kubectl version --client && helm version && terraform -version && gcloud --version`.

## 2. Put the repo on GitHub (once, ~5 min)

```bash
cd ~/"Modern Cloud Computing"/k8s-power-budget-operator
gh auth login                       # once
git init -b main
git add .
git commit -m "Phase 0: scaffold, lever-curve loader, ADRs 0001-0003, Stage 0 study"
gh repo create karthikmattu06-hue/k8s-power-budget-operator --public --source=. --push
make install && make check          # same checks CI runs; should be all green
```

## 3. GCP project on the course credit (once, ~15 min)

A separate project keeps the course credit, quota request and resources apart from any
other project, so accounting and clean-up are unambiguous.

```bash
gcloud auth login
gcloud billing accounts list                                   # note the course billing account ID
gcloud projects create pbo-course-$RANDOM --name="Power budget operator"
gcloud config set project <the project id printed above>
gcloud billing projects link <project id> --billing-account=<course billing account ID>
gcloud services enable compute.googleapis.com container.googleapis.com \
    monitoring.googleapis.com pubsub.googleapis.com
```

**Budget alert.** Console → Billing → Budgets & alerts → Create: scope = this project,
amount $40, alerts at 50 % and 90 %. (If the course billing account does not let you create
budgets, skip it and check Billing → Reports after every session.)

## 4. GPU quota request (today; approval can take days or be refused)

Console → IAM & Admin → **Quotas & System Limits** → filter, select, *Edit quotas*:

| Quota | Region | New value |
|---|---|---|
| GPUs (all regions) | global | 1 |
| NVIDIA T4 GPUs | us-central1 | 1 |
| Preemptible NVIDIA T4 GPUs | us-central1 | 1 |

Spot VMs draw on the *preemptible* GPU quota when it exists, so ask for it too.
Justification text: *"Graduate course project (RIT, Modern Cloud Computing): one T4 Spot GPU
for a few hours a week to run a vLLM inference server on GKE; cluster destroyed after each
session; funded by the course credit."*

If it is refused within a minute (new billing accounts often are), reply to the case asking
for the reason, and ask the instructor (below). The project does not stop: phases 1–3 need no GPU.

**Email to the instructor** (send today):

> Subject: GPU quota for my course project
>
> Hi Prof. Fang, my project needs one NVIDIA T4 on GKE for a few hours a week. I have requested
> quota on a project linked to the course billing account. In case new accounts are refused
> automatically, does the course provide a project with GPU quota I could use? Without a GPU
> I can still complete the project on emulated nodes, but one real GPU node lets me validate
> the power telemetry end to end. Thanks, Karthik

## 5. Run Phase 1 on the Mac (~15 min)

Docker Desktop must be running. Give it at least 4 CPUs and 6 GB RAM (Settings → Resources):
six kind nodes plus Prometheus is a real cluster.

```bash
make phase1          # creates the cluster, builds/loads the image, installs Prometheus, deploys
kubectl -n pbo get pods -o wide     # 5 mock-llm (one per worker), 5 gpu-emulator, 1 loadgen
make smoke           # ~10 min; ends with "PHASE 1: PASS"
make prom-ui         # then open http://localhost:9090 and try the queries below
```

Queries to try (Prometheus → Graph):

| Question | PromQL |
|---|---|
| Watts per node | `DCGM_FI_DEV_POWER_USAGE` |
| Cluster watts | `sum(DCGM_FI_DEV_POWER_USAGE)` |
| p99 TTFT per node | `histogram_quantile(0.99, sum by (le, node) (rate(vllm:time_to_first_token_seconds_bucket[1m])))` |
| p99 TTFT per tier (client side) | `histogram_quantile(0.99, sum by (le, tier) (rate(pbo_loadgen_ttft_seconds_bucket[1m])))` |
| Queue length per node | `vllm:num_requests_waiting` |

Change the load by hand: `kubectl -n pbo port-forward deploy/loadgen 9101 &` then
`curl -XPOST localhost:9101/rate -d '{"rps": 13}'`. Capacity: a full batch of 16 runs at 30 ms
per output token, so an average request takes ~4.9 s and one replica serves ~3.3 rps; 5 replicas
saturate near **16 rps**. Above that the queue grows without bound. The baseline operating point
is **8 rps** (~50 %), where all SLOs hold with margin; `make smoke` checks this. (At 10 rps premium
p99 TTFT came out 1.25–1.81 s across runs: too close to the 2 s SLO for a reliable baseline.)

Stop everything with `make kind-down` (the cluster uses CPU even when idle).

## 6. Run Phase 2 on the Mac (~30 min)

`make deploy` installs the `PowerBudget` CRD, the operator and the admission proxy (all client
traffic now enters through it). After `make image deploy`:

```bash
kubectl -n pbo get pods              # + 1 pbo-operator, 1 admission
make calibrate                       # ~7 min, once per cluster: measures the power curve the operator plans with
make smoke2 2>&1 | tee smoke.log     # ~20 min; cap mode, optimiser; ends with "PHASE 2: PASS"
python3 scripts/smoke_phase2.py --policy ladder 2>&1 | tee ladder.log   # the fixed-order baseline, to compare
```

Try an event by hand:

```bash
kubectl apply -f deploy/examples/powerbudget.yaml
kubectl get pb -w                    # phase, target, measured, flex admit, replicas, shortfall
kubectl -n pbo logs deploy/pbo-operator -f
kubectl get pb evening-peak -o jsonpath='{.status.plan}'   # what the optimiser chose and why
kubectl delete pb evening-peak       # ends the event early; replicas and flex come back
```

If `kubectl delete pb` hangs, the operator is not running: it holds a finalizer so it can restore
replicas before the object goes. Start the operator again, or as a last resort
`kubectl patch pb <name> --type merge -p '{"metadata":{"finalizers":[]}}'`. If the operator dies
mid-event, the admission proxy readmits flex by itself after 30 s (lease).

## Roadmap

Learn-first items are what to read *before* starting the phase, and where.

| Phase | Learn first | Build | Done when |
|---|---|---|---|
| **0 Setup** | – | this repo, GCP project, budget, quota request | CI green; quota requested |
| **1 Local cluster** ($0) | Kubernetes basics: Pods, Deployments, Services, Labels ([Learn Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/), 2–3 h); [kind quick start](https://kind.sigs.k8s.io/docs/user/quick-start/) (20 min) | kind cluster (1 + 5 nodes); Prometheus via Helm; mock serving pod (vLLM API and metric names, honours `X-Service-Tier`); power-emulator **DaemonSet** exposing DCGM-named metrics; open-loop load generator ([ADR-0004](adr/0004-emulated-gpu-nodes.md)) | `make smoke` passes: watts and p99 per node rise with load; a node with no pod drops to parked watts, and at constant load the cluster total falls when a replica is removed |
| **2 CRD + operator** ($0) | [Custom Resources](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/), [Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) (1 h); [kopf](https://kopf.readthedocs.io/): Getting started, Handlers, Timers, Status (2 h); [RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/) (20 min); [pod-deletion-cost](https://kubernetes.io/docs/reference/labels-annotations-taints/#pod-deletion-cost) (10 min) | `PowerBudget` CRD (interval and cap modes); kopf operator, closed loop on measured watts; admission proxy with lease; optimiser over flex admission + replicas with a calibrated power model; ladder baseline; graceful drain ([ADR-0005](adr/0005-powerbudget-operator.md), [ADR-0006](adr/0006-admission-and-optimizer.md)) | `make smoke2` passes in cap mode: target reached in the ramp, no reading above the cap, premium SLO held, release in order, no failed premium/standard requests |
| **3 Allocator + levers** ($0) | re-read `studies/stage0/stage0_allocation.py` (its greedy loop is the allocator); [PromQL basics](https://prometheus.io/docs/prometheus/latest/querying/basics/) | allocation across pools (prefill/decode) and per-tier admission rates; drain; event report; ladder vs uniform vs optimiser | uniform vs merit order end to end on kind |
| **4 GKE + T4** (~$5) | [Terraform GCP tutorial](https://developer.hashicorp.com/terraform/tutorials/gcp-get-started) (2 h); GKE docs: GPU node pools, Spot VMs, DCGM metrics, Managed Prometheus (2 h); vLLM *Deploying with Kubernetes* (1 h) | Terraform: cluster, Spot T4 pool, Spot CPU pool; vLLM on the T4; DCGM → Managed Prometheus | real T4 watts respond to a budget |
| **5 Grid + SLO guard** (~$2) | Grafana alerting + webhooks; Pub/Sub Python client | price/emergency replay → Pub/Sub → `PowerBudget`; burn-rate alert → back-off | a replayed event runs hands-off |
| **6 Experiments + report** (~$20) | – | policy × depth × trace × tier-mix matrix; figures; USENIX report | time to depth and hold accuracy per policy |

Target: phases 1–3 by early November, GKE experiments done by late November, report in the
first week of December.
