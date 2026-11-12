# Run `make help` for targets. Everything uses the local virtualenv in .venv/.
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help install fmt lint type test check stage0 clean \
        kind-up image prom deploy phase1 smoke calibrate smoke2 experiments report prom-ui kind-down

# Phase 1: local kind cluster
CLUSTER ?= pbo
IMAGE ?= pbo:dev
# kube-prometheus-stack chart version, pinned for reproducibility
KPS_VERSION ?= 92.11.0

help:
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)

install: $(BIN)/python ## create .venv and install pbo with dev tools
	$(BIN)/pip install -q -e ".[dev]"

fmt: ## auto-format
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

lint: ## style and lint checks (no changes)
	$(BIN)/ruff format --check src tests
	$(BIN)/ruff check src tests

type: ## strict type check
	$(BIN)/mypy

test: ## unit tests
	$(BIN)/pytest

check: lint type test ## everything CI runs

stage0: ## rerun the Stage 0 allocation study (writes into studies/stage0/)
	$(BIN)/pip install -q -e ".[study]"
	cd studies/stage0 && ../../$(BIN)/python stage0_allocation.py \
	  && ../../$(BIN)/python plot_paper.py stage0_results.csv stage0_paper.pdf

kind-up: ## create the local kind cluster (1 control plane + 5 workers), if not already there
	kind get clusters | grep -qx $(CLUSTER) || kind create cluster --name $(CLUSTER) --config deploy/kind/cluster.yaml

image: ## build the container image and copy it into the kind nodes
	docker build -f docker/Dockerfile -t $(IMAGE) .
	kind load docker-image $(IMAGE) --name $(CLUSTER)

prom: ## install Prometheus (kube-prometheus-stack) into the cluster
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update
	helm upgrade --install kps prometheus-community/kube-prometheus-stack --version $(KPS_VERSION) \
	  -n monitoring --create-namespace -f deploy/prometheus/values-kind.yaml --wait --timeout 10m

deploy: ## apply CRD, operator, proxy and workloads, and restart them onto the freshly loaded image
	kubectl apply -k deploy/base
	kubectl -n pbo rollout restart deploy/mock-llm deploy/admission deploy/loadgen deploy/pbo-operator ds/gpu-emulator
	kubectl -n pbo rollout status deploy/mock-llm --timeout=180s
	kubectl -n pbo rollout status ds/gpu-emulator --timeout=180s
	kubectl -n pbo rollout status deploy/admission --timeout=120s
	kubectl -n pbo rollout status deploy/pbo-operator --timeout=120s

phase1: kind-up image prom deploy ## all of the above, in order

smoke: ## Phase 1 acceptance test against the running cluster (~10 min)
	python3 scripts/smoke_phase1.py

calibrate: ## measure the power curve the operator plans with (~7 min, once per cluster)
	python3 scripts/calibrate.py

smoke2: ## Phase 2 acceptance test: cap-mode event planned by the optimiser (~20 min)
	python3 scripts/smoke_phase2.py

experiments: ## run the experiment batch (11 events, ~3 h; resumable with the same BATCH)
	python3 scripts/experiments.py $(if $(BATCH),--batch $(BATCH),)

report: ## plot saved events: make report RUN="results/<dir> [results/<dir> ...]"
	$(BIN)/pip install -q -e ".[study]"
	$(BIN)/python scripts/event_report.py $(RUN)

prom-ui: ## Prometheus UI at http://localhost:9090 (Ctrl-C to stop)
	kubectl -n monitoring port-forward svc/kps-prometheus 9090

kind-down: ## delete the kind cluster
	kind delete cluster --name $(CLUSTER)

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache **/__pycache__
