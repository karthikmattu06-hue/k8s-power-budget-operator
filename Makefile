# Run `make help` for targets. Everything uses the local virtualenv in .venv/.
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help install fmt lint type test check stage0 clean

help:
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

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

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache **/__pycache__
