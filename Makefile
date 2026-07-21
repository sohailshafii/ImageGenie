# ImageGenie — dev setup + run targets.
#
# macOS framework-Python doesn't trust the system cert store, so any command that
# hits the network (objaverse downloads) must point OpenSSL at certifi's CA bundle
# via SSL_CERT_FILE. $(RUN) wires that in for you.

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
MODE   ?= lvis
SHARDS ?= 1
COUNT  ?= 100
COMPOSE := docker compose -f server/docker-compose.yml

GCP_PROJECT  ?= imagegenie-pipeline
GCP_REGION   ?= us-central1
WORKER_IMAGE := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/imagegenie/worker:latest

# Run a script through the venv Python ($(BIN)/python, which has the deps — not
# $(PYTHON), the system interpreter used only to bootstrap the venv in `setup`).
# The venv Python appears twice on purpose: `python -m certifi` prints the path
# to certifi's CA bundle, which is exported as SSL_CERT_FILE for the second
# python that actually runs the script (the cert shim; see header). Uses shell
# `$$(...)`, not make's $(shell ...), so certifi is located at recipe time — not
# at parse time, which would fail (e.g. on `make help`) before the venv exists.
RUN := SSL_CERT_FILE=$$($(BIN)/python -m certifi) $(BIN)/python

.PHONY: setup cloud-tools lint test explore clean help compose-up compose-seed compose-down deploy-image backfill-labels

help: ## show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN{FS=":.*## "}{printf "  %-13s %s\n", $$1, $$2}'

setup: ## create the virtualenv and install ml + server + dev deps
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt -r server/requirements.txt -e ".[dev]"

cloud-tools: ## install cloud-deploy CLIs (terraform, gcloud) — macOS/Homebrew, idempotent
	brew install hashicorp/tap/terraform
	brew install --cask google-cloud-sdk

lint: ## ruff-check the codebase
	$(BIN)/ruff check .

test: ## run the test suite (server tests spin up Postgres via testcontainers)
	$(BIN)/pytest

explore: ## run milestone-1 metadata exploration (MODE=lvis|raw|both)
	$(RUN) ml/explore_metadata.py --mode $(MODE)

classlist: ## build + validate the final class list from LVIS merges (ml/taxonomy.py)
	$(RUN) ml/build_class_list.py

weaklabel: ## Sketchfab weak labeling over sampled shards (SHARDS=N, default 1)
	$(RUN) ml/weak_label.py --shards $(SHARDS)

evalweak: ## evaluate weak labels vs the LVIS gold set (SHARDS=N, default 1)
	$(RUN) ml/eval_weak_labels.py --shards $(SHARDS)

backfill-labels: ## load weak_labels.csv into the DB's label table (idempotent; DRYRUN=1 to preview)
	cd server && ../$(BIN)/python -m app.backfill_labels \
		--labels ../data/exploration/weak_labels.csv \
		--eval ../data/exploration/weak_label_eval.json $(if $(DRYRUN),--dry-run,)

compose-up: ## build + start the pipeline skeleton (Postgres, Pub/Sub emulator, worker)
	$(COMPOSE) up -d --build

compose-seed: ## publish COUNT download jobs into the running skeleton (default 100)
	$(COMPOSE) run --rm seed python -m app.seed --count $(COUNT)

compose-down: ## stop the skeleton and remove its volumes
	$(COMPOSE) down -v

deploy-image: ## build (linux/amd64) + push the worker image to Artifact Registry
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev --quiet
	docker build --platform linux/amd64 -t $(WORKER_IMAGE) server/
	docker push $(WORKER_IMAGE)

clean: ## remove the virtualenv and caches
	rm -rf $(VENV) .ruff_cache
