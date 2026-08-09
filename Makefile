# ImageGenie — dev setup + run targets.
#
# macOS framework-Python doesn't trust the system cert store, so any command that
# hits the network (objaverse downloads) must point OpenSSL at certifi's CA bundle
# via SSL_CERT_FILE. $(PYRUN) wires that in for you.

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
MODE   ?= lvis
SHARDS ?= 1
COUNT  ?= 100
# Separate from COUNT (which sizes a compose smoke-seed) because the dev set is a
# real data run: 1,000 is the size FR-7 is scoped at, not a pilot default.
DEVSET_COUNT ?= 1000
COMPOSE := docker compose -f server/docker-compose.yml

GCP_PROJECT  ?= imagegenie-pipeline
GCP_REGION   ?= us-central1
WORKER_IMAGE := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/imagegenie/worker:latest
# The training image (ml/Dockerfile) is a separate artifact from the worker image:
# CUDA torch instead of CPU torch, and none of the mesh/web stack.
#
# Tagged by COMMIT, never `:latest`. A training job is submitted by tag and then
# runs unattended for hours, so a floating tag makes "the image is older than the
# code" invisible — which is exactly how a paid GPU run once started with a stale
# image, silently ignored every CLI flag, and trained the full set on CPU.
GIT_SHA      := $(shell git rev-parse --short HEAD 2>/dev/null)
TRAIN_IMAGE  := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/imagegenie/train:$(GIT_SHA)
# Identities from infra/training.tf: BUILD_SA builds the image, TRAINER_SA runs
# the Vertex job.
BUILD_SA     ?= imagegenie-build@$(GCP_PROJECT).iam.gserviceaccount.com
TRAINER_SA   ?= imagegenie-trainer@$(GCP_PROJECT).iam.gserviceaccount.com

# Run a script through the venv Python ($(BIN)/python, which has the deps — not
# $(PYTHON), the system interpreter used only to bootstrap the venv in `setup`).
# The venv Python appears twice on purpose: `python -m certifi` prints the path
# to certifi's CA bundle, which is exported as SSL_CERT_FILE for the second
# python that actually runs the script (the cert shim; see header). Uses shell
# `$$(...)`, not make's $(shell ...), so certifi is located at recipe time — not
# at parse time, which would fail (e.g. on `make help`) before the venv exists.
PYRUN := SSL_CERT_FILE=$$($(BIN)/python -m certifi) $(BIN)/python

.PHONY: setup cloud-tools lint test explore clean help devset compose-up compose-seed compose-down deploy-image backfill-labels backfill-metadata reconcile-storage cleanup-raw migrate migration migration-status train smoke-train evaluate review-queue train-image train-cloud

help: ## show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN{FS=":.*## "}{printf "  %-13s %s\n", $$1, $$2}'

setup: ## create the virtualenv and install ml + server + dev deps
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt -r server/requirements.txt -e ".[dev]"

cloud-tools: ## install cloud-deploy CLIs (terraform, gcloud, cloud-sql-proxy) — macOS/Homebrew, idempotent
	brew install hashicorp/tap/terraform
	brew install --cask google-cloud-sdk
	brew install cloud-sql-proxy

deploy-config: ## scaffold .env + infra/terraform.tfvars from the examples (won't clobber)
	@test -f .env || { cp .env.example .env && echo "created .env — fill in the secrets"; }
	@test -f infra/terraform.tfvars || { cp infra/terraform.tfvars.example infra/terraform.tfvars && echo "created infra/terraform.tfvars — fill in project/billing"; }
	@echo "Edit both, then: set -a; source .env; set +a; make deploy-image; scripts/adopt_schema.sh; terraform -chdir=infra apply"

lint: ## ruff-check the codebase
	$(BIN)/ruff check .

test: ## run the test suite (server tests spin up Postgres via testcontainers)
	$(BIN)/pytest

explore: ## run milestone-1 metadata exploration (MODE=lvis|raw|both)
	$(PYRUN) ml/explore_metadata.py --mode $(MODE)

classlist: ## build + validate the final class list from LVIS merges (ml/taxonomy.py)
	$(PYRUN) ml/build_class_list.py

weaklabel: ## Sketchfab weak labeling over sampled shards (SHARDS=N, default 1)
	$(PYRUN) ml/weak_label.py --shards $(SHARDS)

evalweak: ## evaluate weak labels vs the LVIS gold set (SHARDS=N, default 1)
	$(PYRUN) ml/eval_weak_labels.py --shards $(SHARDS)

evalboundary: ## measure the figure/animal boundary — can keywords resolve it? (SHARDS=N, default 8)
	# Reproduces the numbers ml.md#the-figureanimal-boundary rests on: how much of
	# the ambiguous population a keyword precedence rule could even reach, and
	# whether any token carries stance signal. SHARDS=24 for the gold figures cited.
	$(PYRUN) ml/eval_figure_animal.py $(if $(SHARDS),--shards $(SHARDS),)

devset: ## select the second dev set from un-ingested LVIS gold objects (FR-7; DEVSET_COUNT=N)
	# Needs BOTH the cert shim (it reads LVIS annotations over the network) and
	# PYTHONPATH=server (it asks the DB what is already ingested), so it is the one
	# ml target that combines $(PYRUN)'s shim with the DB path. Point it at Cloud SQL
	# through the proxy — against a local DB every uid looks un-ingested.
	SSL_CERT_FILE=$$($(BIN)/python -m certifi) PYTHONPATH=server $(BIN)/python \
	    ml/build_dev_set.py --count $(DEVSET_COUNT)

devset-push: ## copy the existing dev-set CSV to the processed bucket, so cloud jobs can score it
	# Push-only, and deliberately not folded into `devset`: the candidate filter is
	# "no model row", and these objects were ingested so they could be rendered, so
	# re-selecting now would skip the whole current dev set and draw a different
	# 1,000. This uploads what is on disk and selects nothing.
	# No cert shim — it talks to GCS, not the LVIS annotation host.
	PYTHONPATH=server $(BIN)/python ml/build_dev_set.py --push-only

train: ## run a baseline training run (M6); writes training_run + metrics to the DB
	# PYTHONPATH=server so ml/train.py can import the DB layer (app.db, app.models);
	# no cert shim needed — this run only touches Postgres, not the network.
	PYTHONPATH=server $(BIN)/python ml/train.py

evaluate: ## score a finished run against a dev set (M7; RUN=n, DEVSET=test|val|train|lvis)
	# Separate from training because `val` is steered against every epoch and
	# `test` is not — see ml/evaluate.py. DEVSET=lvis scores the second dev set
	# (FR-7): a different corpus rather than a partition of ours, read from the
	# CSV `make devset` writes. PYTHONPATH=server for the DB layer, as with `train`.
	@test -n "$(RUN)" || { echo "usage: make evaluate RUN=<training run id> [DEVSET=test]"; exit 1; }
	# SPLIT was this target's flag until `lvis` arrived. Silently ignoring a stale
	# SPLIT=val would score `test` and store it under the wrong dev set, so refuse.
	@test -z "$(SPLIT)" || { echo "SPLIT= is now DEVSET= for this target (lvis is not a split)"; exit 1; }
	PYTHONPATH=server $(BIN)/python ml/evaluate.py --run $(RUN) $(if $(DEVSET),--dev-set $(DEVSET),)

review-queue: ## build the M8 hand-labeling queue from a run's disagreements (RUN=n, SPLIT=test, LIMIT=N, WORKERS=4)
	# Where the classifier and the stored label disagree, one of them is wrong —
	# and only a human can say which. That split is what separates real model
	# error from the weak-label ceiling (ml.md#bias-analysis).
	@test -n "$(RUN)" || { echo "usage: make review-queue RUN=<training run id> [SPLIT=test] [LIMIT=N] [WORKERS=4]"; exit 1; }
	PYTHONPATH=server $(BIN)/python ml/review_queue.py --run $(RUN) \
		$(if $(SPLIT),--split $(SPLIT),) $(if $(LIMIT),--limit $(LIMIT),) \
		$(if $(WORKERS),--num-workers $(WORKERS),)

smoke-train: ## CPU smoke: seed a tiny class-separable dataset, train end-to-end, assert it learns (self-cleaning)
	# Needs a reachable, migrated Postgres (IMAGEGENIE_DATABASE_URL, default
	# localhost:5432). PYTHONPATH=server:ml so it imports both the DB layer and the
	# ml modules (train/dataset/model/splits).
	PYTHONPATH=server:ml $(BIN)/python ml/smoke_train.py

migrate: ## apply pending schema migrations (alembic upgrade head)
	cd server && ../$(BIN)/alembic upgrade head

migration: ## autogenerate a migration from model changes (MSG="what changed")
	cd server && ../$(BIN)/alembic revision --autogenerate -m "$(MSG)"

migration-status: ## show the current revision and any pending ones
	cd server && ../$(BIN)/alembic current && ../$(BIN)/alembic heads

backfill-metadata: ## fetch Objaverse titles/tags for models missing them (LIMIT=N, DRYRUN=1)
	cd server && ../$(BIN)/python -m app.backfill_metadata \
		$(if $(LIMIT),--limit $(LIMIT),) $(if $(DRYRUN),--dry-run,)

reconcile-storage: ## rebuild the model/artifact tables from object storage (DRYRUN=1 to preview)
	cd server && ../$(BIN)/python -m app.reconcile_from_storage $(if $(DRYRUN),--dry-run,)

cleanup-raw: ## delete raw meshes for models excluded from the dataset (dry run; APPLY=1 to delete)
	# Runs from the repo root, unlike `reconcile-storage`: on the local backend
	# `storage_root` is cwd-relative, so `cd server` would point it at an empty
	# server/data/storage and report nothing to delete. (Against GCS the cwd is
	# irrelevant, which is how that trap stays hidden until someone tests locally.)
	# PYTHONPATH carries both packages — ml/ for the class roster, mirroring how
	# `train` adds server/ so ml code can reach the DB layer.
	PYTHONPATH=server:ml $(BIN)/python -m app.cleanup_raw $(if $(APPLY),--apply,)

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

deploy-image: ## build (linux/amd64) + push the worker/API image to Artifact Registry
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev --quiet
	# Context is the repo root so the build can compile the SPA (web/) alongside
	# the server; the Dockerfile is multi-stage (server/Dockerfile).
	docker build --platform linux/amd64 -f server/Dockerfile -t $(WORKER_IMAGE) .
	docker push $(WORKER_IMAGE)

train-image: ## build + push the CUDA training image via Cloud Build (M6 chunk G)
	# Built in the cloud, not locally: the CUDA base is multi-GB and this host is
	# arm64, so a local linux/amd64 build runs under emulation and then pushes
	# those gigabytes back up. Context is the repo root — the image needs both
	# server/app and ml/ (see ml/Dockerfile).
	# --service-account is required, not optional: this project has no
	# <number>@cloudbuild.gserviceaccount.com (Google stopped auto-creating that
	# legacy default for newer projects), so a build without one has no identity
	# and fails with PERMISSION_DENIED before it starts. The SA is created in
	# infra/training.tf. A user-specified SA also has no default log bucket, hence
	# --default-buckets-behavior.
	gcloud builds submit . --project $(GCP_PROJECT) \
		--config ml/cloudbuild.yaml \
		--substitutions=_IMAGE=$(TRAIN_IMAGE),_LOGS_BUCKET=$(GCP_PROJECT)-build-logs \
		--service-account=projects/$(GCP_PROJECT)/serviceAccounts/$(BUILD_SA)

train-cloud: ## submit a Vertex AI spot-GPU training run (LIMIT=500 for a subset; ARGS='...' to pass more)
	# Cost guardrail (CLAUDE.md): the FIRST cloud run should pass LIMIT to prove
	# the wiring on a few hundred models before paying for the full ~11.8k.
	#   make train-cloud LIMIT=500
	#   make train-cloud                      # the whole trainable set
	#   make train-cloud ARGS='--epochs 5'
	#   make train-cloud SPOT=0                # on-demand, for a long run
	#   make train-cloud MAX_HOURS=2           # tighter hard timeout (default 6)
	#
	# SPOT=0 is not a cost preference, it is what makes a multi-hour run finish.
	# A preemption RESTARTS a run rather than pausing it (see ml/vertex_job.yaml),
	# so a 6-hour job in a contended region can retry indefinitely without ever
	# passing epoch 1 — while the job state stays RUNNING and nothing looks wrong.
	#
	# Two preflight checks, both learned the expensive way. A job runs unattended
	# for hours, so an image that doesn't match the code is not a slow failure —
	# it is a *silent* one: the old entrypoint ignored every flag and trained the
	# full set on CPU, looking like a healthy run the whole time.
	@dirty=$$(git status --porcelain -- ml server/app); \
	if [ -n "$$dirty" ]; then \
		echo "ERROR: uncommitted changes in ml/ or server/app — the image is tagged by"; \
		echo "commit ($(GIT_SHA)), so what would run is NOT what is in your tree:"; \
		echo "$$dirty"; \
		echo "Commit (or stash), then: make train-image"; \
		exit 1; \
	fi
	@gcloud artifacts docker images describe $(TRAIN_IMAGE) --project $(GCP_PROJECT) >/dev/null 2>&1 || { \
		echo "ERROR: no training image for commit $(GIT_SHA)."; \
		echo "Build it first:  make train-image"; \
		exit 1; \
	}
	@set -e; \
	spec=$$(mktemp -t imagegenie-vertex); \
	trap 'rm -f "$$spec"' EXIT; \
	: "$${DEVICE:=cuda}"; : "$${WORKERS:=4}"; \
	if [ "$(SPOT)" = "0" ]; then strategy=STANDARD; else strategy=SPOT; fi; \
	: "$${MAX_HOURS:=6}"; timeout=$$(( MAX_HOURS * 3600 ))s; \
	echo "scheduling: $$strategy, hard timeout $$timeout"; \
	args=$$($(BIN)/python -c 'import json,sys; print(json.dumps(sys.argv[1:]))' \
		--device "$$DEVICE" --num-workers "$$WORKERS" \
		$(if $(LIMIT),--limit $(LIMIT),) $(ARGS)); \
	sed -e 's|__IMAGE__|$(TRAIN_IMAGE)|' \
	    -e 's|__SERVICE_ACCOUNT__|$(TRAINER_SA)|' \
	    -e 's|__INSTANCE__|$(GCP_PROJECT):$(GCP_REGION):imagegenie-pg|' \
	    -e 's|__SECRET__|projects/$(GCP_PROJECT)/secrets/imagegenie-database-url/versions/latest|' \
	    -e "s|__ARGS__|$$args|" \
	    -e "s|__STRATEGY__|$$strategy|" \
	    -e "s|__TIMEOUT__|$$timeout|" \
	    ml/vertex_job.yaml > "$$spec"; \
	echo "--- job spec ---"; cat "$$spec"; echo "----------------"; \
	gcloud ai custom-jobs create --project $(GCP_PROJECT) --region $(GCP_REGION) \
		--display-name=imagegenie-train --config="$$spec"

clean: ## remove the virtualenv and caches
	rm -rf $(VENV) .ruff_cache
