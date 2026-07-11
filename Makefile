# ImageGenie — dev setup + run targets.
#
# macOS framework-Python doesn't trust the system cert store, so any command that
# hits the network (objaverse downloads) must point OpenSSL at certifi's CA bundle
# via SSL_CERT_FILE. $(RUN) wires that in for you.

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
MODE   ?= lvis

# venv Python with the cert shim (see header). `$$(...)` defers to recipe-time so
# it isn't evaluated before the venv exists.
RUN := SSL_CERT_FILE=$$($(BIN)/python -m certifi) $(BIN)/python

.PHONY: setup lint explore clean help

help: ## show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN{FS=":.*## "}{printf "  %-10s %s\n", $$1, $$2}'

setup: ## create the virtualenv and install runtime + dev deps
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt -e ".[dev]"

lint: ## ruff-check the codebase
	$(BIN)/ruff check .

explore: ## run milestone-1 metadata exploration (MODE=lvis|raw|both)
	$(RUN) ml/explore_metadata.py --mode $(MODE)

classlist: ## build + validate the final class list from LVIS merges (ml/taxonomy.py)
	$(RUN) ml/build_class_list.py

clean: ## remove the virtualenv and caches
	rm -rf $(VENV) .ruff_cache
