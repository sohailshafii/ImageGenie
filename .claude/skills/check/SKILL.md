---
name: check
description: Run ImageGenie's lint / typecheck / test gates the way this repo actually defines them — Python (ruff + pytest) and frontend (eslint + tsc). Use before staging or offering a commit, or whenever asked to lint, typecheck, or run the tests. Records the exact script names and working directories, which are easy to guess wrong.
---

# Run the checks

**There is no `npm run typecheck`.** Typechecking happens inside `npm run build`
(`tsc -b && vite build`). Guessing script names wastes a round-trip — the four
commands below are the whole set.

## Python — from the repo root

```
cd /Users/sohailshafii/src/ImageGenie
.venv/bin/ruff check .          # or: make lint
.venv/bin/pytest -q             # or: make test
```

- `pytest` spins up Postgres via testcontainers, so **Docker must be running**
  (`docker info` to confirm). The full suite takes ~90s.
- Run a single file with `.venv/bin/pytest server/tests/test_downloads.py -q`.
- `pyproject.toml` sets `pythonpath=["server","ml"]` and `testpaths` covers both
  `server/tests` and `ml/tests`, so the root is the correct cwd.

## Frontend — from `web/`

```
cd /Users/sohailshafii/src/ImageGenie/web
npm run lint                    # eslint .
npm run build                   # tsc -b && vite build  ← this is the typecheck
```

`npm run format` (prettier) exists but rewrites files — don't run it as a check.

## Working-directory gotchas

- The `Bash` tool's cwd **persists between calls**, so a bare relative path like
  `web/src/index.css` fails after an earlier `cd web`. Prefer absolute paths, or
  `cd` explicitly in the same command.
- `zsh` is the shell: an unquoted `--include=*.py` glob fails with
  "no matches found". Quote it: `grep -rn "pat" --include='*.py' .`

## What "clean" means before staging

All four green: `ruff check` reports "All checks passed", `pytest` shows no
failures **or errors**, `eslint` prints nothing, `vite build` ends with `✓ built`.
Then `git add -A`, show `git status` + the diff, and stop for review — per
CLAUDE.md, never commit without explicit approval.
