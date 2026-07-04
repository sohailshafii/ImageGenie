# Server / Infra — ImageGenie

Backend of the pipeline: ingestion, the queue + workers preprocessing chain, the API layer,
object storage, and the metadata database. See [../CLAUDE.md](../CLAUDE.md) for the project hub.

## Scope

- **Download workers** — pull models from the model store API (FR-1). Polite rate limiting,
  no scraping, resumable.
- **Conversion workers** — STL/OBJ/GLB/FBX → a common internal format.
- **Normalize workers** — center, rescale to a unit bounding box, validate mesh integrity.
- **Render workers** — produce multi-view images (v1) or point clouds (stretch).
- **API layer** — FastAPI. Enqueues jobs, serves labels/metadata/results to the frontend,
  handles auth (see [web.md](../web/web.md#auth--roles)).
- **Storage** — object storage for raw + processed artifacts; metadata DB for everything queryable.

## Queue + Workers

- Embarrassingly parallel; scale by adding workers (NFR-3).
- **Idempotency (NFR-2)** — every worker checks whether its output already exists (by content
  hash / model id + stage) and skips if so. Reruns must be safe. This is the single most
  important backend invariant — a crashed batch must be re-runnable without duplicating work.
- **At-least-once delivery** — assume messages can be redelivered; design handlers to tolerate it.
- **Dead-letter handling** — models that fail a stage N times go to a dead-letter queue with the
  error recorded in the DB, not silently dropped.
- Queue technology is an [open decision](../CLAUDE.md#open-decisions) (managed vs. Redis+Celery).

## Database

Requirements (resolves the DB TODO):

- **Purpose** — the metadata DB is the source of truth for pipeline state and labels; object
  storage holds the heavy binary artifacts (raw meshes, rendered images/point clouds). The DB
  stores paths/keys into object storage, never the blobs themselves.
- **Core entities:**
  - `model` — store id, source URL, license, download status, content hash, raw object key.
  - `artifact` — per-stage outputs (converted / normalized / rendered), object keys, stage status.
  - `label` — model id, class, source (`weak` | `manual`), confidence, annotator, timestamp.
    Keep weak and manual labels as distinct rows so weak-vs-corrected analysis is possible
    (see [ml.md](../ml/ml.md#evaluation)).
  - `training_run` — id, config snapshot, data version, metrics, artifact keys, status
    (feeds the [dashboard](../web/web.md#training-dashboard)).
  - `user` — id, email, role (`user` | `admin`) — see [web.md](../web/web.md#auth--roles).
- **Access patterns:** filter models by label/status for the browse view (paginated); look up a
  single model + its artifacts + labels for the detail view; aggregate label counts per class
  for metadata exploration and dashboards.
- **Requirements:** relational (foreign keys between model/artifact/label), transactional
  status updates, cheap-to-run (fits the budget), managed if the chosen cloud offers a
  low-cost tier.
- **Shortlist to decide from:** Postgres (managed: Cloud SQL / RDS) as the default candidate;
  SQLite acceptable only for the local ~100-model skeleton (milestone 2). **Decide alongside
  the cloud provider** ([open decision](../CLAUDE.md#open-decisions)).

## Coding Standards (backend)

- **Language:** Python 3.11+. Type hints on all public functions; check with a static type checker.
- **Framework:** FastAPI for the API; Pydantic models for request/response and job payloads.
- **Style:** format + lint with Ruff (or black + ruff). No unformatted code committed.
- **Structure:** each worker type is its own module with a single `process(job)` entrypoint;
  shared storage/DB access behind thin client modules — no direct SDK calls scattered in workers.
- **Config:** all cloud identifiers, bucket names, and credentials via environment variables /
  secrets — never hardcoded.
- **Idempotency first:** every handler is written check-output-then-act. Add a test that runs the
  handler twice and asserts no duplicate work.
- **Logging:** structured logs with model id + stage on every message; errors recorded to the DB,
  not just stdout.
- **Tests:** unit-test conversion/normalize logic on tiny fixture meshes; integration-test the
  queue path on the ~100-model set before scaling.
