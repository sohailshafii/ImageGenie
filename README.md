# ImageGenie

An end-to-end pipeline that mass-downloads 3D models, weak-labels them from store
metadata, and trains a multi-view CNN to classify them — combining **distributed
systems, ML, and a web frontend**. Portfolio project.

## What it does

1. **Build annotated data** — bulk-download 3D models from [Objaverse](https://objaverse.allenai.org/)
   (~800k Sketchfab-sourced objects), derive weak labels from categories/tags/titles, and correct
   them in a labeling UI.
2. **Train** — a multi-view CNN (renders → ResNet) over ~12 visually-distinct classes.
3. **Evaluate** — two dev sets with per-class precision/recall + confusion matrices and bias analysis.

## Architecture

A queue + workers pipeline (embarrassingly parallel preprocessing) on **GCP**:

```
Objaverse ─▶ download ─▶ GCS(raw) ─▶ [convert ▶ normalize ▶ render] ─▶ GCS(processed) + Postgres
                                      each stage a Cloud Run worker fed by Pub/Sub
```

Cloud Run (workers) · Pub/Sub (queue) · Cloud Storage (blobs) · Cloud SQL / Postgres (metadata) ·
Vertex AI (training). Every worker is idempotent; the whole thing targets a **~$100 cloud budget**.

## Status

- ✅ **Milestone 1** — metadata exploration + locked 12-class list
- ✅ **Weak labeling (FR-3)** — category gate → keyword resolution → out-of-scope rescue; 57% gold
  coverage at ~0.91 precision, graded against the curated LVIS gold set
- ✅ **Milestone 2** — pipeline skeleton (queue + download worker), verified end-to-end in Docker
- ✅ **Milestone 3** — cloud deployment (Terraform: APIs, budget alerts, storage, Pub/Sub, Cloud SQL,
  Cloud Run); pipeline runs end-to-end on GCP
- ✅ **Milestone 4** — full ingestion. Convert → normalize → render stages deployed to Cloud Run
  (scale-to-zero, per-stage Pub/Sub push + DLQ); ran the labeled 12-class set (32k seeded) with
  resilience tuning (2–4 GiB, one-model-per-instance, in-worker retry + backoff) and a DLQ-replay tool
  to recover transient mirror failures
- ✅ **Milestone 5** — labeling frontend (React + TS + Vite) on a FastAPI backend-for-frontend,
  deployed to Cloud Run. The labeling loop works end to end: sign in, browse real rendered previews,
  open a model in the three.js viewer (its normalized mesh from the pipeline), and confirm or correct
  the label — attributed to the admin who made the change. Also done: admin data upload (FR-9),
  invite-gated signup with email verification (Resend), session cookies with CSRF and rate limiting,
  the weak-label and Objaverse-metadata backfills that populate the catalog, sort-by-least-confidence
  and a keyboard sweep for fast review, an admin dead-letter view over recorded pipeline failures, and
  Alembic migrations. The API and SPA ship as one image on one origin; `scripts/adopt_schema.sh`
  rebuilds the database from the buckets on deploy.
- ✅ **Milestone 6** — baseline training. `ml/train.py` trains a multi-view CNN (resnet18 over the 12
  rendered views → pool → head) on the weak labels, reading renders from the processed bucket, with
  reproducible per-class stratified splits, per-epoch checkpoints, and the NFR-4 bookkeeping every
  run records. It runs **on a Vertex AI spot T4** — proven end-to-end on a real GPU, including
  Cloud SQL over the IAM connector and parallel GCS reads — and can be started either from the
  command line (`make train-cloud`) or from the dashboard's **Start a run** page, which shows the
  measured GPU time and cost before the button. The **dashboard** (run list, cost curve with train /
  val loss and validation accuracy, per-class precision/recall and a confusion matrix) is deployed.
  **The one thing not yet done is the full-set run itself** — the ~11.8k-model baseline is a
  button-press, not missing machinery; it is deferred deliberately because at ~55 min/epoch it wants
  a considered epoch count rather than a default.
- 🚧 **Milestone 7** — evaluation. `make evaluate RUN=n` scores a finished run against the held-out
  **test** split and stores the report per (run, dev set); the run detail page renders it beside the
  run's own `val` metrics, kept deliberately separate so the optimistic number is not mistaken for
  the honest one. The classifier is also usable directly — predict a catalog model from its detail
  page, or upload any mesh at `/classify` and get a class back without ingesting it. The **bias
  writeup** is in [ml/ml.md](ml/ml.md#bias-analysis): class skew and tail collapse, a measured ~9%
  weak-label error ceiling, evidence the model is under- rather than over-trained, and the bias the
  pipeline itself introduces by rendering shape only.
  **Outstanding:** FR-7 asks for *two* dev sets and only one exists. Just 49 LVIS-gold-labeled models
  fall in our held-out split — too few to report — so a real second dev set means ingesting ~1,000
  independently-annotated objects through the existing pipeline. Scoped as a data run, deliberately
  after the remaining MVP work.

## Layout

| Dir | What |
|-----|------|
| `ml/` | class list, weak labeling, evaluation ([ml/ml.md](ml/ml.md)) |
| `server/` | pipeline workers, queue, storage, DB, API ([server/server.md](server/server.md)) |
| `infra/` | Terraform for the GCP resources |
| `web/` | labeling UI + training dashboard ([web/web.md](web/web.md)) |

Design docs are the source of truth — see [CLAUDE.md](CLAUDE.md) for the project hub.

## Run locally

```
make setup          # venv + ml/server/dev deps
make test           # test suite (Postgres via testcontainers)
make weaklabel      # Sketchfab weak labeling over sampled shards
make evalweak       # grade weak labels vs the LVIS gold set
```

**The pipeline** — Postgres + Pub/Sub emulator + a worker per stage:

```
make compose-up
make compose-seed COUNT=100   # download jobs that flow through every stage
make compose-down
```

**The labeling app** — needs a Postgres it can reach, then the API and the dev server:

```
make migrate                  # apply schema migrations (Alembic owns the schema)
make backfill-labels          # load weak_labels.csv into the DB, so the catalog has labels
make backfill-metadata        # fetch Objaverse titles/tags (downloads shard files on first run)

cd server && ../.venv/bin/python -m uvicorn app.api:app --port 8000
cd web && npm install && npm run dev      # http://localhost:5173
```

The dev server proxies `/api` and `/artifacts` to the API so the browser sees a single origin —
the session cookies are `SameSite=Lax` and the CSRF defense depends on that
([web/web.md](web/web.md#auth--roles)).

## Deploy to the cloud

Heavy work runs on GCP (Cloud Run + Cloud SQL + GCS). You bring your own GCP project (with billing
enabled) and a Resend account with a verified domain — nothing here is shared.

```
make cloud-tools              # terraform, gcloud, cloud-sql-proxy (macOS/Homebrew)
make deploy-config            # scaffold .env + infra/terraform.tfvars from the examples
```

Fill in the two scaffolded files:

- `infra/terraform.tfvars` — `project_id`, `region`, `billing_account`, `budget_amount`
- `.env` — `TF_VAR_mail_from`, `TF_VAR_resend_api_key` (Sending-access key), and the admin login

Then run the deploy:

```
set -a; source .env; set +a   # export the secrets for Terraform + the scripts
make deploy-image             # build + push the API/worker image
scripts/adopt_schema.sh       # drop + rebuild the schema from storage (destructive, gated)
terraform -chdir=infra apply  # create the API service; prints api_url
scripts/check_deploy.sh       # health + the URL-signing check
```

Finally set `app_base_url = <api_url>` in `infra/terraform.tfvars` and re-apply, so email links point
at the real host. Full flow and gotchas:
[server/server.md](server/server.md#deploying-the-api-to-cloud-run).

## Distribution policy

**Code only.** Labeled data and trained models are **not** redistributed — you run the pipeline
yourself. This respects Objaverse/Sketchfab licensing.

## License

The **code** in this repository is licensed under the [MIT License](LICENSE). This covers the source
only — it grants no rights to Objaverse/Sketchfab 3D models, any data produced by the pipeline, or
trained models, none of which are distributed here (see the distribution policy above).
