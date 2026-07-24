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
- 🚧 **Milestone 5** — labeling frontend (React + TS + Vite) on a FastAPI backend-for-frontend.
  The labeling loop works end to end: sign in, browse real rendered previews, open a model in the
  three.js viewer (its normalized mesh from the pipeline), and confirm or correct the label —
  attributed to the admin who made the change. Also done: invite-gated signup with email
  verification (Resend), session cookies with CSRF and rate limiting, the weak-label and
  Objaverse-metadata backfills that populate the catalog, sort-by-least-confidence and a keyboard
  sweep for fast review, an admin dead-letter view over recorded pipeline failures, and Alembic
  migrations.
  **Remaining:** admin data upload (FR-9), and deploying the API itself — only the workers are in
  Terraform today.
- ⬜ **Milestone 6** — baseline training (multi-view CNN on weak labels, spot GPU)
- ⬜ **Milestone 7** — evaluation (both dev sets, confusion matrices, bias writeup)

## Layout

| Dir | What |
|-----|------|
| `ml/` | class list, weak labeling, evaluation ([ml/ml.md](ml/ml.md)) |
| `server/` | pipeline workers, queue, storage, DB, API ([server/server.md](server/server.md)) |
| `infra/` | Terraform for the GCP resources |
| `web/` | labeling UI ([web/web.md](web/web.md)); the training dashboard lands with milestone 6 |

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
