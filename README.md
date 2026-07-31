# ImageGenie

<p align="center">
  <img src="docs/images/nacho-genie.jpg" width="360"
       alt="The ImageGenie mascot — a fiery nacho genie in a jewelled turban, rising from a lamp">
</p>

An end-to-end pipeline that mass-downloads 3D models, weak-labels them from store
metadata, and trains a multi-view CNN to classify them — combining **distributed
systems, ML, and a web frontend**. Portfolio project.

## Live demo

A single-instance deployment runs on Cloud Run:
**[imagegenie-api-hhitzs4jka-uc.a.run.app](https://imagegenie-api-hhitzs4jka-uc.a.run.app)**

> 🔑 **Signup is invite-only — you can't self-register.** Accounts are created from an invite minted
> by an admin, then confirmed by email; there is no open signup route. Ask me and I'll issue one for
> your address. An invited **viewer** can browse the catalog, open a model in the three.js viewer,
> see the training dashboard, and classify a mesh at `/classify`; correcting labels and uploading
> models are **admin**-only (NFR-7).

> ⚠️ **Best-effort demo — expect a slow first load, and it may be down.** The service scales to zero,
> so the first request cold-starts the container: **~12 seconds**, because the image carries a CPU
> build of PyTorch for the classify endpoint. After that it responds in about 0.1 s. It runs on a
> hobby budget (roughly $100 total, of which the always-on database is the bulk) and may be taken
> offline without notice. To run your own, see [Deploy to the cloud](#deploy-to-the-cloud).

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

**Milestones 1–6 are done. 7 and 8 are complete but for one item they share** — a second, independently
annotated dev set, which FR-7 asks for and which the active-learning loop needs in order to measure
itself. The pipeline ingests and renders ~12k models,
the labeling UI and training dashboard are deployed, and the current model (**run 15**) scores
**0.4241 accuracy / 0.3401 macro recall** on a sealed held-out split — inside the ~0.42–0.47 band
every full-set run has landed in, on a corpus where the majority-class baseline is ~18%.

Per-milestone checklists, including what is deliberately outstanding and why, are in
**[STATUS.md](STATUS.md)**. The bias analysis behind those numbers is in
[ml/ml.md](ml/ml.md#bias-analysis).

## Layout

| Dir | What |
|-----|------|
| `ml/` | class list, weak labeling, evaluation ([ml/ml.md](ml/ml.md)) |
| `server/` | pipeline workers, queue, storage, DB, API ([server/server.md](server/server.md)) |
| `infra/` | Terraform for the GCP resources |
| `web/` | labeling UI + training dashboard ([web/web.md](web/web.md)) |

Design docs are the source of truth — see [CLAUDE.md](CLAUDE.md) for the project hub and
[STATUS.md](STATUS.md) for milestone progress.

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
