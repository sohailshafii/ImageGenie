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
- 🚧 **Milestone 3** — cloud deployment (Terraform: APIs + budget alerts live; storage/queue/DB/Cloud
  Run in progress)

## Layout

| Dir | What |
|-----|------|
| `ml/` | class list, weak labeling, evaluation ([ml/ml.md](ml/ml.md)) |
| `server/` | pipeline workers, queue, storage, DB, API ([server/server.md](server/server.md)) |
| `infra/` | Terraform for the GCP resources |
| `web/` | labeling UI + dashboard (planned, [web/web.md](web/web.md)) |

Design docs are the source of truth — see [CLAUDE.md](CLAUDE.md) for the project hub.

## Run locally

```
make setup          # venv + ml/server/dev deps
make test           # test suite (Postgres via testcontainers)
make weaklabel      # Sketchfab weak labeling over sampled shards
make evalweak       # grade weak labels vs the LVIS gold set
make compose-up     # local pipeline skeleton (Postgres + Pub/Sub emulator + worker)
make compose-seed COUNT=100
make compose-down
```

## Distribution policy

**Code only.** Labeled data and trained models are **not** redistributed — you run the pipeline
yourself. This respects Objaverse/Sketchfab licensing.

## License

The **code** in this repository is licensed under the [MIT License](LICENSE). This covers the source
only — it grants no rights to Objaverse/Sketchfab 3D models, any data produced by the pipeline, or
trained models, none of which are distributed here (see the distribution policy above).
