# ImageGenie — 3D Model Classification Pipeline

> **What this file is:** the project hub. It holds the cross-cutting context —
> overview, requirements, budget, architecture, milestones, and open decisions.
> Domain-specific detail and coding standards live in the linked domain docs.

## Domain Docs

| Domain | Doc | Covers |
|--------|-----|--------|
| Server / infra | [server/server.md](server/server.md) | Ingestion pipeline, queue + workers, FastAPI, storage, database, idempotency, backend coding standards |
| Web / frontend | [web/web.md](web/web.md) | Labeling UI (browse + detail views), training dashboard, auth/roles, data upload, frontend coding standards |
| ML | [ml/ml.md](ml/ml.md) | Representations, training, evaluation, weak-label policy, dev-set splits, confusion matrices, ML coding standards |

## Project Overview

A portfolio project combining ML, distributed systems, and a frontend. The goal is an
end-to-end pipeline that:

1. **Builds annotated data** — mass-download 3D models from a free model store, label them
   (metadata-derived weak labels + manual correction via a web UI). See [web/web.md](web/web.md)
   for the labeling UI and [ml/ml.md](ml/ml.md) for the labeling policy.
2. **Trains a classifier** — categorize 3D models into ~10–20 well-populated classes
   (e.g., chair, car, lamp, figurine). See [ml/ml.md](ml/ml.md).
3. **Evaluates the model** — against two dev sets, with explicit bias/error analysis. See
   [ml/ml.md](ml/ml.md).

**Distribution policy:** Only code is distributed. Labeled data and trained models are NOT
redistributed (respects model-store licensing). Users run the pipeline themselves.

## Functional Requirements

- **FR-1 Ingestion** — download 3D models (STL/OBJ/GLB/FBX) from a free model store via its
  official API, with polite rate limiting and no scraping.
- **FR-2 Preprocessing** — convert → normalize → render each model to multi-view images (v1)
  or point clouds (stretch), storing outputs plus metadata.
- **FR-3 Weak labeling** — derive candidate labels from store metadata (categories, tags,
  titles). See [ml/ml.md](ml/ml.md#weak-label-policy).
- **FR-4 Manual labeling** — a web UI to confirm/correct labels in either a **browse view**
  (all items, paginated) or a **detail view** (single model). See [web/web.md](web/web.md).
- **FR-5 Training** — train a multi-view CNN classifier (v1); PointNet++ is a stretch goal.
- **FR-6 Training dashboard** — list all training runs and drill into a per-run detail view
  (metrics, curves, config, artifacts). See [web/web.md](web/web.md#training-dashboard).
- **FR-7 Evaluation** — score against two dev sets with per-class precision/recall and
  confusion matrices. See [ml/ml.md](ml/ml.md#evaluation).
- **FR-8 Auth & roles** — login required. **Normal users** can view; **admins** can view and
  correct annotations. See [web/web.md](web/web.md#auth--roles).
- **FR-9 Data upload** — admins can upload additional models into the pipeline. See
  [web/web.md](web/web.md#data-upload).

## Non-Functional Requirements

- **NFR-1 Cost** — total cloud spend ≤ ~$100 (target $50–80). See Constraints + Cost Guardrails.
- **NFR-2 Idempotency** — every worker is idempotent; reruns skip already-processed files.
- **NFR-3 Scalability** — preprocessing is embarrassingly parallel; scale by adding workers.
- **NFR-4 Reproducibility** — training runs record config, data snapshot/version, and metrics
  so results are reproducible.
- **NFR-5 Portability** — heavy work runs in the cloud (local laptop is underpowered); code is
  brought to the data to avoid egress.
- **NFR-6 Licensing compliance** — no redistribution of labeled data or trained models.
- **NFR-7 Security** — authenticated access; label-correction and upload restricted to admins.

## Constraints

- **Budget: ~$100 total cloud spend.** Estimated actual cost: $50–80.
  - Preprocessing (distributed CPU workers): $10–30
  - Storage (object storage, ~200–500 GB raw): $5–15/month
  - Training (spot GPU, e.g., T4/L4): $5–20 total
  - Eval/serving: negligible
- Local laptop is underpowered — all heavy work runs in the cloud.
- Data ingress to cloud is free; avoid egress (keep data in cloud, bring code to data).

## Architecture

Queue + workers pattern (embarrassingly parallel preprocessing):

```
[Model store API] → download workers → object storage (raw)
                         ↓ (queue)
                  conversion workers  (STL/OBJ/GLB/FBX → common format)
                         ↓ (queue)
                  normalize workers   (center, rescale, validate)
                         ↓ (queue)
                  render workers      (multi-view images or point clouds)
                         ↓
                  object storage (processed) + metadata DB
```

Component detail and coding standards live in the domain docs:

- **Compute / queue / API / storage / DB** → [server/server.md](server/server.md)
- **Labeling UI / dashboard / auth / upload** → [web/web.md](web/web.md)
- **Training / evaluation** → [ml/ml.md](ml/ml.md)

## Cost Guardrails

- Set billing alerts at $25 / $50 / $75
- Test the full pipeline end-to-end on ~100 models before scaling to tens of thousands
- Make all workers **idempotent** — reruns must skip already-processed files
- Use spot/preemptible instances where possible
- Delete raw files for models excluded from the dataset

## Milestones & v1 Scope

**v1 = milestones 1–8.** Milestone 9 is an explicit stretch goal (v2).

1. **Explore metadata** — pull category/tag distributions from the store API; choose final class list
2. **Pipeline skeleton** — queue + one worker type, running locally in Docker on ~100 models
3. **Cloud deployment** — workers on Cloud Run/Fargate, object storage, billing alerts
4. **Full ingestion** — 20–50k models downloaded and preprocessed
5. **Labeling frontend** — three.js viewer + label confirm/correct UI over weak labels
6. **Baseline training** — multi-view CNN on weak labels, spot GPU
7. **Evaluation** — both dev sets, confusion matrices, bias writeup
8. **Iterate** — active learning loop (hand-label low-confidence examples, retrain)
9. **(Stretch / v2)** PointNet++ comparison; inference demo endpoint

## Open Decisions

Kept open deliberately. Each lists the criteria to decide on when the time comes.

- [x] **Model store / API → Objaverse.** ~800k Sketchfab-sourced objects with tags/categories/
      titles for weak labels, distributed via an official Python package that pulls from a hosted
      mirror — no scraping, no bulk-download ToS friction. See
      [server/server.md](server/server.md#ingestion-source).
- [x] **Cloud provider → GCP.** Picked as the single ecosystem for its scale-to-zero serverless
      compute and zero-idle managed queue, which fit the bursty-preprocessing + $100-budget profile
      better than AWS. Component-by-component rationale in
      [server/server.md](server/server.md#cloud-platform-gcp).
- [x] **Queue technology → Pub/Sub (managed), push subscriptions.** Native at-least-once + retries +
      dead-letter at zero idle cost, vs. self-hosted Redis+Celery's standing cost/ops. One
      topic+subscription per stage, DLQ on each. See [server/server.md](server/server.md#queue).
- [x] **Class list → hybrid weak-labeling, 12 mid-level classes.** Locked roster (animal, food, car,
      chair, weapon, electronics, figure, lamp, aircraft, building, table, plant) — all clear the ≥300
      bar on the clean LVIS-merge seed, defined in `ml/taxonomy.py`. Expanded to the full ~798k via
      Sketchfab category+tag rules (pass 2, next). See
      [ml/ml.md](ml/ml.md#metadata-exploration-milestone-1).
- [x] **Database → PostgreSQL everywhere.** Managed Cloud SQL Postgres in cloud (smallest tier);
      Postgres in Docker locally (not SQLite) so dev shares prod's upsert + concurrency semantics —
      critical for the idempotency invariant (NFR-2) and parallel-worker scale (NFR-3). See
      [server/server.md](server/server.md#database).

## Development Workflow

- **Stage, don't commit.** Every change is staged (`git add`) so it can be reviewed as a diff.
  Do **not** commit until the user has reviewed the staged diff and explicitly approved. No
  auto-commits — this applies to code and docs alike.

## Coding Conventions (all languages)

Cross-cutting rules; language-specific standards live in the domain docs.

- **Self-documenting names.** No 1–3 character variable names — a name must be long enough to read
  as what it holds (`annotation`, not `a`; `class_name`, not `c`; `shard_count`, not `n`). This
  applies to locals, loop variables, and comprehension targets alike. The only exemptions are
  established domain terms/acronyms already used across the codebase (e.g. `uid`) — prefer even those
  spelled out when it doesn't hurt readability.

## Doc Maintenance

These docs are the source of truth for design. A Claude Code hook reminds you to update the
relevant domain doc whenever code in its area changes (see `.claude/settings.json`). Keep the
hub thin: cross-cutting facts here, domain detail in the domain docs.
