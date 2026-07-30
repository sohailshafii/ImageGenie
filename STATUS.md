# Status

Milestone-by-milestone progress. **v1 = milestones 1–8.** The design rationale behind each item lives
in the domain docs — [ml/ml.md](ml/ml.md), [server/server.md](server/server.md),
[web/web.md](web/web.md) — and the scope itself in [CLAUDE.md](CLAUDE.md); this file is only the
score.

## ✅ Milestone 1 — metadata exploration

- [x] Category/tag distributions pulled from Objaverse (metadata only, no meshes)
- [x] 12-class roster locked: animal, food, car, chair, weapon, electronics, figure, lamp, aircraft,
      building, table, plant — all clear the ≥300-example bar
- [x] LVIS kept as a curated **gold set** for grading, not as the volume source

## ✅ Weak labeling (FR-3)

- [x] Category gate → keyword resolution → out-of-scope rescue (`ml/weak_label.py`)
- [x] Graded against LVIS gold: **57% coverage, ~0.91 precision, 0.52 recall** (`make evalweak`)
- [x] Tuned from measurements, not by eye — plant/food split, `building` confirm-required, rescue pass
- [x] `figure`/`animal` measured as unresolvable by keywords, and documented as a manual rule instead
      (`make evalboundary` — [ml.md](ml/ml.md#the-figureanimal-boundary))

## ✅ Milestone 2 — pipeline skeleton

- [x] Queue + download worker, verified end-to-end in Docker
- [x] Idempotent by construction (NFR-2): reruns skip already-processed files

## ✅ Milestone 3 — cloud deployment

- [x] Terraform for APIs, budget alerts, Cloud Storage, Pub/Sub, Cloud SQL, Cloud Run
- [x] Pipeline runs end-to-end on GCP

## ✅ Milestone 4 — full ingestion

- [x] Convert → normalize → render stages on Cloud Run (scale-to-zero, per-stage Pub/Sub push + DLQ)
- [x] 32k-uid labeled set seeded; **~12k models fully ingested and rendered**
- [x] Resilience tuning: 2–4 GiB per stage, one model per instance, in-worker retry + backoff
- [x] DLQ-replay tool to recover transient mirror failures (13k messages replayed)

## ✅ Milestone 5 — labeling frontend

React + TS + Vite on a FastAPI backend-for-frontend, deployed to Cloud Run as **one image on one
origin**.

- [x] The labeling loop end to end: sign in → browse rendered previews → open the normalized mesh in
      a three.js viewer → confirm or correct the label, attributed to the admin who changed it
- [x] Admin data upload (FR-9), rejecting FBX at the door rather than deep in the pipeline
- [x] Invite-gated signup, email verification (Resend), session cookies, CSRF, rate limiting
- [x] Weak-label and Objaverse-metadata backfills that populate the catalog
- [x] Sort-by-least-confidence and a keyboard sweep for fast review
- [x] Admin dead-letter view over recorded pipeline failures
- [x] Alembic migrations; `scripts/adopt_schema.sh` rebuilds the DB from the buckets on deploy

## ✅ Milestone 6 — baseline training

`ml/train.py` trains a multi-view CNN (resnet18 over 12 rendered views → pool → head) on the weak
labels, reading renders straight from the processed bucket.

- [x] Reproducible per-class stratified splits, per-epoch checkpoints, full NFR-4 bookkeeping
      (config + data snapshot + metrics recorded for every run)
- [x] Runs on a **Vertex AI T4**, proven end-to-end: Cloud SQL over the IAM connector, parallel GCS
      reads, spot *or* on-demand scheduling
- [x] Launchable from the CLI (`make train-cloud`) or the dashboard's **Start a run** page
- [x] Dashboard deployed: run list, cost curve (train/val loss + validation accuracy), per-class
      precision/recall, confusion matrix
- [x] **The full-set run — run 14:** 11,783 models × 4 epochs, on-demand T4, 1h53m, ~$1.40

## 🚧 Milestone 7 — evaluation

- [x] `make evaluate RUN=n` scores a finished run against the sealed **test** split, replaying the
      exact uids the run held out, and stores one report per (run, dev set)
- [x] Reported beside the run's own `val` numbers but kept visually separate, so the optimistic
      number is not mistaken for the honest one
- [x] **Run 14 scores 0.4484 accuracy / 0.336 macro recall** on its 1,173 held-out models
      (`evaluation 2`), or **0.4689 / 0.3472** once 24 of those test labels were corrected in
      milestone 8 (`evaluation 3`) — the same model against a less noisy yardstick
- [x] Classifier usable directly: predict a catalog model from its detail page, or upload any mesh at
      `/classify` and get a class back without ingesting it
- [x] **Bias writeup** ([ml.md](ml/ml.md#bias-analysis)) — class skew and tail collapse, a measured
      ~9% weak-label error ceiling, evidence that training has *converged* so the ceiling sits
      upstream of it, and the bias the pipeline itself adds by rendering shape only
- [ ] **FR-7 asks for two dev sets and only one exists.** The work is tracked under milestone 8
      below, where the same ingestion serves both purposes.

## 🚧 Milestone 8 — active learning

Queues the models where the classifier **disagrees** with the stored label, rather than the ones it is
least sure about: on a corpus with ~9% wrong labels, each disagreement is either a model error or a
label error, and only a human separates them.

- [x] `make review-queue RUN=n` — scores a split and writes the disagreements with a link per model
- [x] First pass: **74 disagreements judged** → 24 label errors, 27 model errors, 23 unlabelable
- [x] 24 corrections applied — **the corpus's first manual labels**; test accuracy 0.4484 → 0.4689,
      recorded with the caveat that this procedure can only ever move accuracy *up*
      ([ml.md](ml/ml.md#6-what-a-hand-review-of-74-disagreements-actually-found-milestone-8))
- [x] **Retrained on the corrected labels — run 15**, matching run 14's config field-for-field
      (4 epochs, on-demand T4, 123 min) so the labels are the only difference. It scores **0.4241
      accuracy / 0.3401 macro recall** on its own held-out split (`evaluation 4`).
- [x] **Learned that the before/after cannot be read off those two numbers, and why.** Correcting
      labels reshuffles the stratified split, so run 14 and run 15 held out **different sets — only
      29% overlap** (687 of run 15's test models were in run 14's *training* set). Compared naively,
      run 15 looks 4.5 points worse; scored on the 340 models **both** held out, it is 5.3 points
      *better* (0.4441 vs 0.3912). The split, not the model, was driving both readings. Neither
      direction is evidence: 24 relabels are 0.25% of the training set, far too few to move accuracy
      5 points, so this is run-to-run variance seen through two different lenses. **The lesson is
      that an active-learning loop cannot measure itself unless the evaluation set is frozen
      independently of the labels being corrected** — which is the strongest argument yet for the
      second dev set below.
- [ ] **A second dev set — ingest ~1,000 LVIS-annotated objects** through the existing pipeline. It
      lands here rather than under milestone 7 because it does double duty: it satisfies FR-7, *and*
      independent annotations are exactly what this loop is missing. A reviewer who has seen the
      classifier's guess cannot produce an unbiased correction — which is why the accuracy gain above
      carries a caveat — whereas LVIS labels were made without reference to this model at all. It
      also answers the question run 14 raised: 5.8× the data changed nothing, so the ceiling is
      upstream of training, and only independent labels separate a *label* ceiling from a
      *representation* limit. Scoped as a data run of a couple of dollars, mostly waiting.
