# Status

Milestone-by-milestone progress. **v1 = milestones 1–8, and all eight are done.** The design rationale
behind each item lives in the domain docs — [ml/ml.md](ml/ml.md),
[server/server.md](server/server.md), [web/web.md](web/web.md) — and the scope itself in
[CLAUDE.md](CLAUDE.md); this file is the score, plus [what has shipped since v1](#shipped-since-v1)
and the [post-v1 backlog](#post-v1-backlog) at the end, which is the single place remaining work is
tracked.

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

## ✅ Milestone 7 — evaluation

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
- [x] **Both dev sets exist.** The second is 984 independently annotated LVIS objects, ingested
      through the same pipeline and scored with `make evaluate RUN=n DEVSET=lvis` — detail under
      milestone 8, where the same ingestion serves both purposes.

## ✅ Milestone 8 — active learning

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
- [x] **Fixed the reshuffling itself.** A model's partition is now `sha256(f"{seed}:{uid}") mod 10000`,
      a pure function of its own uid, so editing a label can no longer move any *other* model across
      the split boundary ([ml.md](ml/ml.md#why-the-split-is-hashed-not-shuffled)). The old split was
      deterministic but not stable — one shared RNG shuffled the classes in order, so a class losing a
      member re-randomised every class after it alphabetically; measured, **one changed label out of
      3,600 moved 127 models across the test boundary**. Accepted cost: per-class proportions are now
      approximate, and **run 16 onwards gets a split unrelated to runs 14/15** — a one-time
      discontinuity. Runs 14 and 15 stay comparable to each other by scoring both on the intersection
      of their held-out sets.
- [x] **A second dev set — 1,000 LVIS-annotated objects ingested, 984 rendered and scored.** It lands
      here rather than under milestone 7 because it does double duty: it satisfies FR-7, *and*
      independent annotations are exactly what this loop is missing. A reviewer who has seen the
      classifier's guess cannot produce an unbiased correction — which is why the accuracy gain above
      carries a caveat — whereas LVIS labels were made without reference to this model at all. The
      gold labels stay in a CSV and never enter the `label` table, which makes the dev set
      structurally untrainable rather than untrainable by convention.
- [x] **It answers the question run 14 raised.** 5.8× the data changed nothing, so the ceiling sits
      upstream of training, and only independent labels separate a *label* ceiling from a
      *representation* limit. Run 15 scores **0.3730 accuracy / 0.3712 macro recall** on it
      (`evaluation 5`) against 0.4241 / 0.3401 on its own test split. Read macro recall across the
      two — on a balanced set accuracy *is* macro recall, and the fall in accuracy is the skew being
      removed, not the model getting worse. **Clean labels are worth ~3 points of macro recall**:
      real, and far too small to be the cap. `aircraft` scores 0.00 recall against 81 gold examples
      and `weapon` precision falls 0.74 → 0.49 once the skew is gone, so the tail collapse and the
      calibration problem are both confirmed on data the model cannot have gamed. **The shape-only
      renders are now the leading explanation for the ~0.45 plateau**
      ([ml.md](ml/ml.md#what-it-says-run-15-evaluation-5)).

## Shipped since v1

Work that landed after milestone 8 closed. Kept separate from the checklists above so those stay a
record of what v1 was, rather than being quietly rewritten as the project moves on. Newest first.

### Scoring a run from its own page (2026-08-09)

`make evaluate RUN=n` needed a checkout, credentials and a laptop willing to spend ~15 minutes on
CPU. The run's detail page now has an **Evaluate** button instead, admin-only, with a dev-set picker
covering all four (`test`, `val`, `train`, `lvis`).

- [x] `POST /training-runs/{id}/evaluations` submits a **Vertex job**, because the API image ships
      without torch or the ml package and could not score a run given any amount of time. The
      training image already contains `evaluate.py` — `ml/Dockerfile` copies every ml module — so the
      job is the same image with `containerSpec.command` overridden. No second image, no second set
      of IAM bindings.
- [x] **An evaluation is now visible in every state.** `evaluation` gained `status` and `error`, and
      `report` became nullable, so the row is written when scoring *starts* rather than when it
      succeeds: a job in flight reads `running` instead of showing nothing for minutes, and one that
      dies says why instead of never arriving. Migration `4cbd7fc5f228`; the status reuses the
      existing `trainingstatus` enum rather than adding a second one with identical values.
- [x] **The LVIS dev set is readable from a cloud job.** Its selection lives in a gitignored CSV, so a
      job with no checkout could not score it. `make devset-push` copies it to
      `processed/devsets/lvis.csv` and `load_dev_set` reads local-first, bucket-second. The copy is
      still not a `label` row, so the property that makes the dev set structurally untrainable is
      unchanged. ⚠️ Pushing is deliberately *not* re-selecting — see
      [ml.md](ml/ml.md#the-second-dev-set).
- [x] Refusals that used to cost ~15 minutes on a billed GPU happen at the request: an unknown run
      (404) and a run with no saved weights (409).

**Deploy note:** the training image is pinned by commit and now carries the evaluation status
handling, so this needs `make train-image` + a `TF_VAR_train_image` bump, not just an API roll. A
stale image would write evaluation rows that never leave `running`.

### The batch size no longer OOMs the GPU (2026-08-09)

The launch form's `batch_size` asked the GPU for **12× what it said** — a model's 12 rendered views
are folded into the batch by `ml/model.py`'s forward — and nothing bounded it. Three `training_run`
rows (18, 19, 20) died of `torch.OutOfMemoryError` ~45s in at batch 64, which is 768 images; they
were one Vertex job retrying itself three times, not three launches.

- [x] Bounded server-side at `MAX_IMAGES_PER_FORWARD = 512` images per pass (**42 models**), set
      below the measured failure rather than at it — 384 images trained fine, 768 died 148 MiB past
      the T4's 14.58 GiB, and how much room the allocator has left after fragmentation is not
      something a form can know.
- [x] The refusal names the hidden figure rather than the ceiling alone, because the number the admin
      never sees is the one that explains the limit.
- [x] The form states the arithmetic live ("asks the GPU for 384 images per pass"), carries a `max`,
      and blocks the submit with the reason. Both figures come from `GET /training-launch`, so the
      page holds no copy to drift from.

## Post-v1 backlog

**v1 is complete — milestones 1–8 are all closed.** Everything here is optional work beyond it, kept
in one place so it does not scatter across the domain docs. Ordered by evidence, not by appeal.

### What would actually move the model

The temptation is to reach for hyperparameters. The measurements say otherwise, and they are worth
restating because they are easy to forget:

- **Label noise is not the cap.** The [second dev set](#-milestone-8--active-learning) put a number
  on it: independently annotated labels are worth **~3 points of macro recall**. If labels were the
  binding constraint, perfect ones would have bought far more.
- **Training is exhausted.** Run 14 saw 5.8× the data change nothing (0.4611 → 0.4552), with val loss
  flat after epoch 1. That is not a model waiting for a better learning rate.

So the ceiling is in the *representation* or the *calibration*, in that order:

1. **Texture/material A/B — the leading candidate.** Renders are shape-only: `render.py` overrides
   every material with neutral grey, and `convert` exports PLY, which carries no UVs at all, so
   colour is gone two stages before rendering. Testing it means preserving textured geometry through
   convert + normalize and re-rendering a subset, then scoring that subset against the same models
   rendered shape-only. A few dollars of parallel Cloud Run. Most likely to help `food`, `plant` and
   `electronics`, where colour carries the signal that shape does not.
2. **Class weighting at full scale.** The calibration failure is now measured twice: small classes
   are precise but under-predicted, and `weapon` precision falls 0.74 → 0.49 the moment the dev set
   is balanced. The run 3/4 A/B lost its conclusion to the split-fraction defect, not to a null
   result, so this is unfinished rather than answered.
3. **Generic hyperparameter search — last.** Listed for completeness. Nothing measured points here.

### Operational

4. ~~**Deploy `pool_pre_ping`**~~ (`server/app/db.py`) — **DONE 2026-07-31**, revision
   `imagegenie-api-00011-m75`. Shipped alongside the collapsed `held_out` list on the run detail
   page; the database was already at Alembic head, so the deploy carried no migration.
5. **Batch the seed the way the replay is batched.** Publishing 1,000 uids at once overruns the
   download worker (maxScale 10 × one model per instance): Pub/Sub push gets 429s from Cloud Run,
   `max_delivery_attempts = 5` quarantines the message, and the job dead-letters **before the worker
   ever runs**. The 2026-07-30 dev-set seed landed 501 of 1,000 that way and needed
   `app.replay_dlq --max 250` in rounds to recover (496 → 750 → 947 → 984). The recovery logic is the
   fix; it just belongs at seed time.
6. **Push-level rejections leave no `dead_letter` row.** Failures are recorded by the worker at nack
   time — the only place the error text exists — so a message rejected *before* delivery is invisible
   in the admin dead-letter view. During the seed above, 499 quarantined models showed up nowhere in
   the database; only the Pub/Sub backlog metric knew. A real observability gap.
7. **16 models stuck in the convert/normalize/render DLQs** from the dev-set ingestion. The standing
   broken-mesh tail rather than anything transient — replaying them just re-fails. Fine to leave;
   worth a look only to characterise what breaks.

### Optional / housekeeping

8. **Run 16 on the hash-bucketed split**, for a clean post-fix baseline. Nothing requires it — runs
   14 and 15 stay interpretable via intersection scoring.
9. **Promote `common_test.py` into `ml/`** if cross-run intersection scoring recurs. Pattern:
   `load_run_model` both runs → `evaluate_samples` on the intersection of their `held_out` uids.
10. **Retire the `--limit` split-fraction defect** from the backlog after confirming it. It described
    fixed per-class 10/10 slicing, which PR #49 replaced with hash buckets outright, so it is very
    likely already gone. A glance, not a project.
11. **M9 / PointNet++ comparison** — the original stretch goal. Its inference-demo half already
    shipped into v1.
