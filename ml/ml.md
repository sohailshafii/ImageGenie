# ML — ImageGenie

Representations, training, evaluation, and the labeling policy. See [../CLAUDE.md](../CLAUDE.md)
for the project hub.

## Representation

In ascending difficulty:

1. **Multi-view (START HERE):** render each object from ~12 angles, feed to a standard CNN
   (e.g., ResNet). Reuses mature 2D tooling; surprisingly strong baseline.
2. **Point clouds (stretch goal):** PointNet / PointNet++.
3. **Voxels:** skip (mostly historical).

- **Framework:** PyTorch (+ torchvision; Hugging Face for pretrained backbones).
- **Benchmarks for context:** ModelNet40, ShapeNet literature.

## Weak-Label Policy

Store metadata (categories, tags, titles) → thousands of free-but-noisy labels.

- Weak labels are the **bootstrap**, not the final truth. They exist to get training off the
  ground before hand-labeling.
- **Minimum bar for a class to be trainable on weak labels:** a class must clear a minimum
  support threshold before it's included — target **≥ a few hundred weakly-labeled examples per
  class** (align with the "avoid long-tail classes" rule). Set the exact cutoff after the
  metadata exploration in milestone 1; record it here once chosen.
- **Category selection:** pick 10–20 classes AFTER inspecting the metadata distribution; avoid
  long-tail classes with few examples.
- Keep weak (`source = weak`) and manual (`source = manual`) labels as distinct rows in the DB
  ([server.md](../server/server.md#database)) so weak-vs-corrected analysis stays possible.
- **Manual labels** come via the [labeling frontend](../web/web.md#labeling-ui), prioritized by model
  uncertainty (active learning — milestone 8).
- **Precedent:** Objaverse (~800k annotated objects from Sketchfab) shows this practice is accepted.

### Metadata Exploration (milestone 1)

`ml/explore_metadata.py` pulls category/tag distributions from Objaverse (metadata only — no meshes)
to choose the class list and fix the support threshold. Sources:

- **LVIS** (`--mode lvis`) — curated per-object categories; one small download.
- **Raw Sketchfab** (`--mode raw --shards N`) — the weak-label source (tags/categories), sampled by
  *whole metadata shard* (160 shards × ~5k objects). Sampling scattered uids forces downloading nearly
  every shard, so we sample whole shards instead.

Outputs CSVs + `summary.json` to `data/exploration/` (gitignored — derived data isn't redistributed,
NFR-6).

**Run** via the `Makefile` (targets wrap venv creation, deps, and — because macOS
framework-Python doesn't trust the system cert store — the `SSL_CERT_FILE` cert shim):

```
make setup                 # create .venv, install runtime + dev deps
make explore               # metadata exploration, MODE=lvis (default)
make explore MODE=both     # LVIS + sampled raw Sketchfab
```

**Findings:**

- **LVIS is too granular** — 1,156 categories over ~46k objects (~40 each); only `chair` (453) and
  `seashell` (371) clear the ≥300 bar. Clean but sparse: good as a curated eval set or a merge base,
  not a class list on its own.
- **Raw Sketchfab categories are coarse but high-volume** — ~18 top-level categories; on a 5k sample
  the object-like ones (extrapolated ×160 over ~800k) clear the bar with room to spare:
  `furniture-home`, `characters-creatures`, `animals-pets`, `cars-vehicles`, `weapons-military`,
  `electronics-gadgets`, `food-drink`, `nature-plants`. The rest (`architecture`, `art-abstract`,
  `cultural-heritage`, `science-technology`, `places-travel`, `people`…) are too abstract/mixed to be
  visual classes.
- **Tags are noisy** — dominated by tool/style tags (`lowpoly`, `blender`, `substancepainter`) and
  uploader batches (a `stair`/`staircase`/`staircon`/`pamir` cluster). Usable only with heavy curation.

**Class-list approach — hybrid (chosen).** Mid-level, visually-distinct classes. Labels come from two
passes:

1. **LVIS merge — clean seed + gold set.** Merge related fine LVIS categories into each class (e.g.
   `chair` + `folding_chair` + `highchair` → chair). Gives clean, curated labels for the ~46k
   LVIS-annotated objects and — crucially — a **gold set to tune and measure the weak-label rules**
   below (does the `chair` rule catch what LVIS independently calls chairs, without dragging in
   stools?).
2. **Sketchfab rules — volume.** For the full ~798k corpus, assign a class from raw metadata: the
   coarse **`categories` field as a pre-filter + disambiguator**, then **tags/title keywords** for the
   fine assignment. The category disambiguates polysemous keywords — *"jaguar"* is a car in
   `cars-vehicles` but an animal in `animals-pets`. **This rule-pass is the weak labeling (FR-3)** —
   deliberately noisy, corrected later via the [labeling frontend](../web/web.md#labeling-ui) (FR-4).

**Volume is driven by label-source coverage, not the class list.** LVIS covers only ~46k objects, so
LVIS-only labels cap there; the Sketchfab rules cover the full ~798k, which is where per-class volume
comes from. The class list only changes how those models are *distributed* across classes (broader
classes absorb more of the same corpus).

**Locked class list — 12 classes.** `ml/taxonomy.py` (`CLASS_TO_LVIS_CATEGORIES`) is the source of truth: a curated
map from each class to its exact LVIS category strings (hand-curated, not a keyword sweep — `bowl` is
not an animal, `spear`/`steak_knife` are not food). `ml/build_class_list.py` (`make classlist`) applies
it, counting **unique objects** per class (union of UIDs, no double-counting) and self-checking for
unknown keys + large unassigned categories. Latest run — all 12 clear the ≥300 bar *within LVIS alone*,
0 objects multi-class:

| class | objs | class | objs | class | objs |
|-------|-----:|-------|-----:|-------|-----:|
| animal | 3,003 | electronics | 1,170 | aircraft | 573 |
| food | 1,883 | weapon | 1,175 | building | 454 |
| car | 1,269 | figure | 853 | table | 411 |
| chair | 1,189 | lamp | 754 | plant | 384 |

These are the *LVIS-merged* counts (the clean ~46k subset) — a viability signal + gold set, **not** the
final weak-label volume, which comes from the Sketchfab pass. `plant`/`building`/`table` are thin here
(LVIS is object-centric) and lean on pass 2 for volume.

The 12 classes cover **13,118 / 46,207 LVIS objects (~28%)**; the other ~33k sit in 972 out-of-roster
categories (`seashell`, `mug`, `guitar`, `shoe`, …) — expected for a curated 12-class list, and a
non-issue since LVIS is the gold set, not the volume source.

**Support threshold: ≥ 300** weak-labeled examples/class (revisit per-class after the Sketchfab pass).
Resolves the class-list [open decision](../CLAUDE.md#open-decisions).

### Sketchfab weak labeling (pass 2, FR-3)

`ml/weak_label.py` (`make weaklabel [SHARDS=N]`) assigns a class per object from raw Sketchfab metadata,
built up in stages so each is measurable. It writes `weak_label_coverage.json` (the by-reason/per-class
tally) and `weak_labels.csv` (`uid, class, reason` for every labeled object) — the latter is the
**ingestion input**: the pipeline seeds download jobs from these uids (`server/app/seed.py`).

- **Stage 1 — category gate (done).** `taxonomy.SKETCHFAB_CATEGORY_TO_CLASSES` maps the 18 top-level
  Sketchfab categories to the candidate roster classes under each. Single-candidate categories
  (`weapons-military`→weapon, `architecture`→building) label directly; three are multi-candidate and
  deferred to keyword rules (`furniture-home`→chair/table/lamp, `cars-vehicles`→car/aircraft,
  `characters-creatures`→figure/animal); unmapped categories (abstract/mixed: `art-abstract`,
  `science-technology`, …) yield no label.
- **Stage 2 — keyword resolution (done).** `taxonomy.CLASS_TO_KEYWORDS` tag/title keywords pick one class
  within a multi-candidate set; the category gate having already narrowed candidates means homographs
  disambiguate for free (*"jaguar"* under `cars-vehicles` only scores car/aircraft, never animal). No
  clear winner → left ambiguous, never guessed. On a 5k shard: **19% category + 7% keyword = 26%
  labeled, 14% ambiguous, 60% out-of-scope**; all 12 classes populated, smallest ~15/shard (×160 ≈ 2.4k,
  clears the bar). Residual ambiguity is mostly generic metadata (`furniture`/tool tags) that names no
  sub-class — correctly left for manual labeling (FR-4).
- **Stage 3 — gold-set eval + tuning (in progress).** `ml/eval_weak_labels.py` (`make evalweak`)
  measures the weak labels against the LVIS gold set (objects with both a weak and a clean label) to
  get per-class precision/recall and drive keyword tuning (e.g. `seat` catching toilet seats, the
  figure/animal boundary). Landed so far: the gold-label lookup (`uid → roster class`, inverting
  `CLASS_TO_LVIS_CATEGORIES`, counts matching `build_class_list`) and weak-vs-gold **coverage** — on a
  5k shard, 325 of ~13k gold objects fall in the sample and the weak rules label only **39%** of them
  (rest ambiguous or out-of-scope), a recall ceiling the keyword rules alone can't lift. Per-class
  precision/recall (`per_class_metrics`) over 4 shards: **precision is high where the labeler commits**
  (car/chair/food/lamp 1.00, animal 0.96, weapon 0.95) — the conservative design working — while
  **recall stays low** (0.22–0.61) from that coverage ceiling. Weak spots to tune: **building (0.38)**
  and **plant (0.42)** precision (over-predicting non-members). The `confusion_matrix` shows *where*:
  **`building` is a false-positive magnet** (animal/chair/electronics/figure/food/lamp all bleed in —
  `architecture` gate too broad), and **`food → plant` is the single biggest confusion** (produce
  caught by plant rules). Both were category-gate FPs (single-candidate categories auto-committing), not
  keyword misfires.
- **Tuning — plant/food boundary (done).** Made `nature-plants` multi-candidate `[plant, food]` and
  added `food`/`plant` keyword lists so produce (apple, pumpkin, mushroom) resolves to food, flora to
  plant. Result (4 shards): `food → plant` **8 → 1**; plant precision **0.42 → 0.86**; food recall
  **0.26 → 0.47** (produce recovered) at ~no precision cost; other classes unchanged.
- **Tuning — building confirm-required (done).** `architecture` is a grab-bag (benches, statues,
  streetlights), so `building` is now in `taxonomy.CONFIRM_REQUIRED_CLASSES`: a class there is never
  auto-committed from its category alone — a keyword must confirm it (else the object is left
  ambiguous). Added building keywords. Result (4 shards): building-column FPs **~8 → 1**, building
  precision **0.38 → 0.75**; recall **0.21 → 0.12** (the accepted cost — real buildings without a
  keyword abstain). Building stays a weak, low-recall class (LVIS is object-sparse for structures);
  it leans on manual labeling (FR-4).
- **Tuning — out-of-scope rescue (done).** When no category maps, `label_object` now tries a keyword
  rescue over *all* keyworded classes (reason `rescue`) before giving up — recovering objects whose
  category is abstract/unmapped but whose title/tags name a class. Result (4 shards): gold coverage
  **40% → 52%** (+66 labels), recall up across nearly every class (food 0.47→0.73, building 0.12→0.33)
  at held blended precision (~0.93). The rescue has no category gate, so it leans entirely on
  keyword specificity — measured rescue precision ~0.88. Adding `weapon`/`electronics` keyword lists
  (kept conservative for the gateless rescue — no `mouse`→animal, `drone`→aircraft) made those two
  rescuable as well: electronics recall 0.33→0.58, weapon 0.50→0.72, lifting coverage to **56%** at
  ~0.94 blended precision. All 12 classes now rescuable.
- **Stable headline (`make evalweak SHARDS=20`, 1,851 gold objects):** gold coverage **57%**, blended
  precision **0.91**, recall **0.52**. Per-class precision is 0.78–1.00 except **figure (0.62)** — the
  figure↔animal boundary is genuinely fuzzy (teddy bears/creatures bleed both ways) and stays the
  weakest class; recall spans 0.22–0.73. (The per-tuning deltas above were measured at 4 shards, so
  small-sample; these 20-shard numbers are the reproducible reference.)

## Training

The baseline is a **multi-view CNN** (see [Representation](#representation)) trained on the 12-class
weak labels, reading the rendered views from the processed bucket. Per CLAUDE.md's M6 complexity
budget, the script starts as a plain loop and grows only when a result demands it — the one
non-negotiable is NFR-4 bookkeeping.

`ml/train.py` (`make train`) is built around that budget, in four small modules:

- **Model** (`ml/model.py`, `MultiViewCNN`) — the shared 2D `backbone` (resnet18/resnet50, its final
  fc swapped for `Identity` so it emits features) runs on each view; `forward` folds the views into
  the batch so the shared backbone sees them all, then regroups and pools across views (`view_pool`
  = `max`/`mean`) before a small head (`Linear→ReLU→Dropout→…→Linear`) maps to `num_classes`. Built
  from the run `Config` via `from_config`; unknown backbone/pool and a `feature_dim` that doesn't
  match the backbone are rejected up front.
- **Dataset** (`ml/dataset.py`, `MultiViewDataset`) — one item per model: its 12 views loaded via
  `view_keys` + `Storage.get_bytes`, decoded and ImageNet-normalized (the backbone is pretrained) to
  a `[num_views, 3, H, W]` tensor, plus the class index from the canonical `ROSTER`
  (`ml/taxonomy.py`, a sorted tuple so the index order is stable and recorded per run). Reads through
  the `Storage` abstraction, so the same code trains local and cloud (NFR-5).
- **Splits** (`ml/splits.py`, `stratified_split`) — the trainable set is partitioned **per class**
  into train/val/test (~80/10/10), deterministic from `Config.seed` (sorted → seeded shuffle →
  `floor` slice, train keeps the remainder), so a run reproduces (NFR-4); tiny classes fall back to
  train. The test split is held for [evaluation](#evaluation) (M7).
- **`Config` dataclass** — all hyperparameters, persisted verbatim to `training_run.config` so
  adding a knob needs no migration (config-over-code). Four groups: *Architecture* (`backbone`,
  `pretrained`, `num_views`, `view_pool`, `feature_dim`, `head_hidden_dims` — one int = nodes in a
  head layer, `dropout`, `num_classes`), *Optimization* (`epochs`, `batch_size`, `learning_rate`,
  `optimizer`, `momentum`, Adam `beta1`/`beta2`/`eps`, `seed`, `log_every`), *Regularization*
  (`weight_decay`, `label_smoothing`, `class_weighting` — see below), and *Runtime* (`device`
  — default `cpu`, the cloud config sets `cuda`; `num_workers`).
- **Class weighting** (`_build_loss`) — the roster is skewed ~7.7:1 (weapon 2134 … aircraft/table
  278), and under plain cross-entropy the rare classes barely move the average loss, so the model
  buys accuracy by favouring the head and the tail collapses. `class_weighting="balanced"` scales
  each class's loss contribution by `total / (num_classes * count)` — average-sized class ≈ 1.0, so
  the loss keeps its overall scale instead of silently acting as a learning-rate cut — making one
  rare-class mistake cost as much as several common-class ones. Counts come from the **training
  split only**; using the whole trainable set would leak val/test composition into training. A class
  absent from training keeps weight 1.0 (it is never a target, so the value is unused).
  **Caveat:** a weighted run's `loss`/`val_loss` are on a different scale from an unweighted run's —
  compare those two runs on `val_accuracy` and the per-class report, not on the cost curve.
- **`load_trainable_samples()` + `data_snapshot()`** — the trainable set is the current label per
  live model (manual-over-weak, as the labeling API resolves it) **that is also rendered** (joined to
  a done `rendered` artifact), so training never faults on an unrendered model. The snapshot records
  `{label_count, label_hash, as_of, filter, class_counts, splits, held_out}`; the `label_hash`
  (sha256 over the sorted `(uid, class)` pairs) identifies the set, so a changed hash flags data
  drift, and `held_out` names the val/test uids so [evaluation](#scoring-a-finished-run-m7--c1) can
  replay the partition rather than recompute one the labeled set has moved out from under.
- **Bookkeeping helpers** — `create_run` / `log_metric` / `finalize_run`, each committing on its own
  `session_scope` so the [dashboard](../web/web.md#training-dashboard) sees a **live** run with a
  growing loss curve. Written directly through a DB session (like the pipeline workers); the API only
  reads these rows. `log_every` throttles the `training_metric` writes — a point every N steps,
  always keeping each epoch's last step so its `val_loss` is never dropped.
- **`run_training()`** — the real epoch/step loop: cross-entropy over the configured optimizer,
  per-step train loss logged (throttled), and once per epoch the val split is evaluated for loss and
  accuracy — both persisted onto the epoch's last step, so the dashboard shows the train/val gap
  and the accuracy curve. Accuracy is stored rather than derived because on a ~7.7:1 skewed
  corpus a falling loss can hide a model that has collapsed onto the majority class; the
  accuracy sitting flat at that rate is what makes it visible. Weights are
  checkpointed to `processed/models/{run_id}.pt` after every epoch (overwriting), so a spot
  preemption keeps the latest epoch; the key becomes the run's `weights_uri` on success.
- **`main()`** — load samples → split → snapshot → `create_run` → train → `finalize_run(completed,
  weights_uri)`; an empty trainable set exits early, and any exception marks the run `failed` (so it
  never lingers as `running`) and re-raises. A short flag list overrides `Config` — `--device`,
  `--num-workers`, `--epochs`, `--batch-size`, `--learning-rate`, plus `--limit` and `--notes`.
  Deliberately short: the knobs a *run* varies (where it runs, how big, how much data), not every
  hyperparameter — the rest stay `Config` defaults edited in code. Each defaults to `None`, so an
  unset flag leaves the `Config` default alone rather than overwriting it. Whatever they resolve to
  is what gets recorded, so a cloud run is as reproducible as a local one.
- **`--limit N`** takes a seeded random subset — the cost guardrail for a first cloud run: prove the
  wiring on a few hundred models before paying for the full set. It is **proportional, not
  class-balanced**, so a small run rehearses the real (~7.7:1 skewed) distribution rather than an
  easier balanced version of it, and the snapshot records `limit` so a subset run is never mistaken
  for a full one when comparing `label_count`s.

### Running in the cloud (M6 chunk G)

```
make train-image                     # build + push the CUDA image (Cloud Build)
make train-cloud LIMIT=500           # a first, small, paid run
make train-cloud                     # the full trainable set
make train-cloud ARGS='--epochs 5'   # extra flags straight through
```

⚠️ **The image is tagged by commit, and `train-cloud` refuses to submit without a matching one.**
Two preflight checks: uncommitted changes under `ml/` or `server/app` abort the submit, and so does
a missing image for the current commit. This is not tidiness — it is the fix for a real, expensive
failure. A first attempt was submitted against a `:latest` image built *before* the CLI flags
existed; the old entrypoint ignored `--limit`, `--epochs` and `--device` entirely and began training
the full 11,783-model set on **CPU** on a paid GPU node, looking like a healthy run the whole time.
A job runs unattended for hours, so a code/image mismatch is a silent failure, not a slow one.

- **The training image is separate from the worker image** (`ml/Dockerfile`, `ml/requirements-train.txt`).
  Training reads PNGs and writes rows, so it needs none of the mesh stack, none of the web layer, and
  neither Pub/Sub nor objaverse. **`torch`/`torchvision` are deliberately absent from the
  requirements**: they come from the CUDA base image, and re-installing them would pull the CPU
  wheels over the top — a GPU-less job on a GPU being paid for. Built by Cloud Build, since the base
  is multi-GB and the dev host is arm64.
- **`ml/vertex_job.yaml`** is the job spec: one `n1-standard-8` + one **T4**, `scheduling.strategy:
  SPOT`, running as the `imagegenie-trainer` service account. Spot is what keeps the training line
  inside NFR-1's $5–20; a preemption costs the run, not the work, since weights are checkpointed to
  the processed bucket after every epoch.
- **Reaching the database.** Cloud Run mounts a Unix socket for Cloud SQL; Vertex has no equivalent,
  and its egress IP is dynamic so authorized networks cannot cover it. The job sets
  `IMAGEGENIE_CLOUDSQL_INSTANCE` and dials through the Cloud SQL connector over IAM instead. The URL
  itself arrives as `IMAGEGENIE_DATABASE_URL_SECRET` — a secret *name*, fetched at startup — because
  everything in a job's env block is visible in its metadata and the URL carries the password. See
  [server.md](../server/server.md#training-gpu).
- ⚠️ **`num_workers > 0` requires the picklable-storage fix.** An epoch reads ~141k views, so
  single-threaded loading would leave the GPU waiting on I/O — but a `google.cloud.storage.Client`
  cannot cross a process boundary by either route (spawn pickles it and it refuses; fork shares its
  HTTPS connection pool). `GcsStorage` therefore pickles as its bucket name and rebuilds the client
  per process; see [server.md](../server/server.md#object-storage).

Run it with `make train`, which sets `PYTHONPATH=server` so the DB layer (`app.db`, `app.models`)
imports; no cert shim, since training only touches Postgres. `ml/smoke_train.py` (`make smoke-train`)
seeds a small class-separable dataset and runs the loop end to end on CPU — a repeatable check that it
learns and the bookkeeping/checkpoint land, without a GPU or the real renders. Every finished run now
writes its own dev-set [report](#dev-set-report-b4) into `metrics`; that blob is null only for a run
still training, or one that failed before the end. The reproducibility schema this writes to is detailed
under [Coding Standards](#coding-standards-ml).

## Dataset Splits

Resolves the dev-set-percentage TODO.

- **Train / dev(val) / test = ~80 / 10 / 10** of the own labeled data, stratified by class so
  every class appears in every split. Small classes may need a fixed minimum count per split
  rather than a strict percentage.
- Both dev sets below are intentionally small — a few hundred to ~2k examples is statistically
  sufficient for 10–20 classes (ModelNet40's test set is only ~2.5k).
- Splits are versioned so a `training_run` can reference exactly which data it used (NFR-4
  reproducibility).

## Evaluation

Two dev sets:

1. **Held-out split from own labeled data** → measures the model itself.
2. **Objaverse slice** → measures generalization / domain gap.
   - Requires mapping own taxonomy onto Objaverse annotations.
   - Expect distribution shift (different artists, styles, mesh quality) — analyze it explicitly
     rather than treating it as noise.

**MVP scope: (1) only** (decided 2026-07-27). The `evaluation` table is keyed by dev set precisely so
(2) can be added later without a migration or a second code path. What is being given up is worth
stating plainly rather than discovering later: every label in (1) comes from the *same* weak-labeling
pipeline as the training data, so the test split inherits its biases — including the measured 0.62
precision on `figure`. A model can therefore score well by faithfully reproducing the weak labeler's
mistakes, and (1) alone cannot tell that apart from being right. It measures generalization to unseen
**objects**; it does not measure whether the labels themselves are correct. If a cheap version of (2)
is wanted, the **LVIS gold set** from milestone 1 is already independently annotated and already has
tooling (`ml/eval_weak_labels.py`) — a far smaller job than the Objaverse-slice mapping above.

### Metrics

- **Per-class precision and recall** on both dev sets.
- **Confusion matrix** (resolves the confusion-matrix TODO): an N×N table for the N classes where
  entry (i, j) = the number of examples whose **true** class is i that the model **predicted** as
  j. The diagonal is correct predictions; off-diagonal entries show which classes get confused
  for which. Report one per dev set. It's the primary tool for the bias analysis below.

### Dev-set report (B4)

`ml/metrics.py` computes the report every run stores in `training_run.metrics`: confusion matrix,
per-class precision/recall/F1/support, and macro averages. Pure functions over class indices — no
torch, no sklearn, no I/O — so the same code serves the end-of-run report and the M7 evaluation
over both dev sets.

- **Scored on `val`, deliberately not `test`.** Training consults val every epoch anyway, whereas
  test is held back for the evaluation below; scoring test at the end of every run would erode it
  through repeated peeking long before M7 looked at it.
- **Computed once, after the final epoch** — it summarises the finished run, so doing it per epoch
  would buy nothing but an extra forward pass over val.
- **Undefined is not zero.** A class the model never predicted has *undefined* precision; reporting
  0.0 would claim it predicted that class and got them all wrong. Those return `None`, and macro
  averages skip them rather than being dragged toward zero.
- **Macro, not micro.** Micro-averaging collapses to plain accuracy, which this corpus's ~7.7:1 skew
  lets `weapon` dominate. Macro weights the 278-example `aircraft` like the 2,134-example `weapon`,
  which is what makes a collapsed model look as bad as it is — 80% majority class answered entirely
  with the majority label scores 0.80 accuracy against 0.33 macro recall.

### Scoring a finished run (M7 / C1)

```
make evaluate RUN=4              # the held-out test split
make evaluate RUN=4 SPLIT=val    # re-score val, e.g. to compare two methods
```

`ml/infer.py` rebuilds the model and `ml/evaluate.py` scores it, storing one `evaluation` row per
(run, dev set) — see [server.md](../server/server.md#database). Separate from training on purpose:
the trainer reports on `val`, which it consults every epoch and therefore cannot score honestly,
while `test` exists precisely so one number survives that nothing steered against. Keeping it a
distinct command is what stops "evaluate the model" becoming another training-time metric.

- **A run's stored config is the source of truth, never `Config`'s current defaults.** Backbone,
  view pooling and head shape decide the shape of the saved tensors, so weights only load back into
  the architecture that produced them. `rebuild_config` lets stored values win and fills only what a
  run predates (runs 2 and 3 have no `class_weighting`), keeping keys this checkout has since
  dropped. Filling a missing *architecture* key is an unguarded guess on purpose: a wrong one fails
  loudly in `load_state_dict` on a shape mismatch, which beats predicting from the wrong model. It
  is also why architecture is absent from the [launch form](../web/web.md#starting-a-training-run).
- **The partition is replayed from the run, not recomputed.** `data_snapshot` records the uids it
  held out (`held_out.val` / `held_out.test`). Recomputing *is* deterministic given the same samples
  and the run's own seed — but the partition is a function of the whole sample set, so one label
  added or corrected reshuffles every class and moves models between train, val and test, and the
  resulting number looks no different for being wrong. Writing the answer down is what makes an
  evaluation reproducible rather than merely deterministic. Only val and test are stored: train is
  large and nothing evaluates against it, which keeps the blob to ~2,300 uids at full scale.
  - **Labels come from the current database, not from training time** — only the uids are replayed.
    Scoring asks "is the model right?", and a corrected label answers that better than the one the
    run trained against. It is also what makes the M8 loop legible: hand-correct, re-score the same
    models, see the difference.
  - **A recorded model that has left the trainable set** (soft-deleted, unlabeled, unrendered) is
    skipped rather than fatal, and the count is printed — a shrinking dev set changes what the
    numbers mean.
  - **Runs predating the field** (2 through 4), and `train`, which is never recorded, fall back to
    recomputation, warning when the labels have moved since. Either way the report records the
    `label_hash` it was scored under.
  - **A `--limit` run's subsample is reproduced before splitting.** Such a run held out a split of
    its *subset*, so splitting the full trainable set puts models it trained on into its own test
    set — measured on run 4, 141 of 1,173 recomputed `test` models were ones the run had trained on.
    `subsample` is seeded and public for exactly this reason.
- **An empty split is refused**, rather than reporting metrics over zero samples — which would
  render on the dashboard as a real-looking result.
- **`classify_model` returns the whole roster ranked**, not just the top class: a single label hides
  a near-tie between `figure` and `animal`, where the model is lucky rather than right.

### Bias Analysis

- Per-class precision/recall + confusion matrices on both dev sets.
- **Key question:** which categories do metadata-derived weak labels systematically corrupt?
  Compare weak-label-trained vs. hand-label-corrected performance **per class** — a class that
  improves a lot after manual correction is one the weak labels were poisoning.

## Coding Standards (ML)

- **Language/framework:** Python 3.11+, PyTorch. Type hints on public functions.
- **Reproducibility (NFR-4):** every run records config, data snapshot, and metrics, persisted to the
  `training_run` entity ([server.md](../server/server.md#database)) so the
  [dashboard](../web/web.md#training-dashboard) can show them. The schema maps NFR-4's three pillars to
  three JSONB blobs — `config` (hyperparameters), `data_snapshot` (which labels the run trained on:
  count, content hash, as-of time, filter), and `metrics` (dev-set evaluation) — chosen over typed
  columns so a new hyperparameter or metric needs no migration. The per-step loss curve lives in a
  sibling `training_metric` table (`(run_id, step)` → `loss`, nullable `val_loss` for the train/val gap
  that reveals variance). The **training script writes these rows directly through a DB session** (like
  the pipeline workers); the API only *reads* them, so there are no training write endpoints.
- **Config over code:** hyperparameters in config files, not hardcoded in scripts.
- **Data loading:** stream renders/point clouds from object storage; never assume the full
  dataset fits in memory or on the local disk.
- **Cost:** train on spot/preemptible GPU; checkpoint often so a preemption doesn't lose the run.
- **Evaluation code is shared:** the same metric functions produce the numbers for both dev sets
  and the dashboard — no re-implementations that can drift.
- **Formatting/lint:** Ruff; no unformatted code committed.
