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
- **Manual labels** come via the [labeling frontend](../web/web.md#labeling-ui), prioritized by
  classifier/label **disagreement** rather than raw uncertainty (active learning — milestone 8; see
  [the review queue](#the-review-queue-milestone-8)).
- **Ambiguous boundaries are written down, not left to the labeler's judgement on the day** — an
  unwritten rule gets applied differently by different people, and differently by the same person a
  week apart, so the labels disagree with each other. A rule that is arguable but consistent beats
  that. The roster has exactly one such boundary:
  [figure vs animal](#the-figureanimal-boundary), resolved by *stance* — a biped with arms is
  `figure` whatever its head.
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
- **Boundary measurement.** `ml/eval_figure_animal.py` (`make evalboundary [SHARDS=N]`) answers a
  narrower question than `evalweak`: for the one ambiguous class pair, could a keyword rule resolve it
  at all? Separate script because the answer needs the *ambiguous population* rather than per-class
  precision — and because its two headline numbers (reach, abstention) are questions about the
  labeler's behaviour rather than its correctness, so they are measured over every gated object
  instead of the gold intersection `evalweak` is limited to. See
  [the figure/animal boundary](#the-figureanimal-boundary).
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
- **Not tuned — the figure/animal boundary, deliberately.** See below for why keywords cannot fix it.

### The figure/animal boundary

**The rule: stance decides, not the head.** A bipedal thing with arms is `figure` whatever its head —
a cat-person, a fox in a T-pose and a robot are all `figure` — while `animal` means the animal body
plan: quadrupeds, birds, fish, insects, dinosaurs. Chosen because it matches what the trained model
already learned, making the disagreements cheap label fixes rather than a boundary it has to relearn.
The LVIS merges in `ml/taxonomy.py` already followed it (`teddy_bear`, `mascot`, `puppet` are
`figure`), and it is now written there so a labeling session doesn't have to re-derive it.

This is an **FR-4 manual-labeling rule, not an FR-3 keyword rule** — and that is measured, not
assumed. It is the roster's one genuinely ambiguous boundary, the weakest weak-label class (0.62
precision, above) and the trained model's largest confusion pair (81 of 647 disagreements —
[section 6](#6-what-a-hand-review-of-74-disagreements-actually-found-milestone-8)), so the obvious
move is a "stance outranks species" precedence rule in `CLASS_TO_KEYWORDS`. Two measurements say not
to bother. Reproduce both with **`make evalboundary SHARDS=24`** (`ml/eval_figure_animal.py`); every
number below is from that run — 120,000 objects, 7,818 of them (6.5%) gating to `{figure, animal}`.

- **Reach — 4.3%.** A precedence rule reorders a *contested* decision, so it can only touch objects
  matching keywords from both `CLASS_TO_KEYWORDS["figure"]` and `CLASS_TO_KEYWORDS["animal"]`. Of
  those 7,818 objects, **58.4% match neither keyword list** (already ambiguous, so there is nothing to
  reorder), 23.8% match figure's alone and 13.4% animal's alone (already decided). That leaves
  **339, or 4.3%**, matching both. The ceiling holds however good the stance vocabulary is, and it
  needs no gold labels to establish, which is what makes it the load-bearing argument here.
- **Validity — the signal points the wrong way.** Over the 143 objects in that gate carrying an LVIS
  gold label, `character` — the one token that looks like stance — sits on **11 gold animals against 2
  gold figures** (figure-share 0.15): Sketchfab users tag quadrupeds "character" just as readily, so
  it means *game asset*, not *biped*. No token reaches a figure-share above 0.33 at ≥8 gold objects,
  and the most "discriminative" ones are modelling-tool tags (`zbrush`, `substancepainter`, `blender`)
  — milestone 1's noisy-tags finding again. An explicit stance vocabulary barely exists in the corpus:
  against `character`'s 1,089 occurrences, `humanoid` has 38, `anthro` 32, `mascot` 11, `biped` 5,
  `fursona` 3.

  *Limitation:* only 24 of those 143 gold objects are `figure`, so "no figure-discriminative token
  exists" is suggestive rather than settled — a token could be missed for want of gold examples. The
  reach ceiling does not depend on it.

**The dominant failure here is abstention, not mislabeling** — which the 0.62 precision hides. The
labeler leaves **4,819 of the 7,818 gated objects unlabeled: 61.6%**. That figure needs no gold set —
"did the labeler answer?" is not a question about correctness — so unlike the precision numbers it is
measured over the whole gated population rather than the 143-object gold slice, where the rate is a
comparable 56.6%. It splits into 4,565 objects where neither keyword list fires and 254 where both
fire and tie. On the gold slice, where correctness *can* be checked, the 143 objects break down as 81
silent, 55 correct and only **7 outright wrong** (4 animal→figure, 3 figure→animal) — so when the
labeler does commit here it is right **88.7%** of the time (55 of 62). That is the conservative design
working exactly as intended: `resolve_by_keywords` returns `None` on a tie or a zero score rather than
guessing. This class is not bad at deciding, it declines to decide.

The 10 objects it called `figure` in this gate were 6 gold figures — **0.60 precision**, near enough
to the 0.62 corpus-wide headline to suggest `figure`'s precision problem is largely *generated* by
this one gate's animal→figure leakage rather than spread across the corpus.

So nearly two-thirds of `characters-creatures` never enters training at all, and the class's real
problem is silence rather than error. Sharper keywords do not fix silence — the rule that would
break the ties reaches 4.3% of the population, as above. Coverage by hand (FR-4) does.

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
- **Splits** (`ml/splits.py`, `stratified_split`) — every model is assigned to train/val/test by
  **hashing its uid with the seed** into one of 10,000 buckets (`<1000` test, `<2000` val, else
  train), so the partition is ~80/10/10 and reproducible from `Config.seed` (NFR-4). The test split
  is held for [evaluation](#evaluation) (M7). See [why hashing](#why-the-split-is-hashed-not-shuffled)
  — it is the one design choice here that a previous version got wrong in a way that silently
  invalidated a comparison.
  - **NOT IMPLEMENTED: the fractions are fixed at 10/10 regardless of dataset size.** The right
    proportions depend on how much data there is — a small set needs a *larger* held-out share (or
    k-fold cross-validation) to measure anything stably, while at hundreds of thousands of examples
    98/1/1 leaves a dev set that is still plenty. A constant 80/10/10 is wrong at both ends.
    **This already cost us a result:** a `--limit 2000` run leaves 193 val models — about 4 per class
    for the smallest — and the class-weighting A/B (runs 3 vs 4) came back inconclusive precisely
    because tail movement was unresolvable at that size. At the full 11,783 the same fractions give
    1,173, which is fine, so the defect only bites on limited runs. Fixing it means scaling
    `val_fraction`/`test_fraction` with `len(samples)` — or at minimum warning when a split leaves
    fewer than ~20 models in the smallest class.

#### Why the split is hashed, not shuffled

The first implementation partitioned each class with **one shared seeded RNG** — sort the class's
uids, shuffle, take the first 10% as test — which is perfectly *deterministic* (same input, same
output) but not *stable*: a slightly different input gives a wildly different output. Those are not
the same property, and only the second one lets two runs be compared.

Changing one label breaks it twice over. The model leaves one class list and joins another, and both
lists change length — so each shuffles differently, **and** the shorter list consumes one fewer draw
from the shared RNG, which shifts the stream for every class after it in sort order. Classes whose
labels nobody touched get re-randomised. Measured on a 12-class toy corpus: **one changed label out
of 3,600 moved 127 models in or out of the test split.**

That is what broke the milestone-8 loop. Run 14 and run 15 straddled a 24-label correction pass and
shared only **29%** of their test sets, with 687 of run 15's test models sitting in run 14's
*training* set. Scored on their own splits the retrain looked 4.5 points worse; scored on the 340
models both held out it looked 5.3 points better. Same two models, opposite conclusions, from nothing
but which models each was asked about — and both readings look like results.

So a model's bucket is now `sha256(f"{seed}:{uid}")` mod 10,000 and depends on **nothing else** —
not the labels, not the corpus size, not any other model. Correcting a label moves a model between
classes but never between train and test, so runs across a correction pass are comparable by
construction rather than by remembering to check.

- **sha256, not the builtin `hash()`** — the latter is randomised per process unless `PYTHONHASHSEED`
  is pinned, so a run's partition would depend on the interpreter that produced it. That is
  reproducibility which evaporates silently on the next process, which is worse than none.
- **The cost is exact per-class proportions.** Each class now lands *near* 80/10/10 rather than on
  it. At ~300+ members per roster class the drift is a handful of models, every run records its
  actual sizes, and stratification survives where it matters: the hash is independent of class, so
  each class enters each partition at the same expected rate.
- **Runs 2 through 4 predate this** and recorded no `held_out` either, so recomputing their split
  gives a partition that is not theirs under *any* label state. Both `ml/evaluate.py` and
  `ml/review_queue.py` now warn unconditionally on that path rather than only when the `label_hash`
  moved — an unchanged hash says the data is the same, not that the partition is.
- **Runs 5 onward are unaffected**: they recorded `held_out`, and replaying recorded uids never
  consults the split function at all.
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
    run trained against. It is also what makes re-scoring *the same run* legible after a correction
    pass: same uids, better labels, honest delta (run 14 went 0.4484 → 0.4689 that way).
  - ⚠️ **Replay does not make two *different* runs comparable, and the M8 retrain proved it.**
    Replaying protects a run's own partition, but a run trained *after* a correction pass computed
    its split from the corrected labels — so it held out a different set. Run 14 and run 15 overlap
    on only **29%** of their test models; 687 of run 15's were in run 14's *training* set. Scored on
    their own splits run 15 looks 4.5 points worse (0.4241 vs 0.4689); scored on the 340 models both
    held out it is 5.3 points better (0.4441 vs 0.3912). **The same pair of models, opposite
    conclusions, purely from which models each was asked about.** Neither reading is evidence about
    the corrections — 24 relabels are 0.25% of the training set — but both look like results.
    Comparing two runs across a label change means scoring both on the intersection of their held-out
    sets, or freezing an evaluation set that the correction pass never touches (see
    [the second dev set](#follow-up-a-real-second-dev-set)).
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
- **The score is printed before it is stored.** Scoring is minutes of compute over thousands of blob
  reads; storing is one INSERT. Found the hard way: a 15-minute run over the 1,173-model `test` split
  finished and then lost its report to `server closed the connection unexpectedly`, because the
  pooled connection had gone stale while nothing touched SQL. The engine now pre-pings
  (server.md#database), and this ordering means a failed write costs the row, not the measurement.
- **`classify_model` returns the whole roster ranked**, not just the top class: a single label hides
  a near-tie between `figure` and `animal`, where the model is lucky rather than right.

### The Review Queue (milestone 8)

```
make review-queue RUN=14                      # the run's own held-out test split
make review-queue RUN=14 SPLIT=val LIMIT=100  # the 100 most confident disagreements
```

`ml/review_queue.py` scores a finished run against a split and writes
`data/m8/review_queue.csv` — one row per model where the classifier and the stored label
**disagree**, each with a link to the model's page and two blank columns for the reviewer's proposed
label and verdict.

- **Disagreement, not low confidence.** The textbook active-learning queue is the least-confident
  models, and the browse UI already sorts that way. Low confidence finds models the classifier is
  unsure about; disagreement finds models where somebody is *demonstrably* wrong — and on a corpus
  whose labels are ~9% wrong ([bias analysis](#bias-analysis)), each disagreement is either model
  error or label error, which is exactly the distinction the analysis needs a human for.
- **Most-confident disagreements first.** Those are where the classifier commits hardest against the
  label, so they are the most informative to adjudicate and the least likely to be a coin-flip.
- **Scoring goes through the dataset and a DataLoader** (`infer.rank_samples`), like evaluation does
  — not `classify_model` per model. A model is twelve separate view reads from the bucket, so
  scoring one at a time serializes them all: over 2 s/model measured against GCS, dominated by read
  latency rather than the forward pass. Batching with worker processes is the difference between
  ~40 minutes and ~15 for the 1,173-model `test` split. `classify_model` stays the right tool for
  the single-model prediction endpoint, and the wrong one for a whole split.
- **Labeling inside `test` sharpens the yardstick, not the model.** With ~9% of labels wrong, the
  held-out score is measured against a noisy ruler; correcting those labels makes the number mean
  what it says immediately. Correcting *training* labels only helps with volume, so a couple of
  review passes are not expected to move aggregate metrics — that is the design, not a failure.

### Bias Analysis

**Where the numbers come from.** The headline figures are **run 14** — the full trainable set,
11,783 models × 4 epochs on an on-demand T4 (1h53m) — and its held-out score, `evaluation 2`, over
the 1,173 `test` models it recorded as held out. Runs 3 and 4 (2,000 × 3, plain cross-entropy and
balanced class weighting) are cited where a comparison is the point. Each figure names its run.

#### 1. The class distribution is skewed 7.7:1, and the tail collapses

weapon 2,134 · animal 1,687 · figure 1,594 · building 1,572 · electronics 1,279 · food 800 ·
car 693 · plant 572 · lamp 472 · chair 424 · table 278 · aircraft 278.

A majority-class baseline scores ~18%, so **top-line accuracy is nearly uninformative here** — which
is why macro averages sit beside it everywhere in this project. The gap between the two *is* the
finding: run 14 scores **0.4484 accuracy against 0.336 macro recall** on `test` (macro precision
0.521, macro F1 0.385, n=1,173).

Per-class on `test` (run 14, precision/recall/support):

| class | prec | recall | support | | class | prec | recall | support |
|---|---:|---:|---:|---|---|---:|---:|---:|
| weapon | 0.57 | 0.67 | 213 | | food | 0.36 | 0.31 | 80 |
| animal | 0.53 | 0.37 | 168 | | car | **0.87** | 0.39 | 69 |
| figure | 0.35 | 0.60 | 159 | | plant | 0.30 | 0.25 | 57 |
| building | 0.46 | 0.70 | 157 | | lamp | 0.36 | 0.19 | 47 |
| electronics | 0.28 | 0.24 | 127 | | chair | **0.90** | 0.21 | 42 |
| | | | | | table | **0.75** | 0.11 | 27 |
| | | | | | aircraft | **—** | 0.00 | 27 |

**`aircraft` is never predicted at all** — 27 held-out models, zero predictions, undefined precision.

**The most actionable pattern is the precision/recall split on the small classes:** `chair`
0.90/0.21, `table` 0.75/0.11, `car` 0.87/0.39. When the model says "chair" it is almost always
right; it just says it far too rarely. That is not a model unable to recognise chairs — it is a
decision rule biased toward the big classes by the 7.7:1 skew, and it is the strongest evidence that
part of the tail problem is **calibration, not capability**.

Which class absorbs the others is unstable: here `figure` over-predicts (0.35 precision on 0.60
recall), while in run 3 `animal` played that role (0.31/0.77, taking 14 of 28 figures). That one
class does is not.

#### 2. Class weighting did not fix it (and that is informative)

Run 4 repeated run 3 with `class_weighting=balanced`, identical data (same `label_hash`) and seed.
Macro recall — the thing weighting exists to move — went **0.334 → 0.337**, inside the noise on a
193-sample val split, while accuracy and macro precision each fell ~9 points. It redistributed
rather than repaired: `table` went 0.00/0.00 → 0.11/0.75 (predicting it often, mostly wrongly) while
`plant` went 0.33/0.11 → 0.00/0.00.

Read as **inconclusive rather than settled**: run 4 trailed run 3 at every epoch and ended with a
higher val loss that was still falling, so weighting made optimization harder and three epochs never
recovered. The deeper problem is that a 193-sample val split cannot resolve tail effects at all —
`aircraft` has 3 samples in it.

**This is now the experiment most worth rerunning.** Run 14 gives a 1,173-model val split, which can
resolve tail movement, and (1) supplies a specific reason to expect weighting to help that was not
visible before: the small classes are *precise but under-predicted* (`chair` 0.90/0.21), which is the
calibration failure class weighting exists to correct. One full-set run with
`class_weighting=balanced`, compared against run 14, would settle it.

#### 3. The labels themselves impose a ceiling — measured at ~9%

The weak labeler disagrees with the independently-annotated LVIS gold labels on **8.8%** of the 475
models where both exist (42 of 475). That independently confirms FR-3's 0.91 blended precision, this
time on the actually-ingested corpus rather than a sample.

It is **not uniform across classes**, which is what makes it bias rather than noise: FR-3 measured
per-class weak-label precision of 0.78–1.00 for most classes but **0.62 for `figure`**, whose
boundary with `animal` is genuinely fuzzy. In run 14 `figure` is precisely the class that
over-predicts — 0.35 precision on 0.60 recall, absorbing others — so the class whose labels are
least trustworthy is also the one whose behaviour is hardest to interpret: we cannot tell how much
of that 0.35 is the model being wrong and how much is the *label* being wrong.

A model that perfectly reproduced these labels would still look wrong wherever they are wrong, so
**some fraction of the measured 0.4484 is unreachable by any amount of training** — and with (4)
showing training itself is exhausted, this ceiling is now the leading explanation rather than one of
several.

Note the two measurements are not independent methods: both FR-3's 0.91 and this 8.8% come from
comparison against LVIS gold. They are different *samples* — FR-3 over sampled shards, this over the
ingested corpus — which is corroboration, not confirmation by a second technique.

A hand review of run 14's most-confident disagreements confirmed this ceiling directly — and found
that `figure` is where it concentrates, exactly as the 0.62 predicts. See (6), including why the
resulting accuracy gain must **not** be read as the model improving.

#### 4. It has converged at ~0.45 — the ceiling is upstream of training

At shakedown scale this looked like under-training: run 3 ended with train loss 1.7661 against val
loss 1.7665 — no gap at all — and val accuracy still climbing steeply (0.306 → 0.378 → 0.461). The
obvious prescription was more epochs and more data.

**Run 14 tested that and refuted it.** With 5.8× the data and a fourth epoch:

| | run 3 (2,000 × 3) | run 14 (11,783 × 4) |
|---|---:|---:|
| val accuracy | 0.4611 | 0.4552 |
| macro precision | 0.484 | 0.482 |
| macro recall | 0.334 | 0.345 |
| macro F1 | 0.367 | 0.385 |

Accuracy did not move (slightly down); macro F1 improved modestly. If data volume were the binding
constraint, this is exactly where it would have shown, and it did not.

Run 14's own curve says the same thing from the other side. Val loss went 1.766 → 1.696 → 1.729 →
1.703 — flat after epoch 1 — while train loss fell 2.03 → 2.01 → 1.51 → 1.46. The model kept
extracting more from the training set and none of it transferred. That is the far side of the
underfitting boundary: **more epochs would now widen the gap rather than close it.**

So the two cheap levers are spent. Whatever caps this model at ~0.45 is not epochs and not corpus
size, which leaves the two candidates below: the label ceiling in (3), and the shape-only
representation in (5). Distinguishing them needs independently-annotated data — see the follow-up.

#### 5. Bias introduced by the pipeline itself, not the data

- **Renders are shape-only.** Every mesh is drawn with one neutral grey material
  (`workers/render.py`), discarding its own textures and colours. Deliberate — it stops the model
  keying on an artist's palette — but it means classes separable mainly by *appearance* rather than
  *silhouette* cannot be learned at all, and a 12-view orbit at fixed elevation adds a viewpoint
  prior on top.
- **The roster is a choice with consequences.** `chair`/`table`/`lamp` are three of the four
  smallest classes and could have been one `furniture` class — which would have removed two tail
  classes at the cost of the resolution that makes the classifier useful. Splitting them was chosen
  for visual coherence (see [Class-list approach](#metadata-exploration-milestone-1)); the tail
  problem in (1) is partly the price.
- **The corpus is Sketchfab's, not the world's.** Whatever artists choose to model and upload sets
  the distribution; nothing here corrects for it, and the skew in (1) is that choice showing through.

#### 6. What a hand review of 74 disagreements actually found (milestone 8)

The first hand-labeling pass ([the review queue](#the-review-queue-milestone-8)) judged the **74
disagreements above 0.7 confidence** from run 14's `test` split, from 12-view contact sheets
(`capture_renders.py --gcs`). Verdicts: **24 label_wrong · 27 model_wrong · 23 unclear**.

**Confidence separates the three, which makes it a usable triage signal.**

| confidence | label_wrong | model_wrong | unclear | n |
|------------|------------:|------------:|--------:|---:|
| >0.95      | **10**      | 1           | 0       | 11 |
| 0.85–0.95  | 5           | 6           | 2       | 13 |
| 0.70–0.85  | 9           | **20**      | **21**  | 50 |

Where the classifier commits hardest against a label it is almost always right and the *label* is
wrong. Lower down, disagreement mostly means the model is wrong or the object is unlabelable. So a
future pass should work from the top and stop when the mix turns, rather than budget a fixed count.

**`figure` absorbs almost every model error: 24 of the 27 wrong predictions were `figure`**, the
largest class at 1,594 training examples — against true labels of `animal` ×14 (bison, mammoth
skeleton, two dinosaurs, a fish, birds, quadrupeds), `aircraft` ×4, `weapon` ×4, and one each of
`chair` (a bar stool, at 0.945), `electronics`, `food`, `building`. This is (1)'s precise-but-
under-predicted tail seen from the other side: the failure is not that a bison is unrecognisable, it
is that the majority class collects everything the model is unsure of. That is a **calibration**
mechanism rather than a capability limit, and it is the concrete case for rerunning the (2)
class-weighting A/B at full scale.

**23 of 74 have no correct answer in the roster** — an architectural corbel with a carved face
(twice), a fluted column, gridded wall panels, a lattice truss, a bone specimen, concentric rings, a
cube with an arm attached, several unidentifiable slivers. Nobody can label these correctly, so they
cap achievable accuracy independently of both the label ceiling in (3) and the representation limit
in (5). At ~31% of this band it is not a rounding error, and it argues for either an explicit
`other` class or an out-of-roster exclusion — see (5) on the roster being a choice.

**⚠️ The accuracy gain from correcting labels is a biased estimator — read it with care.** Applying
the 24 corrections moved `test` accuracy **0.4484 → 0.4689** and macro recall **0.336 → 0.3472**
(`evaluation 3`, the same 1,173 replayed uids). But 24/1173 = **+0.0205**, exactly the observed gain:
*every* correction flipped a prediction from wrong to right. That is guaranteed by the method — a
`label_wrong` verdict is the reviewer siding with the classifier — so **this procedure can only ever
move accuracy up**, whether or not the new labels are truer. The reviewer also saw the same
shape-only renders and knew the prediction. What the pass legitimately establishes is that the old
number *overstated* error by at least two points; it establishes nothing about the model. The
counterweight against rubber-stamping is that the reviewer sided with the **label** more often (27
model_wrong), and those move accuracy by exactly zero. An unbiased estimate needs an annotator who
never sees the prediction — which is what the [second dev set](#follow-up-a-real-second-dev-set)
would provide.

**The tail is untouched by any of this.** After correction `aircraft` recall is still **0.000** on 26
models — the model has never once got an aircraft right — with `table` at 0.120 and `electronics` at
0.254. Twenty-four labels do not dent that, and nothing in this pass suggests they would.

#### Follow-up (post-MVP): render textures, and A/B them

The other candidate for the ~0.45 ceiling, and the one that would be easiest to
under-scope. **Shading is not the gap** — the render stage already lights each mesh with
camera-attached key and fill lights precisely so shading reveals form, and the tone was tuned twice
to get there. What is discarded is **texture and colour**: `workers/render.py` overrides every mesh's
own material with one neutral grey, so a wooden chair and a steel chair are identical inputs, and
`food`, `plant` and `electronics` lose their most distinguishing cue. Those are three of the classes
performing worst.

**It is a pipeline change, not a render tweak.** The convert stage exports **PLY, which carries no UV
textures at all**, so materials are gone two stages before rendering. Using them means preserving
textured geometry through convert and normalize, then **re-rendering all 11,783 models** — a few
dollars and a couple of hours of parallel Cloud Run, plus a schema question about what the converted
artifact is.

Test it as a controlled A/B rather than a migration: re-render one subset with textures, train on it,
and compare against the same subset shape-only. Anything less cannot separate "textures help" from
"this subset is easier". Weigh it against the risk it introduces — with textures the model can key on
an artist's palette or a render style rather than the object, which is a *new* bias in exchange for
the one it removes.

#### Follow-up: a real second dev set

Everything above rests on one corpus labeled by one weak labeler, so it measures generalization to
unseen **objects**, not whether the labels are right. After (4), that is no longer a nicety: with
training exhausted, the open question is *what* caps the model at ~0.45, and only clean labels
separate "the labels are wrong" from "the representation cannot express it". The intended fix is measured and scoped rather
than hypothetical: LVIS gold has 13,118 objects, of which only **475** are in our trainable set and
only **49** in the held-out split — roughly 4 per class, too thin to report. A genuine second dev set
means **ingesting ~1,000 of the ~12,600 LVIS-annotated objects not yet in the corpus** through the
existing (idempotent) pipeline: a data run, not a code change. That closes FR-7 properly and answers
the (3)-versus-(4) question in the same stroke.

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
