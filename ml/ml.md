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

### Metrics

- **Per-class precision and recall** on both dev sets.
- **Confusion matrix** (resolves the confusion-matrix TODO): an N×N table for the N classes where
  entry (i, j) = the number of examples whose **true** class is i that the model **predicted** as
  j. The diagonal is correct predictions; off-diagonal entries show which classes get confused
  for which. Report one per dev set. It's the primary tool for the bias analysis below.

### Bias Analysis

- Per-class precision/recall + confusion matrices on both dev sets.
- **Key question:** which categories do metadata-derived weak labels systematically corrupt?
  Compare weak-label-trained vs. hand-label-corrected performance **per class** — a class that
  improves a lot after manual correction is one the weak labels were poisoning.

## Coding Standards (ML)

- **Language/framework:** Python 3.11+, PyTorch. Type hints on public functions.
- **Reproducibility (NFR-4):** every run records config, data-split version, random seed, and
  metrics; persist them to the `training_run` entity ([server.md](../server/server.md#database)) so the
  [dashboard](../web/web.md#training-dashboard) can show them.
- **Config over code:** hyperparameters in config files, not hardcoded in scripts.
- **Data loading:** stream renders/point clouds from object storage; never assume the full
  dataset fits in memory or on the local disk.
- **Cost:** train on spot/preemptible GPU; checkpoint often so a preemption doesn't lose the run.
- **Evaluation code is shared:** the same metric functions produce the numbers for both dev sets
  and the dashboard — no re-implementations that can drift.
- **Formatting/lint:** Ruff; no unformatted code committed.
