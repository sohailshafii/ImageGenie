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
