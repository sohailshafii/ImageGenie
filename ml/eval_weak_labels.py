"""FR-3 pass 2, stage 3 — evaluate Sketchfab weak labels against the LVIS gold set.

The LVIS-labeled objects are clean, human-curated labels (pass 1). By comparing
the *noisy* Sketchfab weak labels against them on the objects that have both, we
can measure per-class precision/recall of the weak-labeling rules and tune the
keyword lists from data instead of by eyeball. See `ml/ml.md#sketchfab-weak-labeling`.

Built up in small steps:

* **gold-label lookup** — `uid -> roster class`, inverting
  `taxonomy.CLASS_TO_LVIS_CATEGORIES` over the curated LVIS annotations (per-class
  counts match `build_class_list.py`).
* **coverage** — sample whole shards, pair each gold object's clean label with the
  weak labeler's guess, and report how much of the gold set the weak rules label at
  all (a recall proxy). Per-class precision/recall + confusion matrix come next.

Metadata only; no meshes downloaded. Output JSON is gitignored (NFR-6).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import objaverse
from io_utils import write_json
from taxonomy import CLASS_TO_LVIS_CATEGORIES
from weak_label import label_object, sample_uids_by_shard


def build_uid_to_gold_class() -> dict[str, str]:
    """Map each LVIS gold object UID to its roster class (inverse of the merges).

    Our classes are disjoint at the object level (`build_class_list` reports 0
    multi-class objects), so each gold UID resolves to exactly one class. The
    `setdefault` is a defensive first-wins tie-break in case that ever changes.
    """
    category_to_uids: dict[str, list[str]] = objaverse.load_lvis_annotations()
    uid_to_gold_class: dict[str, str] = {}
    for class_name, categories in CLASS_TO_LVIS_CATEGORIES.items():
        for category in categories:
            for uid in category_to_uids.get(category, []):
                uid_to_gold_class.setdefault(uid, class_name)
    return uid_to_gold_class


def collect_gold_weak_pairs(
    uid_to_gold_class: dict[str, str], shard_count: int
) -> tuple[list[tuple[str, str | None, str]], list[str]]:
    """Pair (gold_class, weak_label, reason) for gold objects in the sampled shards.

    Only objects that carry an LVIS gold label are kept — they're the ones we can
    score the weak labeler against. `weak_label` is None when the labeler left the
    object unlabeled (reason "ambiguous" / "out-of-scope").
    """
    sample_uids, shard_ids = sample_uids_by_shard(shard_count)
    uid_to_annotation = objaverse.load_annotations(sample_uids)
    gold_weak_pairs: list[tuple[str, str | None, str]] = []
    for uid, annotation in uid_to_annotation.items():
        gold_class = uid_to_gold_class.get(uid)
        if gold_class is None:
            continue
        weak_label, reason = label_object(annotation)
        gold_weak_pairs.append((gold_class, weak_label, reason))
    return gold_weak_pairs, shard_ids


def per_class_metrics(
    gold_weak_pairs: list[tuple[str, str | None, str]],
) -> dict[str, dict[str, object]]:
    """Per-class precision/recall of the weak labels against gold.

    recall    = correctly labeled class / gold objects of that class (unlabeled
                objects count against it — the recall ceiling).
    precision = correctly labeled class / everything labeled that class.
    Either is None when its denominator is 0 (class absent from this sample).
    """
    gold_total = Counter()
    predicted_total = Counter()
    true_positive = Counter()
    for gold_class, weak_label, _ in gold_weak_pairs:
        gold_total[gold_class] += 1
        if weak_label is None:
            continue
        predicted_total[weak_label] += 1
        if weak_label == gold_class:
            true_positive[gold_class] += 1

    class_to_metrics: dict[str, dict[str, object]] = {}
    for class_name in sorted(CLASS_TO_LVIS_CATEGORIES):
        gold_count = gold_total[class_name]
        predicted_count = predicted_total[class_name]
        correct = true_positive[class_name]
        class_to_metrics[class_name] = {
            "gold": gold_count,
            "predicted": predicted_count,
            "true_positive": correct,
            "precision": correct / predicted_count if predicted_count else None,
            "recall": correct / gold_count if gold_count else None,
        }
    return class_to_metrics


def _format_ratio(value: float | None) -> str:
    """Format a precision/recall value in [0, 1], or a dash when undefined."""
    return f"{value:.2f}" if value is not None else "   -"


def confusion_matrix(
    gold_weak_pairs: list[tuple[str, str | None, str]],
) -> dict[str, Counter[str]]:
    """Confusion counts per gold class (rows=gold, cols=weak label).

    Only committed labels (weak_label not None) appear — unlabeled gold objects
    are recall misses, not confusions. The diagonal is correct predictions; a
    row's off-diagonal entries show which classes that gold class gets called.
    """
    gold_to_weak_counts: dict[str, Counter[str]] = {
        class_name: Counter() for class_name in CLASS_TO_LVIS_CATEGORIES
    }
    for gold_class, weak_label, _ in gold_weak_pairs:
        if weak_label is not None:
            gold_to_weak_counts[gold_class][weak_label] += 1
    return gold_to_weak_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/exploration"),
                        help="output directory for the eval summary (gitignored)")
    parser.add_argument("--shards", type=int, default=1,
                        help="number of whole metadata shards to sample (~5k each)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    uid_to_gold_class = build_uid_to_gold_class()
    gold_weak_pairs, shard_ids = collect_gold_weak_pairs(uid_to_gold_class, args.shards)

    n_gold_in_sample = len(gold_weak_pairs)
    reason_to_count = Counter(reason for _, _, reason in gold_weak_pairs)
    n_labeled = sum(1 for _, weak_label, _ in gold_weak_pairs if weak_label is not None)

    print(f"=== Weak-vs-gold coverage (shards {', '.join(shard_ids)}) ===")
    print(f"gold objects total: {len(uid_to_gold_class):,}")
    print(f"gold objects in sample: {n_gold_in_sample:,}")
    if n_gold_in_sample:
        print(f"weak labeler assigned a label to {n_labeled}/{n_gold_in_sample} "
              f"({n_labeled / n_gold_in_sample * 100:.0f}%):")
        for reason in ("category", "keyword", "rescue", "ambiguous", "out-of-scope"):
            print(f"  {reason_to_count[reason]:5,}  {reason}")

    class_to_metrics = per_class_metrics(gold_weak_pairs)
    print("\n=== Per-class precision/recall (weak vs gold) ===")
    print(f"{'class':<13}{'gold':>5}{'pred':>6}{'TP':>4}{'prec':>7}{'recall':>8}")
    for class_name in sorted(CLASS_TO_LVIS_CATEGORIES):
        class_metrics = class_to_metrics[class_name]
        print(f"{class_name:<13}{class_metrics['gold']:>5}{class_metrics['predicted']:>6}"
              f"{class_metrics['true_positive']:>4}"
              f"{_format_ratio(class_metrics['precision']):>7}"
              f"{_format_ratio(class_metrics['recall']):>8}")
    print("(small per-class gold counts at 1 shard are noisy — raise SHARDS for stable numbers.)")

    gold_to_weak_counts = confusion_matrix(gold_weak_pairs)
    classes = sorted(CLASS_TO_LVIS_CATEGORIES)
    header_label = "gold \\ weak"
    header = "".join(f"{class_name[:4]:>5}" for class_name in classes)
    print("\n=== Confusion matrix (rows=gold, cols=weak; '.'=0, diagonal=correct) ===")
    print(f"{header_label:<13}{header}")
    for gold_class in classes:
        cells = "".join(
            f"{(gold_to_weak_counts[gold_class][weak_class] or '.'):>5}"
            for weak_class in classes
        )
        print(f"{gold_class:<13}{cells}")

    write_json(args.out_dir / "weak_label_eval.json", {
        "shard_ids": shard_ids,
        "n_gold_total": len(uid_to_gold_class),
        "n_gold_in_sample": n_gold_in_sample,
        "n_labeled": n_labeled,
        "by_reason": dict(reason_to_count),
        "per_class_metrics": class_to_metrics,
        "confusion_matrix": {
            gold_class: dict(weak_counts)
            for gold_class, weak_counts in gold_to_weak_counts.items()
        },
    })
    print(f"\nwrote {args.out_dir / 'weak_label_eval.json'}")


if __name__ == "__main__":
    main()
