"""Measure the figure/animal boundary — can keywords resolve it, or only a human?

`figure` vs `animal` is the roster's one genuinely ambiguous boundary: the weakest
weak-label class (0.62 precision) and the trained model's largest confusion pair.
The agreed rule is **stance decides, not the head** — a biped with arms is `figure`
whatever its head (ml/taxonomy.py). The obvious way to act on that is a
"stance outranks species" precedence rule inside `CLASS_TO_KEYWORDS`, and this
script exists to answer whether that would work *before* anyone writes it.

It reports the two numbers ml.md#the-figureanimal-boundary rests on:

* **Reach** — over every object whose category gate yields exactly
  {figure, animal}, which keyword lists fire. A precedence rule can only change
  objects that match **both** lists; everything else is already decided or already
  ambiguous. This needs no gold labels, and it caps the benefit however good the
  stance vocabulary turns out to be.
* **Validity** — over the subset carrying an LVIS gold label, which tokens actually
  discriminate gold `figure` from gold `animal`, plus the labeler's confusion and
  abstention rate on them. A token that looks like stance to a human ("character")
  may simply mean "game asset".

Kept separate from `eval_weak_labels.py`, which grades *all* classes against gold:
this asks a narrower question — is one specific rule worth writing — and answering
it needs the ambiguous population itself, not per-class precision.

Metadata only; no meshes downloaded. Output JSON is gitignored (NFR-6).

    make evalboundary            # 8 shards
    make evalboundary SHARDS=24  # the sample ml.md's gold numbers cite
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import objaverse
from eval_weak_labels import build_uid_to_gold_class
from io_utils import write_json
from taxonomy import CLASS_TO_KEYWORDS
from weak_label import category_candidates, label_object, sample_uids_by_shard, tokens

# The ambiguous pair. A frozenset so it can be compared against the gate's output
# directly, and so it cannot be mutated by a caller.
PAIR = frozenset({"figure", "animal"})

# A token needs at least this many gold objects before its figure/animal split
# means anything — below it, one uploader's tagging habit looks like a signal.
MIN_GOLD_SUPPORT = 8

# Tokens that might describe how a thing is *built or posed* rather than what
# species it is: the vocabulary a stance rule would need. Listed here to be
# measured, deliberately NOT merged into `CLASS_TO_KEYWORDS` — the whole point is
# that they have to earn a place there, and the measurement says they don't.
STANCE_CANDIDATES = (
    "character", "anthro", "anthropomorphic", "humanoid", "biped", "bipedal",
    "mascot", "mascotte", "fursona", "furry", "avatar", "vrchat", "vtuber", "oc",
    "npc", "rigged", "rig", "animated", "gamecharacter", "cartoon", "toon",
)


def ambiguous_objects(uid_to_annotation: dict[str, dict]) -> dict[str, dict]:
    """The objects whose category gate yields exactly {figure, animal}.

    These are the ones the keyword resolver has to split, and the only ones any
    figure/animal rule could affect.
    """
    return {
        uid: annotation
        for uid, annotation in uid_to_annotation.items()
        if category_candidates(annotation) == PAIR
    }


def keyword_bucket(annotation: dict) -> str:
    """Which of the figure/animal keyword lists fire for this object.

    "both" is the bucket that matters: a precedence rule reorders a contested
    decision, so an object matching one list or neither is untouched by it.
    """
    tokens_set = tokens(annotation)
    hits_figure = bool(tokens_set & set(CLASS_TO_KEYWORDS["figure"]))
    hits_animal = bool(tokens_set & set(CLASS_TO_KEYWORDS["animal"]))
    if hits_figure and hits_animal:
        return "both"
    if hits_figure:
        return "figure only"
    if hits_animal:
        return "animal only"
    return "neither"


def measure_reach(uid_to_annotation: dict[str, dict]) -> dict[str, object]:
    """How many ambiguous objects a precedence rule could reach, and today's outcome."""
    bucket_to_count: Counter[str] = Counter()
    bucket_outcome_to_count: Counter[tuple[str, str]] = Counter()
    for annotation in uid_to_annotation.values():
        bucket = keyword_bucket(annotation)
        bucket_to_count[bucket] += 1
        class_name, reason = label_object(annotation)
        bucket_outcome_to_count[(bucket, class_name or reason)] += 1

    total = len(uid_to_annotation) or 1
    return {
        "ambiguous_count": len(uid_to_annotation),
        "by_bucket": dict(bucket_to_count),
        "both_lists_fire_share": round(bucket_to_count["both"] / total, 4),
        "neither_fires_share": round(bucket_to_count["neither"] / total, 4),
        "by_bucket_and_outcome": {
            f"{bucket} -> {outcome}": count
            for (bucket, outcome), count in bucket_outcome_to_count.most_common()
        },
    }


def measure_validity(
    uid_to_annotation: dict[str, dict], uid_to_gold_class: dict[str, str]
) -> dict[str, object]:
    """Token skew, labeler confusion and abstention over the gold-labeled subset."""
    gold_subset = {
        uid: annotation
        for uid, annotation in uid_to_annotation.items()
        if uid_to_gold_class.get(uid) in PAIR
    }
    if not gold_subset:
        return {"gold_count": 0}

    gold_weak_to_count: Counter[tuple[str, str]] = Counter()
    abstained = 0
    for uid, annotation in gold_subset.items():
        class_name, reason = label_object(annotation)
        gold_weak_to_count[(uid_to_gold_class[uid], class_name or reason)] += 1
        abstained += class_name is None

    token_to_gold_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for uid, annotation in gold_subset.items():
        for token in tokens(annotation):
            token_to_gold_counts[token][uid_to_gold_class[uid]] += 1

    # Rank by distance from an even split: a token at 0.5 says nothing, one near
    # 0 or 1 separates the classes. Reported for both directions, because a
    # strongly animal-leaning token is just as usable as a figure-leaning one.
    skews = []
    for token, gold_counts in token_to_gold_counts.items():
        support = gold_counts["figure"] + gold_counts["animal"]
        if support < MIN_GOLD_SUPPORT:
            continue
        figure_share = gold_counts["figure"] / support
        skews.append(
            {
                "token": token,
                "figure": gold_counts["figure"],
                "animal": gold_counts["animal"],
                "figure_share": round(figure_share, 3),
                "skew": round(abs(figure_share - 0.5), 3),
            }
        )
    skews.sort(key=lambda row: row["skew"], reverse=True)

    stance_to_count = {
        candidate: sum(
            candidate in tokens(annotation) for annotation in uid_to_annotation.values()
        )
        for candidate in STANCE_CANDIDATES
    }
    return {
        "gold_count": len(gold_subset),
        "gold_by_class": dict(Counter(uid_to_gold_class[uid] for uid in gold_subset)),
        "abstained": abstained,
        "abstention_share": round(abstained / len(gold_subset), 4),
        "gold_to_weak": {
            f"{gold} -> {weak}": count
            for (gold, weak), count in gold_weak_to_count.most_common()
        },
        "min_gold_support": MIN_GOLD_SUPPORT,
        "token_skew": skews,
        "stance_candidate_counts": stance_to_count,
    }


def run(out_dir: Path, shard_count: int) -> dict[str, object]:
    sample_uids, shard_ids = sample_uids_by_shard(shard_count)
    print("=== figure/animal boundary — can keywords resolve it? ===")
    print(f"shards {', '.join(shard_ids)} ({len(sample_uids):,} objects); "
          f"downloading metadata ...")
    uid_to_annotation = objaverse.load_annotations(sample_uids)
    gated = ambiguous_objects(uid_to_annotation)

    reach = measure_reach(gated)
    share = 100 * len(gated) / (len(uid_to_annotation) or 1)
    print(f"\n{len(gated):,} of {len(uid_to_annotation):,} objects gate to "
          f"{{figure, animal}} ({share:.1f}%)")
    print("\nreach — which keyword lists fire (only 'both' is reachable by a "
          "precedence rule):")
    for bucket, count in Counter(reach["by_bucket"]).most_common():
        print(f"  {count:6,}  ({100 * count / (len(gated) or 1):4.1f}%)  {bucket}")

    validity = measure_validity(gated, build_uid_to_gold_class())
    if validity["gold_count"]:
        print(f"\nvalidity — {validity['gold_count']} of them carry an LVIS gold label "
              f"{validity['gold_by_class']}")
        print(f"  the labeler abstains on {validity['abstained']} "
              f"({100 * validity['abstention_share']:.0f}%)")
        print(f"\n  most discriminative tokens (>= {MIN_GOLD_SUPPORT} gold objects):")
        print(f"    {'token':22s} {'figure':>7s} {'animal':>7s} {'fig share':>10s}")
        for row in validity["token_skew"][:15]:
            print(f"    {row['token']:22s} {row['figure']:7d} {row['animal']:7d} "
                  f"{row['figure_share']:10.2f}")
        if not validity["token_skew"]:
            print(f"    none reaches {MIN_GOLD_SUPPORT} gold objects at this sample size")
    else:
        print("\nno LVIS gold overlap at this sample size — raise --shards")

    print("\n  candidate stance tokens present in the gated population:")
    for candidate, count in sorted(
        validity.get("stance_candidate_counts", {}).items(),
        key=lambda pair: pair[1],
        reverse=True,
    )[:10]:
        print(f"    {candidate:18s} {count:5,}")

    result: dict[str, object] = {
        "shard_ids": shard_ids,
        "sample_size": len(sample_uids),
        "n_annotations": len(uid_to_annotation),
        "reach": reach,
        "validity": validity,
    }
    out_path = out_dir / "figure_animal_boundary.json"
    write_json(out_path, result)
    print(f"\nwrote {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the figure/animal boundary.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/exploration"),
                        help="output directory for the summary (gitignored)")
    parser.add_argument("--shards", type=int, default=8,
                        help="number of whole metadata shards to sample (~5k each)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(args.out_dir, args.shards)


if __name__ == "__main__":
    main()
