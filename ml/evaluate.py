"""Score a finished run against a held-out dev set (FR-7, M7 / C1).

Separate from training on purpose. The trainer reports on ``val``, which it has
consulted every epoch and therefore cannot score honestly; ``test`` is held back
precisely so one number exists that nothing steered against. Keeping this a
distinct command is what stops "evaluate the model" quietly becoming another
training-time metric.

    make evaluate RUN=4                # score run 4 on the held-out test split
    make evaluate RUN=4 SPLIT=val      # re-score val, e.g. to compare methods

The partition is **replayed from the run**, not recomputed: `data_snapshot` records
the uids it held out. Recomputing is deterministic given the same samples and
seed, but the partition is a function of the whole sample set, so one label added
or corrected reshuffles every class and moves models between train, val and test
— and the resulting number looks no different for being wrong. Replaying is what
makes an evaluation reproducible rather than merely deterministic.

Runs predating that field (2 through 4), and `train`, which is never recorded,
fall back to recomputation with a warning when the labels have moved since.
Either way the report records the `label_hash` it was scored under.
"""

from __future__ import annotations

import argparse

from infer import evaluate_samples as score
from infer import load_run_model
from splits import DatasetSplit, stratified_split
from train import data_snapshot, load_trainable_samples

from app.config import get_settings
from app.db import session_scope
from app.models import Evaluation
from app.storage import build_storage

# The partitions `stratified_split` produces, by name.
SPLITS = ("test", "val", "train")


def record_evaluation(
    run_id: int, dev_set: str, report: dict, label_hash: str | None
) -> int:
    """Store one dev-set report and return its id."""
    with session_scope() as session:
        evaluation = Evaluation(
            run_id=run_id, dev_set=dev_set, report=report, label_hash=label_hash
        )
        session.add(evaluation)
        session.flush()  # assigns the id before the scope commits
        return evaluation.id


def resolve_scored_samples(
    run_id: int,
    dev_set: str,
    snapshot: dict,
    samples: list[tuple[str, str]],
    split: DatasetSplit,
) -> list[tuple[str, str]]:
    """The models to score: the run's *recorded* partition where one exists.

    A run records the uids it held out (`data_snapshot["held_out"]`), because a
    recomputed partition is only the run's own while the labeled set is
    unchanged — one label added or corrected reshuffles every class and moves
    models across train/val/test. Replaying the recorded uids is what makes an
    evaluation reproducible rather than merely deterministic.

    Labels come from the **current** database, not from training time. The point
    of scoring is "is the model right?", and a corrected label is a better answer
    to that than the one the run trained against — which is also what makes the
    M8 loop work: hand-correct, re-score, see the difference.

    Falls back to the recomputed split for runs predating the field (2 through 4)
    and for `train`, which is never recorded, warning when the labels have moved.
    """
    recorded = (snapshot.get("held_out") or {}).get(dev_set)
    uid_to_class = dict(samples)

    if recorded is None:
        recorded_hash = snapshot.get("label_hash")
        current_hash = data_snapshot(samples, split)["label_hash"]
        if recorded_hash and recorded_hash != current_hash:
            print(
                f"WARNING: run {run_id} recorded no {dev_set} split and the labeled "
                f"set has changed since ({recorded_hash[:19]}… -> "
                f"{current_hash[:19]}…), so this partition is not the one it held out."
            )
        return getattr(split, dev_set)

    # A recorded uid can leave the trainable set: soft-deleted, label removed, or
    # renders gone. Skipping is right — the alternative is failing an evaluation
    # over a model that no longer exists — but the count is reported, because a
    # shrinking dev set changes what the numbers mean.
    scored = [(uid, uid_to_class[uid]) for uid in recorded if uid in uid_to_class]
    missing = len(recorded) - len(scored)
    if missing:
        print(
            f"note: {missing} of {len(recorded)} recorded {dev_set} models are no "
            "longer trainable (deleted, unlabeled, or unrendered) and were skipped"
        )
    return scored


def evaluate_run(
    run_id: int, dev_set: str = "test", num_workers: int = 0
) -> dict:
    """Load a run, score it on `dev_set`, store the report, and return it."""
    storage = build_storage(get_settings())
    model, config, snapshot = load_run_model(run_id, storage)

    samples = load_trainable_samples()
    # The run's own seed, not Config's default: a recomputed partition is only
    # reproducible under the seed that produced it, and a run may have set its own.
    split = stratified_split(samples, config.seed)
    current_hash = data_snapshot(samples, split)["label_hash"]
    scored = resolve_scored_samples(run_id, dev_set, snapshot, samples, split)
    if not scored:
        raise SystemExit(f"the {dev_set} split is empty — nothing to score")

    print(f"scoring run {run_id} on {len(scored)} {dev_set} models ({config.backbone})")
    report = score(model, scored, storage, dev_set, num_workers=num_workers)
    evaluation_id = record_evaluation(run_id, dev_set, report, current_hash)

    accuracy = report["accuracy"]
    macro_recall = report["macro_recall"]
    print(
        f"evaluation {evaluation_id}: {dev_set} accuracy {accuracy:.4f}, "
        f"macro recall {macro_recall if macro_recall is None else round(macro_recall, 4)}"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a training run (M7).")
    parser.add_argument("--run", type=int, required=True, help="training run id")
    parser.add_argument(
        "--split",
        default="test",
        choices=SPLITS,
        help="which partition to score (default: the held-out test split)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_run(args.run, args.split, args.num_workers)


if __name__ == "__main__":
    main()
