"""Score a finished run against a held-out dev set (FR-7, M7 / C1).

Separate from training on purpose. The trainer reports on ``val``, which it has
consulted every epoch and therefore cannot score honestly; ``test`` is held back
precisely so one number exists that nothing steered against. Keeping this a
distinct command is what stops "evaluate the model" quietly becoming another
training-time metric.

    make evaluate RUN=4                # score run 4 on the held-out test split
    make evaluate RUN=4 SPLIT=val      # re-score val, e.g. to compare methods

The split itself is **recomputed**, not stored — a run records only its sizes.
`stratified_split` is deterministic given the same samples and seed, so the same
trainable set reproduces the same partition exactly. What breaks that is the
trainable set changing: a label added or corrected since the run moves models
between partitions, and the recomputed "test" is then not the one held out. The
report records the `label_hash` it was computed under so that stays detectable
after the fact; comparing it is deliberately not enforced yet.
"""

from __future__ import annotations

import argparse

from infer import evaluate_samples as score
from infer import load_run_model
from splits import stratified_split
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


def evaluate_run(
    run_id: int, dev_set: str = "test", num_workers: int = 0
) -> dict:
    """Load a run, score it on `dev_set`, store the report, and return it."""
    storage = build_storage(get_settings())
    model, config, snapshot = load_run_model(run_id, storage)

    samples = load_trainable_samples()
    # The run's own seed, not Config's default: the partition is only reproducible
    # under the seed that produced it, and a run may have set its own.
    split = stratified_split(samples, config.seed)
    scored = getattr(split, dev_set)
    if not scored:
        raise SystemExit(f"the {dev_set} split is empty — nothing to score")

    recorded_hash = snapshot.get("label_hash")
    current_hash = data_snapshot(samples, split)["label_hash"]
    if recorded_hash and recorded_hash != current_hash:
        # Not fatal by decision: the number is still worth having, and the
        # mismatch is recorded on the row so it can be revisited.
        print(
            f"WARNING: the labeled set has changed since run {run_id} "
            f"({recorded_hash[:19]}… -> {current_hash[:19]}…), so this "
            f"{dev_set} split is not the one the run held out."
        )

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
