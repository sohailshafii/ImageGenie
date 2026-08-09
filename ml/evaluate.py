"""Score a finished run against a held-out dev set (FR-7, M7 / C1).

Separate from training on purpose. The trainer reports on ``val``, which it has
consulted every epoch and therefore cannot score honestly; ``test`` is held back
precisely so one number exists that nothing steered against. Keeping this a
distinct command is what stops "evaluate the model" quietly becoming another
training-time metric.

    make evaluate RUN=4                 # score run 4 on the held-out test split
    make evaluate RUN=4 DEVSET=val      # re-score val, e.g. to compare methods
    make evaluate RUN=4 DEVSET=lvis     # the second dev set (FR-7)

Two kinds of dev set, and the difference is the point. A **partition** (test, val,
train) is a slice of our own corpus carrying our own weak labels, so it measures
generalization to unseen *objects* and cannot tell a correct model from one
faithfully reproducing the weak labeler's mistakes. **`lvis`** is a separate set
of objects annotated outside this project entirely (`ml/build_dev_set.py`), so it
measures whether the labels are right at all — the question milestone 8's own
corrections structurally cannot answer.

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

from build_dev_set import load_dev_set
from infer import evaluate_samples as score
from infer import load_run_model
from model import MultiViewCNN
from splits import DatasetSplit, stratified_split
from sqlalchemy import select
from train import data_snapshot, label_hash, load_trainable_samples, subsample

from app.config import get_settings
from app.db import session_scope
from app.models import (
    Artifact,
    ArtifactStage,
    ArtifactStatus,
    Evaluation,
    Model,
    TrainingStatus,
)
from app.storage import Storage, build_storage

# The partitions `stratified_split` produces, by name.
PARTITIONS = ("test", "val", "train")
# The second dev set (FR-7): LVIS gold objects selected by `ml/build_dev_set.py`,
# labeled outside this project entirely. Not a partition of anything — it is a
# different corpus — which is why it reads its samples from a file rather than
# from a split.
LVIS = "lvis"
DEV_SETS = (*PARTITIONS, LVIS)


def record_evaluation(
    run_id: int, dev_set: str, report: dict, label_hash: str | None
) -> int:
    """Store one finished dev-set report and return its id.

    `status` is passed explicitly rather than left to the column default, which is
    `running`: this writes a row that is already done. A scoring job that wants to
    announce itself before it starts is a different call (see the button path).
    """
    with session_scope() as session:
        evaluation = Evaluation(
            run_id=run_id,
            dev_set=dev_set,
            status=TrainingStatus.completed,
            report=report,
            label_hash=label_hash,
        )
        session.add(evaluation)
        session.flush()  # assigns the id before the scope commits
        return evaluation.id


def load_rendered_uids(uids: list[str]) -> set[str]:
    """Which of `uids` are live and finished rendering — the ones we can score.

    The dev set is selected before it is ingested, so at scoring time some of it
    may not have arrived: a download that dead-lettered, a mesh the converter
    rejected, a soft-delete. Scoring is only possible where the 12 views exist,
    so this is the gate, and the shortfall is reported rather than assumed away.
    """
    if not uids:
        return set()
    with session_scope() as session:
        stmt = (
            select(Model.uid)
            .join(Artifact, Artifact.model_uid == Model.uid)
            .where(Model.uid.in_(uids))
            .where(Model.deleted_at.is_(None))
            .where(Artifact.stage == ArtifactStage.rendered)
            .where(Artifact.status == ArtifactStatus.done)
        )
        return set(session.scalars(stmt).all())


def resolve_lvis_dev_set() -> list[tuple[str, str]]:
    """The second dev set: gold-labeled objects, filtered to what is scorable and clean.

    Two filters, and the second is the point of the whole exercise. Models that
    have not finished rendering cannot be scored at all. Models that are
    *trainable* — labeled and rendered — must not be scored either, because a
    label in the `label` table is what puts a model into a training run, and a
    dev set that a run trained on measures memorisation.

    That second case should be impossible: the gold labels live in a CSV and
    never enter the `label` table, so these models stay unlabeled and therefore
    untrainable by construction. The one route in is `make backfill-labels`,
    which loads `weak_labels.csv` and overlaps this selection on ~75 uids.
    Dropping them keeps the number honest where failing would leave no number at
    all, and the count is loud because it means the contamination guard has
    actually fired.
    """
    dev_set = load_dev_set()
    rendered_uids_set = load_rendered_uids([uid for uid, _ in dev_set])
    scorable = [(uid, class_name) for uid, class_name in dev_set
                if uid in rendered_uids_set]
    if not scorable:
        raise SystemExit(
            f"none of the {len(dev_set)} selected dev-set models are rendered yet — "
            "seed them through the pipeline first "
            "(`python -m app.seed --from-labels data/devset/lvis_dev.csv`)"
        )
    if len(scorable) < len(dev_set):
        print(
            f"note: {len(dev_set) - len(scorable)} of {len(dev_set)} dev-set models "
            "are not rendered (still in flight, dead-lettered, or deleted) and were "
            "skipped"
        )

    trainable_uids_set = {uid for uid, _ in load_trainable_samples()}
    clean = [(uid, class_name) for uid, class_name in scorable
             if uid not in trainable_uids_set]
    contaminated = len(scorable) - len(clean)
    if contaminated:
        print(
            f"WARNING: {contaminated} dev-set models carry a label in the database, "
            "which makes them trainable — they were dropped, but this dev set is "
            "supposed to stay unlabeled. Check whether `make backfill-labels` ran "
            "over the ~75 uids it shares with weak_labels.csv."
        )
    if not clean:
        raise SystemExit("every scorable dev-set model is trainable — nothing independent left")
    return clean


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
        # Runs 2 through 4 predate `held_out` AND predate hash-bucketed splitting
        # (ml.md#dataset-splits), so a recomputed partition is not theirs whatever
        # the labels have done — the shuffle-and-slice scheme that produced their
        # split no longer exists. The warning is therefore unconditional here, not
        # contingent on drift; the label_hash comparison only says how *much* else
        # has moved.
        recorded_hash = snapshot.get("label_hash")
        current_hash = data_snapshot(samples, split)["label_hash"]
        drift = (
            "the labeled set has also changed since "
            f"({recorded_hash[:19]}… -> {current_hash[:19]}…)"
            if recorded_hash and recorded_hash != current_hash
            else "the labeled set is unchanged, but that does not help here"
        )
        print(
            f"WARNING: run {run_id} recorded no {dev_set} split, so this partition is "
            f"recomputed under the current hash-bucket scheme rather than the one it "
            f"actually held out — {drift}. Treat the number as indicative."
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


def score_and_record(
    model: MultiViewCNN,
    run_id: int,
    dev_set: str,
    scored: list[tuple[str, str]],
    storage: Storage,
    num_workers: int,
    backbone: str,
    scored_label_hash: str | None = None,
) -> dict:
    """Score `scored`, print the headline, store the report, return it.

    Shared by both kinds of dev set so a partition report and an external one are
    produced by the same code — the two are only comparable if nothing differs
    between them but the models.

    `scored_label_hash` defaults to a hash of the scored pairs themselves. That is
    the right fingerprint for a dev set defined outside the corpus; the partition
    path passes the trainable set's hash instead, because there the question is
    whether the corpus moved under the split.
    """
    print(f"scoring run {run_id} on {len(scored)} {dev_set} models ({backbone})")
    report = score(model, scored, storage, dev_set, num_workers=num_workers)

    # Report before storing. Scoring is the expensive part — minutes of GPU or CPU
    # over thousands of blob reads — and storing is one INSERT that can fail on a
    # transient DB error after all of it. Printing first means a failed write costs
    # the row, not the measurement.
    accuracy = report["accuracy"]
    macro_recall = report["macro_recall"]
    print(
        f"{dev_set} accuracy {accuracy:.4f}, macro recall "
        f"{macro_recall if macro_recall is None else round(macro_recall, 4)}"
    )
    if scored_label_hash is None:
        scored_label_hash = label_hash(scored)
    evaluation_id = record_evaluation(run_id, dev_set, report, scored_label_hash)
    print(f"stored as evaluation {evaluation_id}")
    return report


def evaluate_run(
    run_id: int, dev_set: str = "test", num_workers: int = 0
) -> dict:
    """Load a run, score it on `dev_set`, store the report, and return it."""
    storage = build_storage(get_settings())
    model, config, snapshot = load_run_model(run_id, storage)

    if dev_set == LVIS:
        # No split to replay and none to recompute: these objects were never in
        # the corpus the run partitioned, which is exactly what makes them a
        # second dev set rather than another view of the first.
        scored = resolve_lvis_dev_set()
        return score_and_record(
            model, run_id, dev_set, scored, storage, num_workers, config.backbone
        )

    samples = load_trainable_samples()
    # Reproduce the run's subsample before splitting. A `--limit` run held out a
    # split of its *subset*, so splitting the full trainable set would put models
    # it trained on straight into its own test set — measured on run 4: 141 of
    # 1,173 recomputed "test" models were ones the run had trained on, and the
    # score would have been inflated by exactly the amount that matters.
    limit = snapshot.get("limit")
    if limit is not None:
        samples = subsample(samples, limit, config.seed)
    # The run's own seed, not Config's default: a recomputed partition is only
    # reproducible under the seed that produced it, and a run may have set its own.
    split = stratified_split(samples, config.seed)
    current_hash = data_snapshot(samples, split)["label_hash"]
    scored = resolve_scored_samples(run_id, dev_set, snapshot, samples, split)
    if not scored:
        raise SystemExit(f"the {dev_set} split is empty — nothing to score")

    return score_and_record(
        model, run_id, dev_set, scored, storage, num_workers, config.backbone,
        current_hash,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a training run (M7).")
    parser.add_argument("--run", type=int, required=True, help="training run id")
    parser.add_argument(
        "--dev-set",
        default="test",
        choices=DEV_SETS,
        help="which dev set to score: a partition of our own corpus, or `lvis` "
             "(default: the held-out test split)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_run(args.run, args.dev_set, args.num_workers)


if __name__ == "__main__":
    main()
