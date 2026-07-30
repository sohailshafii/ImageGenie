"""Build the hand-labeling queue for the active-learning loop (M8).

Scores a finished run against a split and keeps the models where the classifier
and the stored label **disagree**. Each disagreement is one of exactly two
things: the model is wrong, or the label is wrong. A human deciding which is the
only way to tell them apart — and that split is what the bias analysis
(ml.md#bias-analysis) needs, because it separates real model error from the
weak-label ceiling that no amount of training removes.

Deliberately not "the least-confident models", which is the textbook
active-learning queue and what the browse UI already offers. Low confidence finds
models the classifier is unsure about; disagreement finds models where somebody
is *demonstrably* wrong. On a corpus whose labels are ~9% wrong, the second is
the better use of a human's attention.

    make review-queue RUN=14                 # the run's own held-out test split
    make review-queue RUN=14 SPLIT=val LIMIT=100

Writes `data/m8/review_queue.csv`, one row per disagreement, with a clickable URL
per model and two blank columns — `proposed_label` and `verdict` — for the
reviewer to fill in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from infer import load_run_model, rank_samples
from io_utils import write_csv
from splits import stratified_split
from train import load_trainable_samples, subsample

from app.config import get_settings
from app.storage import build_storage

OUT_PATH = Path("data/m8/review_queue.csv")

# Scoring a split is read-latency-bound, not compute-bound: every model is twelve
# separate view reads from the bucket. Unlike `make evaluate`, which defaults to a
# single process and is told otherwise for real data, this command only ever runs
# over a whole split — so it defaults to loading in parallel.
DEFAULT_WORKERS = 4

HEADER = (
    "uid",
    "url",
    "weak_label",  # what the model trained against
    "predicted",  # what the classifier says now
    "confidence",
    "runner_up",
    "runner_up_confidence",
    "proposed_label",  # filled in by the reviewer
    "verdict",  # label_wrong | model_wrong | unclear
)


def held_out_uids(snapshot: dict, dev_set: str) -> list[str] | None:
    """The uids a run recorded for `dev_set`, or None if it predates the field."""
    return (snapshot.get("held_out") or {}).get(dev_set)


def review_rows(
    run_id: int,
    dev_set: str,
    limit: int | None,
    base_url: str,
    num_workers: int = DEFAULT_WORKERS,
) -> list[dict]:
    """Score `dev_set` and return a row per model/label disagreement."""
    storage = build_storage(get_settings())
    model, config, snapshot = load_run_model(run_id, storage)

    samples = load_trainable_samples()
    if snapshot.get("limit") is not None:
        samples = subsample(samples, snapshot["limit"], config.seed)
    uid_to_label = dict(samples)

    uids = held_out_uids(snapshot, dev_set)
    if uids is None:
        # Runs 2-4 predate `held_out` and the hash-bucket scheme both, so a
        # recomputed partition is not theirs regardless of label drift — same
        # unconditional caveat as evaluation (ml.md#scoring-a-finished-run).
        split = stratified_split(samples, config.seed)
        print(
            f"WARNING: run {run_id} recorded no {dev_set} split, so this queue is drawn "
            "from a partition recomputed under the current scheme, not the one it held out."
        )
        uids = [uid for uid, _ in getattr(split, dev_set)]

    # A uid can leave the trainable set after the run — soft-deleted, label
    # removed, renders gone — and scoring it would fault on a missing view.
    scored = [(uid, uid_to_label[uid]) for uid in uids if uid in uid_to_label]
    print(f"scoring {len(scored)} {dev_set} models from run {run_id}", flush=True)

    rows = []
    ranked_samples = rank_samples(model, scored, storage, num_workers=num_workers)
    for index, (uid, label, ranked) in enumerate(ranked_samples, 1):
        if index % 200 == 0:
            print(f"  scored {index}/{len(scored)}", flush=True)
        predicted, confidence = ranked[0]
        if predicted == label:
            continue
        rows.append(
            {
                "uid": uid,
                "url": f"{base_url.rstrip('/')}/models/{uid}",
                "weak_label": label,
                "predicted": predicted,
                "confidence": round(confidence, 4),
                "runner_up": ranked[1][0],
                "runner_up_confidence": round(ranked[1][1], 4),
                "proposed_label": "",
                "verdict": "",
            }
        )

    # Most-confident disagreements first: those are where the classifier is
    # committing hardest against the label, so they are the most informative to
    # adjudicate — and the least likely to be a coin-flip either way.
    rows.sort(key=lambda row: row["confidence"], reverse=True)
    return rows[:limit] if limit else rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the M8 review queue.")
    parser.add_argument("--run", type=int, required=True, help="training run id")
    parser.add_argument("--split", default="test", choices=("test", "val", "train"))
    parser.add_argument(
        "--limit", type=int, default=None, help="keep only the N most confident"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="app origin for the per-model links (default: IMAGEGENIE_APP_BASE_URL)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"view-loading processes (default: {DEFAULT_WORKERS}; 0 loads in-process)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_url = args.base_url or get_settings().app_base_url
    rows = review_rows(args.run, args.split, args.limit, base_url, args.num_workers)
    if not rows:
        raise SystemExit("no disagreements — nothing to review")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_PATH, HEADER, [tuple(row[column] for column in HEADER) for row in rows])
    print(f"{len(rows)} disagreements -> {OUT_PATH}")


if __name__ == "__main__":
    main()
