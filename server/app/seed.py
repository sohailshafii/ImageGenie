"""Seed producer — publishes download jobs to kick off the pipeline.

Publishes one ``{"uid": ...}`` job per uid to the download topic; each flows
through download → convert → normalize → render. Two uid sources:

- ``--from-labels weak_labels.csv`` — the **labeled 12-class set** (the real
  ingestion input, produced by ``ml/weak_label.py``). This is what a data run uses.
- default — the first ``--count`` **arbitrary** Objaverse uids (metadata only), for
  a quick pipeline/infra check with no labeling relevance.

``--count`` caps how many jobs to publish (pilot a few hundred before a full run).
Ensures the topic + subscription exist first so messages are retained even if a
consumer starts later. Publish-only: it does not touch the DB (workers create the
schema), so it can target cloud Pub/Sub from a laptop — set
``IMAGEGENIE_PUBSUB_PROJECT`` (and rely on ADC) to seed the deployed pipeline.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import objaverse
from google.cloud import pubsub_v1

from .config import get_settings
from .queue import ensure_subscription, publish_json


def _labeled_uids(labels_path: Path, count: int | None) -> list[str]:
    """Read uids from a ``weak_labels.csv`` (``uid,class,reason``); cap to `count`."""
    with labels_path.open(newline="", encoding="utf-8") as csv_file:
        uids = [row["uid"] for row in csv.DictReader(csv_file)]
    return uids[:count] if count is not None else uids


def _arbitrary_uids(count: int) -> list[str]:
    """The first `count` Objaverse uids (metadata only) — no labeling relevance."""
    return list(objaverse._load_object_paths())[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100,
                        help="how many download jobs to publish (default: 100)")
    parser.add_argument("--from-labels", type=Path, default=None,
                        help="publish uids from a weak_labels.csv (the labeled 12-class "
                             "set); omit to publish arbitrary Objaverse uids")
    args = parser.parse_args()

    settings = get_settings()
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()
    ensure_subscription(
        subscriber, publisher, settings.download_subscription, settings.download_topic
    )

    if args.from_labels is not None:
        uids = _labeled_uids(args.from_labels, args.count)
        source = f"{args.from_labels} (labeled set)"
    else:
        uids = _arbitrary_uids(args.count)
        source = "arbitrary Objaverse uids"

    for uid in uids:
        publish_json(publisher, settings.download_topic, {"uid": uid})
    print(f"seeded {len(uids):,} download jobs to '{settings.download_topic}' from {source}")


if __name__ == "__main__":
    main()
