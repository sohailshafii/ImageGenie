"""Seed producer — publishes N download jobs to kick off the skeleton pipeline.

Enumerates the first N Objaverse uids (metadata only) and publishes one
``{"uid": ...}`` job per uid to the download topic. Ensures the topic and the
worker's pull subscription exist first, so messages are retained for the
subscription even if the worker starts later.
"""

from __future__ import annotations

import argparse

import objaverse
from google.cloud import pubsub_v1

from .config import get_settings
from .db import init_db
from .queue import ensure_subscription, publish_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100,
                        help="how many download jobs to publish (default: 100)")
    args = parser.parse_args()

    init_db()
    settings = get_settings()
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()
    ensure_subscription(
        subscriber, publisher, settings.download_subscription, settings.download_topic
    )

    uids = list(objaverse._load_object_paths())[: args.count]
    for uid in uids:
        publish_json(publisher, settings.download_topic, {"uid": uid})
    print(f"seeded {len(uids)} download jobs to '{settings.download_topic}'")


if __name__ == "__main__":
    main()
