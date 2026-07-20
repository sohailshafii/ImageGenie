"""Replay a stage's dead-letter queue back into its stage topic.

Pulls messages from ``<stage>-jobs-dlq-sub``, re-publishes each job to
``<stage>-jobs``, and acks the DLQ message — recovering models that failed a
stage for a *transient* reason (mirror timeouts/SSL, a since-fixed bug) once the
worker is patched. Downstream is idempotent, so replaying a job that turns out to
already be done is harmless. Genuinely-broken jobs just fail and re-dead-letter.

Publish-only + pull; no DB. Run against cloud with ``IMAGEGENIE_PUBSUB_PROJECT``
set and ADC creds:

    IMAGEGENIE_PUBSUB_PROJECT=imagegenie-pipeline python -m app.replay_dlq --stage download
"""

from __future__ import annotations

import argparse
import time

from google.cloud import pubsub_v1

from .queue import decode_message, publish_json, subscription_path

STAGES = ("download", "convert", "normalize", "render")

# The DLQ drains lazily: give up only after this many consecutive empty pulls,
# pausing between them so we don't spin on an already-empty queue.
_MAX_EMPTY_RETRIES = 3
_EMPTY_RETRY_BACKOFF_SECONDS = 2.0


def replay(stage: str, batch_size: int = 100, max_messages: int | None = None) -> int:
    """Drain ``<stage>-jobs-dlq-sub`` back to ``<stage>-jobs``; return the count replayed."""
    subscriber = pubsub_v1.SubscriberClient()
    publisher = pubsub_v1.PublisherClient()
    dlq_sub_path = subscription_path(subscriber, f"{stage}-jobs-dlq-sub")
    topic = f"{stage}-jobs"

    replayed = 0
    empty_retries = 0
    while empty_retries < _MAX_EMPTY_RETRIES and (max_messages is None or replayed < max_messages):
        response = subscriber.pull(
            request={"subscription": dlq_sub_path, "max_messages": batch_size},
            timeout=15,
        )
        if not response.received_messages:
            empty_retries += 1
            time.sleep(_EMPTY_RETRY_BACKOFF_SECONDS)
            continue
        empty_retries = 0

        ack_ids = []
        for received in response.received_messages:
            publish_json(publisher, topic, decode_message(received.message.data))
            ack_ids.append(received.ack_id)
        subscriber.acknowledge(request={"subscription": dlq_sub_path, "ack_ids": ack_ids})
        replayed += len(ack_ids)
        print(f"replayed {replayed} → {topic}")

    return replayed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True,
                        help="which stage's DLQ to replay")
    parser.add_argument("--max", type=int, default=None,
                        help="cap the number replayed (default: all)")
    args = parser.parse_args()

    total = replay(args.stage, max_messages=args.max)
    print(f"done: replayed {total} messages to '{args.stage}-jobs'")


if __name__ == "__main__":
    main()
