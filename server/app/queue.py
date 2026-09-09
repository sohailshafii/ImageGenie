"""Pub/Sub helpers for the pipeline queue (server.md#queue).

Thin wrappers over ``google-cloud-pubsub``: create topics/subscriptions
idempotently and publish/receive JSON job payloads. Locally these target the
Pub/Sub **emulator** (the client auto-detects ``PUBSUB_EMULATOR_HOST``); in cloud
they hit managed Pub/Sub with no code change. Job payloads are JSON dicts.

The skeleton's download worker uses a **pull** subscription (see
server.md#queue): the cloud objection to pull is scale-to-zero cost, which does
not apply locally, and the download stage is a batch consumer anyway.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from functools import lru_cache

from google.api_core.exceptions import AlreadyExists
from google.cloud import pubsub_v1

from .artifact_keys import DEFAULT_VARIANT
from .config import get_settings


def topic_path(publisher: pubsub_v1.PublisherClient, topic_id: str) -> str:
    return publisher.topic_path(get_settings().pubsub_project, topic_id)


def subscription_path(subscriber: pubsub_v1.SubscriberClient, subscription_id: str) -> str:
    return subscriber.subscription_path(get_settings().pubsub_project, subscription_id)


def ensure_topic(publisher: pubsub_v1.PublisherClient, topic_id: str) -> str:
    """Create the topic if absent; return its path. Idempotent."""
    path = topic_path(publisher, topic_id)
    try:
        publisher.create_topic(name=path)
    except AlreadyExists:
        pass
    return path


def ensure_subscription(
    subscriber: pubsub_v1.SubscriberClient,
    publisher: pubsub_v1.PublisherClient,
    subscription_id: str,
    topic_id: str,
) -> str:
    """Create the pull subscription (and its topic) if absent; return its path."""
    path = subscription_path(subscriber, subscription_id)
    try:
        subscriber.create_subscription(name=path, topic=ensure_topic(publisher, topic_id))
    except AlreadyExists:
        pass
    return path


def publish_json(
    publisher: pubsub_v1.PublisherClient, topic_id: str, payload: dict
) -> str:
    """Publish `payload` as a JSON message; return the assigned message id."""
    data = json.dumps(payload).encode("utf-8")
    return publisher.publish(topic_path(publisher, topic_id), data).result()


def decode_message(data: bytes) -> dict:
    """Decode a Pub/Sub message's data bytes back into the JSON payload dict."""
    return json.loads(data.decode("utf-8"))


@lru_cache
def _publisher() -> pubsub_v1.PublisherClient:
    """Process-wide publisher, reused across messages (opened lazily)."""
    return pubsub_v1.PublisherClient()


def job_payload(uid: str, variant: str = DEFAULT_VARIANT) -> dict:
    """One stage job. The default variant writes no ``variant`` field at all, so an
    ordinary payload is byte-identical to what it was before variants existed —
    messages already in flight during a deploy stay valid.

    One definition because both the stage hand-off and the bulk seeder build these,
    and a payload shape that differs between them is the kind of drift that shows
    up as an arm quietly processing under the wrong variant.
    """
    payload = {"uid": uid}
    if variant != DEFAULT_VARIANT:
        payload["variant"] = variant
    return payload


def publish_next(topic_id: str, uid: str, variant: str = DEFAULT_VARIANT) -> str:
    """Enqueue ``{"uid": uid}`` on `topic_id` — a stage handing off to the next.

    Used by each preprocessing stage to hand a model to the following stage
    (download → convert → normalize → render). Re-publishing on a redelivered
    job is safe: the downstream handler is idempotent and skips already-done work.

    `variant` rides along so a stage hands the *same* arm to the next one
    (app/artifact_keys.py).
    """
    return publish_json(_publisher(), topic_id, job_payload(uid, variant))


# How a bulk publish paces itself. A stage runs one model per instance and scales
# to 15, so a loop that publishes thousands of jobs as fast as it can is asking a
# service that holds fifteen requests to hold thousands. Cloud Run aborts what it
# cannot place, Pub/Sub counts each abort as a failed delivery, and the overflow
# dead-letters: 2,029 of 3,677 jobs on 2026-09-09, and 499 of 1,000 in the same
# way a year earlier. Higher `max_delivery_attempts` (infra/preprocessing.tf) is
# the real fix, because it stops congestion being mistaken for poison; this is the
# cheap half that keeps the burst from happening in the first place.
PUBLISH_BATCH_SIZE = 250
PUBLISH_PAUSE_SECONDS = 20.0


def publish_paced(
    topic_id: str,
    payloads: Sequence[dict],
    batch_size: int = PUBLISH_BATCH_SIZE,
    pause_seconds: float = PUBLISH_PAUSE_SECONDS,
) -> int:
    """Publish `payloads` in batches, pausing between them; return how many landed.

    Deliberately a wall-clock pause rather than anything adaptive. The consumer's
    capacity is a deployment fact this process cannot see, and a feedback loop that
    guessed at it would be one more thing to get wrong in the middle of a paid run.
    A pause long enough to matter costs minutes on a run that takes an hour.

    No pause after the final batch — there is nothing left to pace.
    """
    publisher = _publisher()
    published = 0
    for start in range(0, len(payloads), batch_size):
        batch = payloads[start : start + batch_size]
        for payload in batch:
            publish_json(publisher, topic_id, payload)
        published += len(batch)
        if published < len(payloads):
            print(f"  published {published:,}/{len(payloads):,}, pausing {pause_seconds:.0f}s")
            time.sleep(pause_seconds)
    return published
