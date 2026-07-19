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
from functools import lru_cache

from google.api_core.exceptions import AlreadyExists
from google.cloud import pubsub_v1

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


def publish_next(topic_id: str, uid: str) -> str:
    """Enqueue ``{"uid": uid}`` on `topic_id` — a stage handing off to the next.

    Used by each preprocessing stage to hand a model to the following stage
    (download → convert → normalize → render). Re-publishing on a redelivered
    job is safe: the downstream handler is idempotent and skips already-done work.
    """
    return publish_json(_publisher(), topic_id, {"uid": uid})
