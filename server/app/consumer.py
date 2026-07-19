"""Pull-subscription consumer loop for workers (server.md#queue skeleton exception).

Streams messages from a subscription and runs a handler per message: **ack** on
success so Pub/Sub drops it, **nack** on failure so it redelivers (at-least-once).
Handlers must therefore be idempotent (NFR-2). Repeatedly-failing "poison"
messages exceed the subscription's max delivery attempts and land in the
dead-letter topic (server.md#queue).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.subscriber.message import Message

from .db import init_db
from .queue import decode_message, ensure_subscription, subscription_path

logger = logging.getLogger(__name__)

Handler = Callable[[dict], object]


def _handle_message(message: Message, handler: Handler) -> None:
    """Decode + run `handler`; ack on success, nack (for redelivery) on failure."""
    try:
        handler(decode_message(message.data))
    except Exception:
        logger.exception("handler failed; nacking for redelivery")
        message.nack()
        return
    message.ack()


def consume(subscription_id: str, handler: Handler) -> None:
    """Stream messages from `subscription_id` to `handler` until interrupted."""
    subscriber = pubsub_v1.SubscriberClient()
    path = subscription_path(subscriber, subscription_id)
    streaming_future = subscriber.subscribe(path, callback=lambda m: _handle_message(m, handler))
    logger.info("consuming %s", subscription_id)
    with subscriber:
        try:
            streaming_future.result()
        except KeyboardInterrupt:
            streaming_future.cancel()
            streaming_future.result()


def run_stage(subscription_id: str, topic_id: str, handler: Handler) -> None:
    """Bootstrap DB + subscription, then consume `subscription_id` forever.

    The shared entrypoint every worker's ``main()`` calls: materialize the schema
    (idempotent), ensure the stage's topic + pull subscription exist, and stream
    jobs to `handler`. Locally this is a pull consumer (server.md#queue skeleton
    exception); prod preprocessing stages instead receive Pub/Sub push over HTTP.
    """
    init_db()
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()
    ensure_subscription(subscriber, publisher, subscription_id, topic_id)
    logger.info("worker consuming %s", subscription_id)
    consume(subscription_id, handler)
