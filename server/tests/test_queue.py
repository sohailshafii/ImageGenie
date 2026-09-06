import json

import pytest
from google.cloud import pubsub_v1

from app import queue
from app.artifact_keys import TEXTURED_VARIANT


def test_decode_message_round_trip() -> None:
    payload = {"uid": "abc123", "source": "objaverse"}
    assert queue.decode_message(json.dumps(payload).encode("utf-8")) == payload


def test_path_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Emulator host lets the client construct without real GCP credentials.
    monkeypatch.setenv("PUBSUB_EMULATOR_HOST", "localhost:8085")
    publisher = pubsub_v1.PublisherClient()
    assert queue.topic_path(publisher, "download-jobs").endswith("/topics/download-jobs")


def test_publish_next_omits_the_variant_for_the_default_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary job's payload is byte-identical to what it was before variants.

    Messages already in flight during a deploy stay valid, and nothing downstream
    has to learn a field it will almost never see.
    """
    published: list[tuple[str, dict]] = []
    monkeypatch.setattr(queue, "_publisher", lambda: None)
    monkeypatch.setattr(
        queue,
        "publish_json",
        lambda publisher, topic_id, payload: published.append((topic_id, payload)) or "id",
    )

    queue.publish_next("convert-jobs", "abc123")
    queue.publish_next("convert-jobs", "abc123", TEXTURED_VARIANT)

    assert published == [
        ("convert-jobs", {"uid": "abc123"}),
        ("convert-jobs", {"uid": "abc123", "variant": TEXTURED_VARIANT}),
    ]
