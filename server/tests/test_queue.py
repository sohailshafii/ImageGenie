import json

import pytest
from app import queue
from google.cloud import pubsub_v1


def test_decode_message_round_trip() -> None:
    payload = {"uid": "abc123", "source": "objaverse"}
    assert queue.decode_message(json.dumps(payload).encode("utf-8")) == payload


def test_path_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Emulator host lets the client construct without real GCP credentials.
    monkeypatch.setenv("PUBSUB_EMULATOR_HOST", "localhost:8085")
    publisher = pubsub_v1.PublisherClient()
    assert queue.topic_path(publisher, "download-jobs").endswith("/topics/download-jobs")
