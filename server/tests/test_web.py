import base64
import json

import pytest
from fastapi.testclient import TestClient

from app import web


def _push_envelope(payload: dict) -> dict:
    """Build a Pub/Sub push HTTP body (data is base64-encoded JSON)."""
    data = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
    return {"message": {"data": data, "messageId": "1"}, "subscription": "download-worker"}


def test_push_acks_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    handled: list[dict] = []
    # Default stage is "download"; swap its handler for a recorder.
    monkeypatch.setitem(web._STAGE_HANDLERS, "download", handled.append)

    response = TestClient(web.app).post("/pubsub/push", json=_push_envelope({"uid": "abc"}))

    assert response.status_code == 204  # ack
    assert handled == [{"uid": "abc"}]


def test_push_nacks_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_payload: dict) -> None:
        raise RuntimeError("stage blew up")

    monkeypatch.setitem(web._STAGE_HANDLERS, "download", boom)

    response = TestClient(web.app).post("/pubsub/push", json=_push_envelope({"uid": "abc"}))

    assert response.status_code == 500  # nack -> Pub/Sub redelivers


def test_push_dispatches_to_configured_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import config

    handled: list[dict] = []
    monkeypatch.setattr(web, "get_settings", lambda: config.Settings(stage="render"))
    monkeypatch.setitem(web._STAGE_HANDLERS, "render", handled.append)

    response = TestClient(web.app).post("/pubsub/push", json=_push_envelope({"uid": "xyz"}))

    assert response.status_code == 204
    assert handled == [{"uid": "xyz"}]  # routed to the render handler, not download
