import json
from unittest.mock import MagicMock

from app import consumer


def test_handle_message_acks_on_success() -> None:
    received: list[dict] = []
    message = MagicMock()
    message.data = json.dumps({"uid": "abc123"}).encode("utf-8")

    consumer._handle_message(message, received.append)

    assert received == [{"uid": "abc123"}]
    message.ack.assert_called_once()
    message.nack.assert_not_called()


def test_handle_message_nacks_on_failure() -> None:
    message = MagicMock()
    message.data = json.dumps({"uid": "abc123"}).encode("utf-8")

    def boom(_payload: dict) -> None:
        raise RuntimeError("handler blew up")

    consumer._handle_message(message, boom)

    message.nack.assert_called_once()  # nack -> Pub/Sub redelivers
    message.ack.assert_not_called()
