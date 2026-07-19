"""HTTP push receiver for Cloud Run + Pub/Sub push (server.md#compute).

Pub/Sub **push** delivers each queue message as an HTTP POST; this endpoint
decodes the envelope and runs the download handler. A 2xx response **acks** the
message; a 5xx **nacks** it so Pub/Sub redelivers (at-least-once → the handler is
idempotent). Cloud Run enforces the OIDC auth on the push subscription, so the app
trusts requests that reach it.

Run under an ASGI server: `uvicorn app.web:app`.
"""

from __future__ import annotations

import base64
import logging

from fastapi import FastAPI, Request, Response
from fastapi.concurrency import run_in_threadpool

from .queue import decode_message
from .workers.download import process

logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/pubsub/push")
async def pubsub_push(request: Request) -> Response:
    envelope = await request.json()
    message = envelope.get("message", {})
    payload = decode_message(base64.b64decode(message.get("data", "")))
    try:
        # process() is blocking (download + DB); run it off the event loop.
        await run_in_threadpool(process, payload)
    except Exception:
        logger.exception("download failed; nacking for redelivery")
        return Response(status_code=500)  # nack
    return Response(status_code=204)  # ack
