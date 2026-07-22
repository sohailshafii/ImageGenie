"""HTTP push receiver for Cloud Run + Pub/Sub push (server.md#compute).

Pub/Sub **push** delivers each queue message as an HTTP POST; this endpoint
decodes the envelope and runs the stage's handler. Every stage runs the *same*
image and app; ``IMAGEGENIE_STAGE`` selects which stage's ``process`` handles the
message (download / convert / normalize / render), so one service definition
serves all four. A 2xx response **acks** the message; a 5xx **nacks** it so Pub/Sub
redelivers (at-least-once → the handler is idempotent). Cloud Run enforces the OIDC
auth on the push subscription, so the app trusts requests that reach it.

Run under an ASGI server: `uvicorn app.web:app`.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.concurrency import run_in_threadpool

from .config import get_settings
from .db import init_db, session_scope
from .dead_letters import record_failure
from .models import PipelineStage
from .queue import decode_message
from .workers import convert, download, normalize, render

logger = logging.getLogger(__name__)

# One image serves every stage; the stage's env var picks the handler.
_STAGE_HANDLERS: dict[str, Callable[[dict], object]] = {
    "download": download.process,
    "convert": convert.process,
    "normalize": normalize.process,
    "render": render.process,
}


def _handler() -> Callable[[dict], object]:
    stage = get_settings().stage
    try:
        return _STAGE_HANDLERS[stage]
    except KeyError:
        raise RuntimeError(
            f"IMAGEGENIE_STAGE={stage!r} is not one of {sorted(_STAGE_HANDLERS)}"
        ) from None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Materialize the schema before serving (idempotent; skeleton bootstrap).
    init_db()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/pubsub/push")
async def pubsub_push(request: Request) -> Response:
    envelope = await request.json()
    message = envelope.get("message", {})
    payload = decode_message(base64.b64decode(message.get("data", "")))
    stage = get_settings().stage
    try:
        # process() is blocking (mesh work + DB); run it off the event loop.
        await run_in_threadpool(_handler(), payload)
    except Exception as failure:
        logger.exception("%s failed; nacking for redelivery", stage)
        # Record it here because this is the only place the error text exists —
        # a Pub/Sub dead-letter message carries the payload and a delivery count,
        # never the reason (server.md#dead-letters). Recording must not itself
        # break the nack, so its own failure is swallowed.
        try:
            with session_scope() as session:
                record_failure(
                    session,
                    uid=payload["uid"],
                    stage=PipelineStage(stage),
                    error=f"{type(failure).__name__}: {failure}",
                    delivery_attempt=envelope.get("deliveryAttempt"),
                )
        except Exception:
            logger.exception("could not record the failure for %s", payload.get("uid"))
        return Response(status_code=500)  # nack
    return Response(status_code=204)  # ack
