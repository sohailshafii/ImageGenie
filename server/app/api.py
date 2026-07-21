"""Backend-for-frontend REST API (server.md#api-layer).

The single FastAPI app the labeling frontend talks to. This module serves the
**models + labels** endpoints; auth, dead-letters, and upload land in later
chunks. Read endpoints resolve each model's *current* label — the most recent
`label` row, so a manual correction wins over the weak label. Run under an ASGI
server: `uvicorn app.api:app`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select

from .db import init_db, session_scope
from .models import Label, LabelSource, Model

PAGE_SIZE_MAX = 100

# TODO(metadata-backfill): title/tags come from Objaverse annotations, which the
# download worker doesn't yet persist — placeholder until that backfill lands.
# TODO(auth): PUT /label should be admin-gated + use the caller as annotator once
# the auth chunk exists; hardcoded for now.
_PLACEHOLDER_ANNOTATOR = "admin@imagegenie.dev"


class ModelSummaryOut(BaseModel):
    uid: str
    title: str
    tags: list[str]
    class_name: str | None  # None until the model is labeled
    source: str | None
    confidence: float | None


class ModelPageOut(BaseModel):
    items: list[ModelSummaryOut]
    total: int
    page: int
    page_size: int


class LabelIn(BaseModel):
    class_name: str


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    init_db()  # materialize schema (idempotent) before serving
    yield


app = FastAPI(lifespan=lifespan, title="ImageGenie API")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


def _latest_labels():
    """Subquery of the most-recent label per model — manual wins over weak."""
    return (
        select(Label.model_uid, Label.class_name, Label.source, Label.confidence)
        .distinct(Label.model_uid)
        # id.desc() breaks ties when a weak + manual label share a created_at:
        # the later-inserted (manual) row wins.
        .order_by(Label.model_uid, Label.created_at.desc(), Label.id.desc())
        .subquery()
    )


def _summary(uid: str, class_name, source, confidence) -> ModelSummaryOut:
    return ModelSummaryOut(
        uid=uid,
        title=f"model {uid[:8]}",
        tags=[],
        class_name=class_name,
        source=source.value if source is not None else None,
        confidence=confidence,
    )


@app.get("/models", response_model=ModelPageOut)
def list_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=PAGE_SIZE_MAX),
    class_name: str | None = None,
    source: LabelSource | None = None,
) -> ModelPageOut:
    latest = _latest_labels()
    query = select(
        Model.uid, latest.c.class_name, latest.c.source, latest.c.confidence
    ).outerjoin(latest, Model.uid == latest.c.model_uid)
    if class_name is not None:
        query = query.where(latest.c.class_name == class_name)
    if source is not None:
        query = query.where(latest.c.source == source)

    with session_scope() as session:
        total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
        rows = session.execute(
            query.order_by(Model.uid).limit(page_size).offset((page - 1) * page_size)
        ).all()
        items = [_summary(*row) for row in rows]
    return ModelPageOut(items=items, total=total, page=page, page_size=page_size)


@app.get("/models/{uid}", response_model=ModelSummaryOut)
def get_model(uid: str) -> ModelSummaryOut:
    latest = _latest_labels()
    with session_scope() as session:
        row = session.execute(
            select(Model.uid, latest.c.class_name, latest.c.source, latest.c.confidence)
            .outerjoin(latest, Model.uid == latest.c.model_uid)
            .where(Model.uid == uid)
        ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown model")
    return _summary(*row)


@app.put("/models/{uid}/label", response_model=ModelSummaryOut)
def set_label(uid: str, body: LabelIn) -> ModelSummaryOut:
    """Record a **manual** label (confirm keeps the class, correct changes it)."""
    with session_scope() as session:
        if session.get(Model, uid) is None:
            raise HTTPException(status_code=404, detail="unknown model")
        session.add(
            Label(
                model_uid=uid,
                class_name=body.class_name,
                source=LabelSource.manual,
                confidence=None,
                annotator=_PLACEHOLDER_ANNOTATOR,
            )
        )
    return get_model(uid)
