"""Backend-for-frontend REST API (server.md#api-layer).

The single FastAPI app the labeling frontend talks to. This module serves the
**auth** and **models + labels** endpoints; dead-letters and upload land in later
chunks. Read endpoints resolve each model's *current* label — the most recent
`label` row, so a manual correction wins over the weak label.

Every endpoint below `/healthz` and `/auth/login` requires a session: any logged-in
user may read, only admins may write labels (FR-8). Run under an ASGI server:
`uvicorn app.api:app`.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import func, select

from .config import get_settings
from .db import init_db, session_scope
from .models import Label, LabelSource, Model, User
from .ratelimit import BackoffRule, FixedWindowRateLimiter, LoginBackoff, RateLimitRule
from .security import (
    CSRF_COOKIE,
    CSRF_HEADER,
    SESSION_COOKIE,
    SESSION_TTL,
    create_session,
    csrf_tokens_match,
    delete_session,
    generate_csrf_token,
    resolve_session,
    verify_password,
)

PAGE_SIZE_MAX = 100

# TODO(metadata-backfill): title/tags come from Objaverse annotations, which the
# download worker doesn't yet persist — placeholder until that backfill lands.


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

# ── CSRF (server.md#csrf) ───────────────────────────────────────────────────
# Methods that don't change state, so they need no token.
CSRF_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# Paths exempt from the double-submit check. Only login: it runs *before* a
# session exists and is what mints the token in the first place. Logout is
# deliberately NOT exempt — it's a state change, and a cross-site forced logout
# is exactly the nuisance CSRF protection exists to stop.
CSRF_EXEMPT_PATHS = frozenset({"/auth/login"})


@app.middleware("http")
async def enforce_csrf(request: Request, call_next):
    """Reject unsafe requests whose CSRF header doesn't match the cookie.

    Middleware rather than a per-route dependency so it **fails closed**: a new
    state-changing endpoint is protected the day it's added, and skipping the
    check has to be a deliberate edit to `CSRF_EXEMPT_PATHS`.
    """
    if (
        request.method not in CSRF_SAFE_METHODS
        and request.url.path not in CSRF_EXEMPT_PATHS
        and not csrf_tokens_match(
            request.cookies.get(CSRF_COOKIE), request.headers.get(CSRF_HEADER)
        )
    ):
        return JSONResponse(status_code=403, content={"detail": "csrf_failure"})
    return await call_next(request)


def _set_auth_cookies(response: Response, session_token: str) -> None:
    """Set the session + CSRF cookie pair with matching attributes."""
    secure = get_settings().cookie_secure
    max_age = int(SESSION_TTL.total_seconds())
    # samesite=lax already blocks the cross-site form POST; the double-submit
    # token is the second layer, covering fetch-issued requests.
    response.set_cookie(
        SESSION_COOKIE, session_token, httponly=True, secure=secure,
        samesite="lax", max_age=max_age,
    )
    response.set_cookie(
        CSRF_COOKIE, generate_csrf_token(), httponly=False, secure=secure,
        samesite="lax", max_age=max_age,
    )


# ── Rate limiting (server.md#rate-limiting) ─────────────────────────────────
# Per-IP volumetric cap on login: bounds one host sweeping many accounts. The
# per-account escalation is LOGIN_BACKOFF's job, not this one's.
LOGIN_PER_IP = RateLimitRule(max_hits=20, window_seconds=10 * 60)
# 3 free attempts (typos), then lock 1s, 2s, 4s … doubling to 15 minutes.
LOGIN_BACKOFF_RULE = BackoffRule(free_retries=3, base_seconds=1.0, max_seconds=15 * 60)
# Label writes are admin-only and admins are trusted, so this is not an abuse
# control — it is a runaway guard. Every PUT inserts a `label` row, so a looping
# frontend would otherwise grow the table without bound. Set well above human
# labeling speed (1/s sustained) so it can't interrupt a real labeling session.
LABEL_WRITE_PER_USER = RateLimitRule(max_hits=600, window_seconds=10 * 60)

login_limiter = FixedWindowRateLimiter()
label_limiter = FixedWindowRateLimiter()
login_backoff = LoginBackoff(LOGIN_BACKOFF_RULE)


def _client_ip(request: Request) -> str:
    """The caller's IP for rate-limit keying.

    Only consults `X-Forwarded-For` when configured to trust it — believing the
    header unconditionally would let a caller rotate the header per request and
    walk straight around every per-IP cap.
    """
    if get_settings().trust_proxy_headers:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()  # left-most = original client
    return request.client.host if request.client else "unknown"


def _too_many_requests(retry_after_seconds: float) -> HTTPException:
    """429 carrying `Retry-After`, so the client waits rather than hammering."""
    return HTTPException(
        status_code=429,
        detail="rate_limited",
        headers={"Retry-After": str(max(1, math.ceil(retry_after_seconds)))},
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


# ── Auth (session core) ─────────────────────────────────────────────────────
class LoginIn(BaseModel):
    email: str
    password: str


class MeOut(BaseModel):
    email: str
    role: str


class AuthUser(BaseModel):
    id: int
    email: str
    role: str


def current_user(request: Request) -> AuthUser:
    """Resolve the httpOnly session cookie to the caller, or 401. Route dependency."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="unauthorized")
    with session_scope() as session:
        user = resolve_session(session, token)
        if user is None:
            raise HTTPException(status_code=401, detail="unauthorized")
        # Capture fields while the session is open (attributes expire on commit).
        return AuthUser(id=user.id, email=user.email, role=user.role.value)


# Annotated dependency (modern FastAPI idiom) — avoids a call in an arg default.
CurrentUser = Annotated[AuthUser, Depends(current_user)]


def require_admin(user: CurrentUser) -> AuthUser:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="forbidden")
    return user


AdminUser = Annotated[AuthUser, Depends(require_admin)]

# Routes that only need the caller *authenticated* (not their identity) declare
# this in `dependencies=` rather than taking an unused parameter.
LOGIN_REQUIRED = [Depends(current_user)]


@app.post("/auth/login", response_model=MeOut)
def login(body: LoginIn, request: Request, response: Response) -> MeOut:
    email = body.email.strip().lower()  # normalized so the backoff key is stable
    ip_key = f"login:ip:{_client_ip(request)}"
    account_key = f"login:account:{email}"

    if not login_limiter.check(ip_key, LOGIN_PER_IP):
        raise _too_many_requests(login_limiter.retry_after(ip_key))
    # Checked before the DB read and before bcrypt: while locked out we do no
    # work, which is the point — bcrypt is expensive by design, so an unthrottled
    # login endpoint is a CPU-exhaustion lever as much as a guessing one.
    locked_for = login_backoff.retry_after(account_key)
    if locked_for > 0:
        raise _too_many_requests(locked_for)

    with session_scope() as session:
        user = session.scalar(select(User).where(User.email == email))
        if user is None or not verify_password(body.password, user.password_hash):
            login_backoff.record_failure(account_key)
            raise HTTPException(status_code=401, detail="invalid_credentials")
        if not user.verified:
            # Correct password — the account just isn't verified. Not a guess, so
            # it must not feed the backoff ladder.
            raise HTTPException(status_code=403, detail="unverified")
        login_backoff.record_success(account_key)
        token = create_session(session, user)
        me = MeOut(email=user.email, role=user.role.value)
    _set_auth_cookies(response, token)
    return me


@app.get("/auth/me", response_model=MeOut)
def me(user: CurrentUser) -> MeOut:
    return MeOut(email=user.email, role=user.role)


@app.post("/auth/logout")
def logout(request: Request) -> Response:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with session_scope() as session:
            delete_session(session, token)
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE)
    response.delete_cookie(CSRF_COOKIE)  # clear the pair together
    return response


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


@app.get("/models", response_model=ModelPageOut, dependencies=LOGIN_REQUIRED)
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


def _load_summary(uid: str) -> ModelSummaryOut:
    """Read one model's current label, or 404. Shared by the GET and PUT routes."""
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


@app.get("/models/{uid}", response_model=ModelSummaryOut, dependencies=LOGIN_REQUIRED)
def get_model(uid: str) -> ModelSummaryOut:
    return _load_summary(uid)


@app.put("/models/{uid}/label", response_model=ModelSummaryOut)
def set_label(uid: str, body: LabelIn, admin: AdminUser) -> ModelSummaryOut:
    """Record a **manual** label (confirm keeps the class, correct changes it).

    Admin-only (FR-8); the correction is attributed to the calling admin.
    """
    write_key = f"label:user:{admin.id}"
    if not label_limiter.check(write_key, LABEL_WRITE_PER_USER):
        raise _too_many_requests(label_limiter.retry_after(write_key))
    with session_scope() as session:
        if session.get(Model, uid) is None:
            raise HTTPException(status_code=404, detail="unknown model")
        session.add(
            Label(
                model_uid=uid,
                class_name=body.class_name,
                source=LabelSource.manual,
                confidence=None,
                annotator=admin.email,
            )
        )
    return _load_summary(uid)
