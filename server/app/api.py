"""Backend-for-frontend REST API (server.md#api-layer).

The single FastAPI app the labeling frontend talks to: **auth**, **models +
labels**, **artifacts**, **dead-letters**, and admin **upload**. Read endpoints
resolve each model's *current* label — the most recent `label` row, so a manual
correction wins over the weak label.

Access model (FR-8): `/healthz` and the pre-auth flows (login, invite-gated
signup, email verification and its resend) are public; everything else needs a
session, and label writes plus invite creation need the `admin` role. Run under
an ASGI server: `uvicorn app.api:app`.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .artifact_keys import (
    RAW_SUFFIX_TO_FILE_TYPE,
    normalized_key,
    raw_key,
    view_key,
    view_keys,
)
from .config import get_settings
from .db import init_db, session_scope
from .dead_letters import list_dead_letters, replay
from .mail import send_invite_email, send_verification_email
from .models import (
    DownloadStatus,
    EmailVerification,
    Invite,
    Label,
    LabelSource,
    Model,
    User,
    UserRole,
)
from .queue import publish_next
from .ratelimit import BackoffRule, FixedWindowRateLimiter, LoginBackoff, RateLimitRule
from .security import (
    CSRF_COOKIE,
    CSRF_HEADER,
    INVITE_TTL,
    SESSION_COOKIE,
    SESSION_TTL,
    VERIFICATION_TTL,
    create_session,
    csrf_tokens_match,
    delete_session,
    generate_csrf_token,
    generate_token,
    hash_password,
    hash_token,
    resolve_session,
    verify_password,
)
from .storage import build_storage

logger = logging.getLogger(__name__)

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
    # First rendered view, for the grid. Emitted **without** checking the blob
    # exists: a 24-card page would otherwise cost 24 round-trips to object
    # storage just to draw thumbnails. Signing is local, so this is free; the
    # client falls back to a placeholder if the image 404s.
    thumbnail: str | None


class ModelPageOut(BaseModel):
    items: list[ModelSummaryOut]
    total: int
    page: int
    page_size: int


class LabelIn(BaseModel):
    class_name: str


class DeadLetterOut(BaseModel):
    id: int
    uid: str
    stage: str
    error: str
    delivery_attempt: int | None
    failed_at: datetime
    replayed_at: datetime | None


class ModelSort(str, enum.Enum):
    """Browse ordering. `confidence` is least-confident-first — the review queue."""

    uid = "uid"
    confidence = "confidence"


class ModelArtifactsOut(BaseModel):
    uid: str
    views: list[str]  # rendered view URLs, in view order; empty if not yet rendered
    mesh: str | None  # normalized PLY, or None if the stage hasn't run


# Short by design: a signed URL is readable by anyone holding it, so it should
# outlive a page render and not much else.
ARTIFACT_URL_TTL = timedelta(minutes=15)

_ARTIFACT_MEDIA_TYPES = {".png": "image/png", ".ply": "application/octet-stream"}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    init_db()  # materialize schema (idempotent) before serving
    yield


app = FastAPI(lifespan=lifespan, title="ImageGenie API")

# ── CSRF (server.md#csrf) ───────────────────────────────────────────────────
# Methods that don't change state, so they need no token.
CSRF_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# Paths exempt from the double-submit check: the pre-auth flows, which run
# *before* any session or CSRF cookie exists. Exempting them is safe precisely
# because they are unauthenticated — CSRF abuses a victim's ambient credentials,
# and these endpoints don't consult any. Logout is deliberately NOT exempt: it is
# a state change on a live session, and a cross-site forced logout is exactly the
# nuisance CSRF protection exists to stop. Nor is /auth/invites — it's admin-only.
CSRF_EXEMPT_PATHS = frozenset(
    {
        "/auth/login",
        "/auth/signup",
        "/auth/verify-email",
        "/auth/verify-email/resend",
    }
)


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
# Signup is invite-gated, but each attempt still costs a bcrypt hash.
SIGNUP_PER_IP = RateLimitRule(max_hits=10, window_seconds=10 * 60)
# Verification tokens are 256-bit random, so guessing is hopeless — but an
# unthrottled token endpoint is still a free oracle, and leaving one uncapped is a
# gap worth not shipping.
VERIFY_PER_IP = RateLimitRule(max_hits=20, window_seconds=10 * 60)
# Resend triggers an outbound email, so it is capped on both dimensions: per IP
# (one host spraying) and per address (mailbox-bombing one victim).
RESEND_PER_IP = RateLimitRule(max_hits=5, window_seconds=10 * 60)
RESEND_PER_EMAIL = RateLimitRule(max_hits=5, window_seconds=10 * 60)
INVITE_PER_ADMIN = RateLimitRule(max_hits=50, window_seconds=10 * 60)
# Uploads are far heavier than a label write (a parse plus an object write), so
# the cap is a runaway guard on an admin-only route, not abuse control.
UPLOAD_PER_ADMIN = RateLimitRule(max_hits=120, window_seconds=10 * 60)

login_limiter = FixedWindowRateLimiter()
label_limiter = FixedWindowRateLimiter()
signup_limiter = FixedWindowRateLimiter()
upload_limiter = FixedWindowRateLimiter()
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


# Mirrors the frontend's mock contract (web/src/api/types.ts) so swapping the
# mock for this API doesn't change any component.
PASSWORD_MIN_LENGTH = 8


class SignupIn(BaseModel):
    email: str
    password: str


class VerifyEmailIn(BaseModel):
    token: str


class ResendIn(BaseModel):
    email: str


class InviteIn(BaseModel):
    email: str


class InviteOut(BaseModel):
    email: str
    expires_at: datetime
    accepted: bool


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


def _rate_limit(limiter: FixedWindowRateLimiter, key: str, rule: RateLimitRule) -> None:
    """Consume one hit for `key`, raising 429 with `Retry-After` if over the cap."""
    if not limiter.check(key, rule):
        raise _too_many_requests(limiter.retry_after(key))


def _issue_verification(session: Session, user: User, background: BackgroundTasks) -> None:
    """Replace any outstanding verification token for `user` and email a new one.

    Replacing rather than appending keeps exactly one live token per account, so
    a resend invalidates the previous link and the table can't be grown by
    repeatedly asking for one.

    The send is queued as a background task, so a slow mail provider doesn't hold
    the response open — and, because tasks run only after a successful response,
    a rolled-back signup never emails a link for an account that doesn't exist.
    """
    session.execute(delete(EmailVerification).where(EmailVerification.user_id == user.id))
    token = generate_token()
    session.add(
        EmailVerification(
            token_hash=hash_token(token),
            user_id=user.id,
            expires_at=datetime.now(UTC) + VERIFICATION_TTL,
        )
    )
    background.add_task(send_verification_email, user.email, token)


@app.post("/auth/signup", status_code=204)
def signup(body: SignupIn, request: Request, background: BackgroundTasks) -> Response:
    """Create an unverified account. **Invite-gated** — FR-8 keeps signup closed.

    Error ordering is deliberate: the invite is checked *first*, so a caller with
    no invite for the address learns nothing about whether an account exists.
    `email_taken` is only reachable once an invite record exists for that email,
    i.e. by someone who already knows an admin invited it.
    """
    _rate_limit(signup_limiter, f"signup:ip:{_client_ip(request)}", SIGNUP_PER_IP)
    email = body.email.strip().lower()
    if len(body.password) < PASSWORD_MIN_LENGTH:
        raise HTTPException(status_code=400, detail="validation_error")

    with session_scope() as session:
        invite = session.get(Invite, email)
        if invite is None or invite.expires_at < datetime.now(UTC):
            raise HTTPException(status_code=403, detail="invite_required")
        if session.scalar(select(User).where(User.email == email)) is not None:
            raise HTTPException(status_code=409, detail="email_taken")
        if invite.accepted:
            raise HTTPException(status_code=403, detail="invite_required")

        user = User(
            email=email,
            role=UserRole.user,  # invites never grant admin; promotion is manual
            password_hash=hash_password(body.password),
            verified=False,
        )
        session.add(user)
        session.flush()  # assign user.id before the verification row FKs it
        invite.accepted = True
        _issue_verification(session, user, background)
    return Response(status_code=204)


@app.post("/auth/verify-email", status_code=204)
def verify_email(body: VerifyEmailIn, request: Request) -> Response:
    """Consume a one-time verification token and mark the account verified."""
    _rate_limit(signup_limiter, f"verify:ip:{_client_ip(request)}", VERIFY_PER_IP)
    # The failure is recorded and raised *after* the transaction commits, not
    # inside it: raising within `session_scope` rolls the block back, which would
    # undo the delete below and leave a spent token replayable.
    failure: str | None = None
    with session_scope() as session:
        record = session.get(EmailVerification, hash_token(body.token))
        if record is None:
            failure = "invalid_token"
        else:
            expired = record.expires_at < datetime.now(UTC)
            user = session.get(User, record.user_id)
            # Consumed either way: a one-time token must not survive its own use,
            # and an expired one is spent rather than left to linger.
            session.delete(record)
            if expired:
                failure = "expired_token"
            elif user is None:
                failure = "invalid_token"
            else:
                user.verified = True
    if failure is not None:
        raise HTTPException(status_code=400, detail=failure)
    return Response(status_code=204)


@app.post("/auth/verify-email/resend", status_code=204)
def resend_verification(
    body: ResendIn, request: Request, background: BackgroundTasks
) -> Response:
    """Re-issue a verification link.

    Always 204, whatever the address: a status that varied would turn this into
    an account-existence oracle, and it is reachable without logging in.
    """
    email = body.email.strip().lower()
    _rate_limit(signup_limiter, f"resend:ip:{_client_ip(request)}", RESEND_PER_IP)
    _rate_limit(signup_limiter, f"resend:email:{email}", RESEND_PER_EMAIL)
    with session_scope() as session:
        user = session.scalar(select(User).where(User.email == email))
        if user is not None and not user.verified:
            _issue_verification(session, user, background)
    return Response(status_code=204)


@app.post("/auth/invites", response_model=InviteOut, status_code=201)
def create_invite(
    body: InviteIn, admin: AdminUser, background: BackgroundTasks
) -> InviteOut:
    """Mint an email-bound signup invite. Admin-only (FR-8).

    Idempotent per email — re-inviting refreshes the existing invite rather than
    accumulating rows, matching the frontend's contract.
    """
    _rate_limit(signup_limiter, f"invite:admin:{admin.id}", INVITE_PER_ADMIN)
    email = body.email.strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="validation_error")

    expires_at = datetime.now(UTC) + INVITE_TTL
    with session_scope() as session:
        invite = session.get(Invite, email)
        if invite is None:
            invite = Invite(email=email)
            session.add(invite)
        invite.expires_at = expires_at
        invite.accepted = False  # re-inviting reopens a spent invite
        invite.invited_by = admin.email
    background.add_task(send_invite_email, email, admin.email)
    return InviteOut(email=email, expires_at=expires_at, accepted=False)


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


def _thumbnail_url(storage, uid: str) -> str:
    """URL for a model's first rendered view. No existence check — see the field."""
    key = view_key(uid, 0)
    return storage.signed_url(key, ARTIFACT_URL_TTL) or f"/artifacts/{key}"


def _summary(storage, uid, title, tags, class_name, source, confidence) -> ModelSummaryOut:
    return ModelSummaryOut(
        uid=uid,
        # Falls back to the uid until `app.backfill_metadata` has run — a card
        # with no caption at all would be worse than a dull one.
        title=title or f"model {uid[:8]}",
        tags=tags or [],
        class_name=class_name,
        source=source.value if source is not None else None,
        confidence=confidence,
        thumbnail=_thumbnail_url(storage, uid),
    )


# Selected by both the list and detail queries, so the two can't drift apart.
_SUMMARY_COLUMNS = (Model.uid, Model.title, Model.tags)


@app.get("/dead-letters", response_model=list[DeadLetterOut])
def list_failures(admin: AdminUser, include_replayed: bool = False) -> list[DeadLetterOut]:
    """Jobs that failed a pipeline stage. Admin-only — it's operational detail.

    A plain DB read: the rows were recorded by the workers at nack time, so this
    never touches Pub/Sub (server.md#dead-letters).
    """
    with session_scope() as session:
        return [
            DeadLetterOut(
                id=row.id,
                uid=row.model_uid,
                stage=row.stage.value,
                error=row.error,
                delivery_attempt=row.delivery_attempt,
                failed_at=row.failed_at,
                replayed_at=row.replayed_at,
            )
            for row in list_dead_letters(session, include_replayed)
        ]


@app.post("/dead-letters/{dead_letter_id}/retry", status_code=204)
def retry_failure(dead_letter_id: int, admin: AdminUser) -> Response:
    """Re-enqueue a failed job on its stage topic.

    Safe to do freely: every stage is idempotent (NFR-2), so replaying a job that
    turns out to have succeeded is a no-op rather than duplicate work.
    """
    with session_scope() as session:
        if replay(session, dead_letter_id) is None:
            raise HTTPException(status_code=404, detail="unknown dead letter")
        logger.info("replayed dead letter %d by %s", dead_letter_id, admin.email)
    return Response(status_code=204)


@app.get("/models", response_model=ModelPageOut, dependencies=LOGIN_REQUIRED)
def list_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=PAGE_SIZE_MAX),
    class_name: str | None = None,
    source: LabelSource | None = None,
    sort: ModelSort = ModelSort.uid,
) -> ModelPageOut:
    latest = _latest_labels()
    query = select(
        *_SUMMARY_COLUMNS, latest.c.class_name, latest.c.source, latest.c.confidence
    ).outerjoin(latest, Model.uid == latest.c.model_uid)
    if class_name is not None:
        query = query.where(latest.c.class_name == class_name)
    if source is not None:
        query = query.where(latest.c.source == source)

    if sort is ModelSort.confidence:
        # Least-confident first — the review queue (web.md, and what the
        # active-learning loop wants). NULLS LAST is explicit rather than relying
        # on Postgres' default: a NULL confidence means a manual label (already
        # reviewed) or no label at all, and neither belongs above a genuinely
        # uncertain weak label.
        order = (latest.c.confidence.asc().nulls_last(), Model.uid)
    else:
        order = (Model.uid,)
    # `Model.uid` always tie-breaks. Without it, ordering by confidence alone is
    # non-deterministic across queries — 1,141 models share figure's 0.622 — and
    # paging would silently repeat and skip rows.

    storage = build_storage(get_settings())  # built once, reused for every row
    with session_scope() as session:
        total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
        rows = session.execute(
            query.order_by(*order).limit(page_size).offset((page - 1) * page_size)
        ).all()
        items = [_summary(storage, *row) for row in rows]
    return ModelPageOut(items=items, total=total, page=page, page_size=page_size)


def _load_summary(uid: str) -> ModelSummaryOut:
    """Read one model's current label, or 404. Shared by the GET and PUT routes."""
    latest = _latest_labels()
    with session_scope() as session:
        row = session.execute(
            select(*_SUMMARY_COLUMNS, latest.c.class_name, latest.c.source, latest.c.confidence)
            .outerjoin(latest, Model.uid == latest.c.model_uid)
            .where(Model.uid == uid)
        ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown model")
    return _summary(build_storage(get_settings()), *row)


@app.get("/models/{uid}", response_model=ModelSummaryOut, dependencies=LOGIN_REQUIRED)
def get_model(uid: str) -> ModelSummaryOut:
    return _load_summary(uid)


def _validated_upload_suffix(filename: str | None) -> str:
    """The raw-key suffix for an uploaded filename, or 415 if we can't ingest it.

    Rejecting here rather than downstream is the point: an unsupported format that
    reached the queue would fail inside the convert worker and surface as a
    dead-letter minutes later, with no way to tell the admin why.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in RAW_SUFFIX_TO_FILE_TYPE:
        supported = ", ".join(sorted(RAW_SUFFIX_TO_FILE_TYPE))
        raise HTTPException(
            status_code=415,
            detail=f"unsupported format '{suffix or filename}' — upload one of: {supported}",
        )
    return suffix


def _read_within_limit(upload: UploadFile, max_bytes: int) -> bytes:
    """Read the upload, refusing anything over `max_bytes` with a 413.

    Reads in chunks and stops at the ceiling instead of loading the whole body
    first, so an oversized file can't exhaust memory on the way to being
    rejected.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := upload.file.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"file exceeds the {max_bytes // (1024 * 1024)} MiB upload limit",
            )
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="the uploaded file is empty")
    return b"".join(chunks)


# `load_mesh` raises these two itself, and they describe the file in terms the
# uploader can act on. Every *other* failure is a parser internal ("buffer size
# must be a multiple of element size"), which is noise to an admin — so those are
# logged and replaced with a generic explanation.
_MESH_REASONS_WORTH_SHOWING = {"scene has no geometry", "mesh has no faces"}


def _reject_unloadable_mesh(data: bytes, file_type: str, filename: str | None) -> None:
    """Fail an upload whose bytes aren't a usable mesh, before it reaches the queue.

    trimesh is imported here rather than at module scope so the API doesn't pull
    the mesh stack in at startup — `artifact_keys` stays deliberately light for
    the same reason.

    Worth the parse cost on this route: it is admin-only and rate-limited, and the
    alternative is a corrupt file being accepted with 201, then dying in the
    convert worker with nothing tying the dead-letter back to the person who
    uploaded it.
    """
    from .workers.mesh import load_mesh

    try:
        load_mesh(data, file_type=file_type)
    except Exception as error:  # noqa: BLE001 — any parse failure is a bad upload
        reason = str(error)
        if reason in _MESH_REASONS_WORTH_SHOWING:
            detail = f"the file has no usable geometry ({reason})"
        else:
            # The real parser error is worth keeping, just not worth showing.
            logger.warning(
                "rejected an unreadable upload",
                # NOT `filename` — LogRecord already defines that, and a clashing
                # `extra` key raises KeyError inside logging itself.
                extra={"upload_filename": filename, "file_type": file_type},
                exc_info=True,
            )
            detail = (
                f"could not read this file as {file_type.upper()} — it may be corrupt, "
                "or saved in a different format than its extension suggests"
            )
        raise HTTPException(status_code=422, detail=detail) from error


@app.post("/models/upload", response_model=ModelSummaryOut, status_code=201)
def upload_model(
    admin: AdminUser,
    file: Annotated[UploadFile, File(description="STL, OBJ, or GLB mesh")],
) -> ModelSummaryOut:
    """Accept a mesh from an admin and enqueue it into the pipeline (FR-9).

    The upload takes the place of the download stage: the file *is* the raw mesh,
    so it lands at `raw/<uid>.<ext>` and goes straight to the convert topic. From
    there it is an ordinary model — the remaining stages and the labeling UI can't
    tell it apart from an ingested one.

    The uid is generated rather than derived from the file, so re-uploading the
    same mesh creates a second model. Content-addressing would deduplicate, but
    it would also make two admins uploading the same file collide on one row.
    """
    settings = get_settings()
    _rate_limit(upload_limiter, f"upload:user:{admin.id}", UPLOAD_PER_ADMIN)

    suffix = _validated_upload_suffix(file.filename)
    data = _read_within_limit(file, settings.upload_max_bytes)
    _reject_unloadable_mesh(data, RAW_SUFFIX_TO_FILE_TYPE[suffix], file.filename)

    uid = uuid4().hex  # same 32-hex shape as an Objaverse uid
    key = raw_key(uid, suffix)
    build_storage(settings).put_bytes(key, data)

    with session_scope() as session:
        session.add(
            Model(
                uid=uid,
                download_status=DownloadStatus.downloaded,
                raw_key=key,
                content_hash=hashlib.sha256(data).hexdigest(),
                # The filename is the only human-readable thing an upload carries,
                # and the labeling UI shows the title — so keep it rather than
                # leaving the admin to identify the model by a random hex uid.
                title=Path(file.filename or uid).stem or uid,
                tags=[],
            )
        )

    publish_next(settings.convert_topic, uid)
    logger.info("uploaded", extra={"uid": uid, "admin": admin.email, "key": key})
    return _load_summary(uid)


@app.get(
    "/models/{uid}/artifacts", response_model=ModelArtifactsOut, dependencies=LOGIN_REQUIRED
)
def get_model_artifacts(uid: str) -> ModelArtifactsOut:
    """URLs for a model's rendered views and its normalized mesh.

    Prefers time-limited signed URLs so the browser reads object storage directly
    — a paginated grid is 12 views per card, and proxying all of that through the
    API would make it the bottleneck and pay egress twice. Falls back to streaming
    via `/artifacts/{key}` when the backend can't sign (local dev).

    Only artifacts that actually exist are returned: a model part-way through the
    pipeline yields fewer views, or none, and the UI shows a placeholder rather
    than broken images.
    """
    storage = build_storage(get_settings())

    def resolve(key: str) -> str | None:
        if not storage.exists(key):
            return None
        return storage.signed_url(key, ARTIFACT_URL_TTL) or f"/artifacts/{key}"

    with session_scope() as session:
        if session.get(Model, uid) is None:
            raise HTTPException(status_code=404, detail="unknown model")

    views = [url for url in (resolve(key) for key in view_keys(uid)) if url is not None]
    return ModelArtifactsOut(uid=uid, views=views, mesh=resolve(normalized_key(uid)))


@app.get("/artifacts/{key:path}", dependencies=LOGIN_REQUIRED)
def stream_artifact(key: str) -> Response:
    """Stream a blob through the API — the fallback when signing isn't available.

    Login is required, because this is the dataset (NFR-7). Note the asymmetry
    with signed URLs, which carry their own bearer-like grant and so are readable
    without a session until they expire; that is the trade for not proxying every
    image, and why the TTL is short.
    """
    if ".." in key:  # defence in depth — keys are ours, but this endpoint is public-facing
        raise HTTPException(status_code=400, detail="validation_error")
    storage = build_storage(get_settings())
    if not storage.exists(key):
        raise HTTPException(status_code=404, detail="unknown artifact")
    return Response(
        content=storage.get_bytes(key),
        media_type=_ARTIFACT_MEDIA_TYPES.get(Path(key).suffix, "application/octet-stream"),
        headers={"Cache-Control": "private, max-age=3600"},
    )


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
