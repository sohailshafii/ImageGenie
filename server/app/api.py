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
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .artifact_keys import (
    MESH_SUFFIX,
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
    Artifact,
    ArtifactStage,
    ArtifactStatus,
    DownloadStatus,
    EmailVerification,
    Invite,
    Label,
    LabelSource,
    Model,
    TrainingMetric,
    TrainingRun,
    User,
    UserRole,
)
from .queue import publish_next
from .ratelimit import BackoffRule, FixedWindowRateLimiter, LoginBackoff, RateLimitRule
from .roster import CLASS_ROSTER
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
from .storage import Storage, build_storage
from .training_jobs import (
    TrainingLaunchError,
    launch_configured,
    submit_training_job,
)

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
    """A manual label. `class_name` is validated against the roster here because
    this is the boundary: the SPA only ever offers the 12, but an API client can
    send anything, and an out-of-roster row does not fail here — it fails inside a
    DataLoader worker *after* a training job has queued for a spot GPU and pulled
    a multi-GB image (ml/dataset.py maps class name to logit index). Rejecting it
    at write time turns a slow, expensive failure into a 422.
    """

    class_name: str

    @field_validator("class_name")
    @classmethod
    def _must_be_in_roster(cls, value: str) -> str:
        if value not in CLASS_ROSTER:
            raise ValueError(
                f"unknown class {value!r}; expected one of {', '.join(CLASS_ROSTER)}"
            )
        return value


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


class TrainingRunSummaryOut(BaseModel):
    """A training run as it appears in the dashboard list (FR-6)."""

    id: int
    status: str
    arch: str | None  # config summary for the row; None if config lacks it
    label_count: int | None  # how many labels the run trained on (from data_snapshot)
    final_loss: float | None  # training loss at the last logged step; None if no metrics
    # Top-1 val accuracy at the last step that measured it. Reported next to the
    # loss because on a ~7.7:1 skewed corpus a falling loss can hide a model that
    # has collapsed onto the majority class.
    final_accuracy: float | None
    started_at: datetime
    finished_at: datetime | None


class TrainingRunDetailOut(BaseModel):
    """One run's full bookkeeping for the detail page — the three NFR-4 blobs
    verbatim plus status/timestamps. The loss curve is fetched separately."""

    id: int
    status: str
    config: dict
    data_snapshot: dict
    metrics: dict | None  # dev-set eval; null until the run is evaluated (B4/M7)
    weights_uri: str | None
    notes: str | None
    started_at: datetime
    finished_at: datetime | None


class TrainingLaunchConfigOut(BaseModel):
    """What the launch form needs to describe the run before it is started."""

    configured: bool  # false on a deployment with no Vertex (local dev)
    image: str | None  # the exact commit-tagged image that will run
    region: str | None
    trainable_count: int  # models that are labeled AND rendered — the full-set size


class TrainingLaunchIn(BaseModel):
    epochs: int = 5
    limit: int | None = None  # None = the whole trainable set
    notes: str | None = None


class TrainingLaunchOut(BaseModel):
    job_name: str  # Vertex resource name
    image: str
    args: list[str]  # exactly what the container was given, for the record


class TrainingMetricOut(BaseModel):
    """One point on a run's loss curve (B2/B3)."""

    step: int
    loss: float
    val_loss: float | None  # null on steps where validation was not evaluated
    val_accuracy: float | None  # top-1 on the val split, 0..1; null when not evaluated


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

    Matches the exemptions against the **mount-relative** path. This middleware
    runs at the outermost layer, before the router strips the mount prefix, so
    mounted under `/api` the exempt `/auth/login` arrives as `/api/auth/login`
    with `root_path == "/api"`. Stripping root_path keeps the exemptions correct
    whether the app runs mounted or standalone (server.md#serving-the-spa).
    """
    path = request.scope["path"]
    root_path = request.scope.get("root_path", "")
    if root_path and path.startswith(root_path):
        path = path[len(root_path) :] or "/"
    if (
        request.method not in CSRF_SAFE_METHODS
        and path not in CSRF_EXEMPT_PATHS
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
    # Viewer (``user``) or ``admin``; defaults to viewer. Typing it as UserRole
    # makes anything else a 422 rather than silently coercing.
    role: UserRole = UserRole.user


class InviteOut(BaseModel):
    email: str
    expires_at: datetime
    accepted: bool
    role: UserRole


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
            # The role the admin chose at invite time — an admin invite grants admin
            # (only admins can invite, so this stays a trusted-caller decision).
            role=invite.role,
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
        invite.role = body.role  # re-inviting can also change the role
    background.add_task(send_invite_email, email, admin.email)
    return InviteOut(email=email, expires_at=expires_at, accepted=False, role=body.role)


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


def _url_prefix(request: Request) -> str:
    """The mount prefix for building same-origin artifact URLs.

    In production the API is mounted at `/api` (so the SPA can own the root), and
    Starlette exposes that as `root_path`; unmounted (local backend dev, the test
    suite) it is empty. A streaming-fallback artifact URL must carry this prefix
    or the browser would request it at the root and hit the SPA shell instead of
    the blob (server.md#serving-the-spa).
    """
    return request.scope.get("root_path", "")


def _thumbnail_url(storage, uid: str, url_prefix: str = "") -> str:
    """URL for a model's first rendered view. No existence check — see the field."""
    key = view_key(uid, 0)
    return storage.signed_url(key, ARTIFACT_URL_TTL) or f"{url_prefix}/artifacts/{key}"


def _summary(
    storage, uid, title, tags, class_name, source, confidence, url_prefix: str = ""
) -> ModelSummaryOut:
    return ModelSummaryOut(
        uid=uid,
        # Falls back to the uid until `app.backfill_metadata` has run — a card
        # with no caption at all would be worse than a dull one.
        title=title or f"model {uid[:8]}",
        tags=tags or [],
        class_name=class_name,
        source=source.value if source is not None else None,
        confidence=confidence,
        thumbnail=_thumbnail_url(storage, uid, url_prefix),
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


def _latest_metric_loss():
    """Subquery of each run's most-recent training loss — the headline for the
    dashboard list. DISTINCT ON keeps one row per run, its highest step."""
    return (
        select(TrainingMetric.run_id, TrainingMetric.loss)
        .distinct(TrainingMetric.run_id)
        .order_by(TrainingMetric.run_id, TrainingMetric.step.desc())
        .subquery()
    )


def _latest_val_accuracy():
    """Subquery of each run's most-recent *measured* validation accuracy.

    Filtered to non-null rather than simply taking the highest step, because
    validation runs once per epoch: a run caught mid-epoch has interval rows on
    top with no accuracy, and reading the plain maximum would report "no
    accuracy" for a run that has perfectly good numbers from earlier epochs.
    """
    return (
        select(TrainingMetric.run_id, TrainingMetric.val_accuracy)
        .where(TrainingMetric.val_accuracy.is_not(None))
        .distinct(TrainingMetric.run_id)
        .order_by(TrainingMetric.run_id, TrainingMetric.step.desc())
        .subquery()
    )


@app.get(
    "/training-runs",
    response_model=list[TrainingRunSummaryOut],
    dependencies=LOGIN_REQUIRED,
)
def list_training_runs() -> list[TrainingRunSummaryOut]:
    """All training runs, newest first — the dashboard list (FR-6).

    Read-only and login-gated (viewing is open to any authenticated user, FR-8):
    the training script writes these rows directly, so there are no write routes.
    """
    latest_loss = _latest_metric_loss()
    latest_accuracy = _latest_val_accuracy()
    with session_scope() as session:
        rows = session.execute(
            select(TrainingRun, latest_loss.c.loss, latest_accuracy.c.val_accuracy)
            .outerjoin(latest_loss, latest_loss.c.run_id == TrainingRun.id)
            .outerjoin(latest_accuracy, latest_accuracy.c.run_id == TrainingRun.id)
            .order_by(TrainingRun.id.desc())
        ).all()
        return [
            TrainingRunSummaryOut(
                id=run.id,
                status=run.status.value,
                arch=run.config.get("arch"),
                label_count=run.data_snapshot.get("label_count"),
                final_loss=final_loss,
                final_accuracy=final_accuracy,
                started_at=run.started_at,
                finished_at=run.finished_at,
            )
            for run, final_loss, final_accuracy in rows
        ]


@app.get(
    "/training-runs/{run_id}",
    response_model=TrainingRunDetailOut,
    dependencies=LOGIN_REQUIRED,
)
def get_training_run(run_id: int) -> TrainingRunDetailOut:
    """One run's full config/snapshot/metrics bookkeeping (404 if unknown)."""
    with session_scope() as session:
        run = session.get(TrainingRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="unknown training run")
        return TrainingRunDetailOut(
            id=run.id,
            status=run.status.value,
            config=run.config,
            data_snapshot=run.data_snapshot,
            metrics=run.metrics,
            weights_uri=run.weights_uri,
            notes=run.notes,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )


@app.get(
    "/training-runs/{run_id}/metrics",
    response_model=list[TrainingMetricOut],
    dependencies=LOGIN_REQUIRED,
)
def get_training_run_metrics(run_id: int) -> list[TrainingMetricOut]:
    """A run's loss curve, in step order — the data behind the cost graph (B3).

    Separate from the detail payload so the (potentially long) curve is fetched
    on its own, mirroring the model/artifacts split.
    """
    with session_scope() as session:
        if session.get(TrainingRun, run_id) is None:
            raise HTTPException(status_code=404, detail="unknown training run")
        points = session.execute(
            select(TrainingMetric)
            .where(TrainingMetric.run_id == run_id)
            .order_by(TrainingMetric.step)
        ).scalars()
        return [
            TrainingMetricOut(
                step=point.step,
                loss=point.loss,
                val_loss=point.val_loss,
                val_accuracy=point.val_accuracy,
            )
            for point in points
        ]


def _trainable_count(session: Session) -> int:
    """How many models are both labeled and rendered — what a full run would use.

    Mirrors `ml/train.py`'s `load_trainable_samples` filter (live, current label,
    a done render). Counted here rather than imported because the API image does
    not contain `ml/`; the two must agree, so a change to one belongs in both.
    """
    return (
        session.scalar(
            select(func.count(func.distinct(Label.model_uid)))
            .select_from(Label)
            .join(Model, Model.uid == Label.model_uid)
            .join(Artifact, Artifact.model_uid == Label.model_uid)
            .where(Model.deleted_at.is_(None))
            .where(Artifact.stage == ArtifactStage.rendered)
            .where(Artifact.status == ArtifactStatus.done)
        )
        or 0
    )


@app.get(
    "/training-launch",
    response_model=TrainingLaunchConfigOut,
    dependencies=[Depends(require_admin)],
)
def get_training_launch_config() -> TrainingLaunchConfigOut:
    """What the launch form shows before anything is spent.

    Admin-only like the launch itself: it reports the image tag and the size of
    the trainable set, which is what lets the form put a cost in front of the
    button rather than after it.
    """
    settings = get_settings()
    with session_scope() as session:
        trainable = _trainable_count(session)
    return TrainingLaunchConfigOut(
        configured=launch_configured(settings),
        image=settings.train_image,
        region=settings.vertex_region if launch_configured(settings) else None,
        trainable_count=trainable,
    )


@app.post(
    "/training-runs",
    response_model=TrainingLaunchOut,
    status_code=202,
    dependencies=[Depends(require_admin)],
)
def launch_training_run(body: TrainingLaunchIn) -> TrainingLaunchOut:
    """Start a Vertex AI spot-GPU training run (admin-only).

    **202, not 201**: this accepts the request and hands it to Vertex. No
    `training_run` row exists yet — `ml/train.py` writes that itself once the
    container starts, which is minutes later after the GPU is provisioned and a
    multi-GB image is pulled. The dashboard shows the run when it appears.

    The API deliberately does not cap the request. The form recommends a small
    default and shows the cost, but an admin who means to launch the full set is
    allowed to; the guardrail here is informed consent, not enforcement.
    """
    if body.epochs < 1:
        raise HTTPException(status_code=422, detail="epochs must be at least 1")
    if body.limit is not None and body.limit < 1:
        raise HTTPException(status_code=422, detail="limit must be at least 1")

    settings = get_settings()
    if not launch_configured(settings):
        # 503, not 500: nothing is broken, this deployment simply has no Vertex
        # to submit to (local dev). The form disables the button on this.
        raise HTTPException(
            status_code=503, detail="training launches are not configured here"
        )

    args = ["--device", "cuda", "--num-workers", "4", "--epochs", str(body.epochs)]
    if body.limit is not None:
        args += ["--limit", str(body.limit)]
    if body.notes:
        args += ["--notes", body.notes]

    try:
        job_name = submit_training_job(settings, args, display_name="imagegenie-train")
    except TrainingLaunchError as error:
        # 502: we reached out and were refused. The message is Vertex's own —
        # quota, IAM, a bad image — which is what the admin needs to see.
        raise HTTPException(status_code=502, detail=str(error)) from error

    logger.info("training job launched", extra={"job": job_name, "args": args})
    return TrainingLaunchOut(job_name=job_name, image=settings.train_image or "", args=args)


def _paginate_summaries(
    query, order, page: int, page_size: int, url_prefix: str = ""
) -> ModelPageOut:
    """Run a summary `query` with `order` and wrap one page as a ModelPageOut."""
    storage = build_storage(get_settings())  # built once, reused for every row
    with session_scope() as session:
        total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
        rows = session.execute(
            query.order_by(*order).limit(page_size).offset((page - 1) * page_size)
        ).all()
        items = [_summary(storage, *row, url_prefix=url_prefix) for row in rows]
    return ModelPageOut(items=items, total=total, page=page, page_size=page_size)


@app.get("/models", response_model=ModelPageOut, dependencies=LOGIN_REQUIRED)
def list_models(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=PAGE_SIZE_MAX),
    class_name: str | None = None,
    source: LabelSource | None = None,
    search: str | None = None,
    sort: ModelSort = ModelSort.uid,
) -> ModelPageOut:
    latest = _latest_labels()
    query = (
        select(*_SUMMARY_COLUMNS, latest.c.class_name, latest.c.source, latest.c.confidence)
        .outerjoin(latest, Model.uid == latest.c.model_uid)
        .where(Model.deleted_at.is_(None))  # soft-deleted models are hidden here
    )
    if class_name is not None:
        query = query.where(latest.c.class_name == class_name)
    if source is not None:
        query = query.where(latest.c.source == source)
    if search is not None and search.strip():
        # Case-insensitive substring match on the title. Escape the LIKE
        # metacharacters in user input first, so a search for "50%" or "a_b"
        # matches those literally instead of acting as wildcards.
        term = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        query = query.where(Model.title.ilike(f"%{term}%", escape="\\"))

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

    return _paginate_summaries(query, order, page, page_size, _url_prefix(request))


# Registered before `GET /models/{uid}` so the literal path wins — otherwise
# "deleted" would bind as a uid and this view would be unreachable.
@app.get("/models/deleted", response_model=ModelPageOut)
def list_deleted_models(
    request: Request,
    admin: AdminUser,
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=PAGE_SIZE_MAX),
) -> ModelPageOut:
    """Soft-deleted models, most-recently-deleted first — the restore queue.

    Admin-only: which models an admin removed is operational detail, not
    something a labeler should page through.
    """
    latest = _latest_labels()
    query = (
        select(*_SUMMARY_COLUMNS, latest.c.class_name, latest.c.source, latest.c.confidence)
        .outerjoin(latest, Model.uid == latest.c.model_uid)
        .where(Model.deleted_at.is_not(None))
    )
    order = (Model.deleted_at.desc(), Model.uid)
    return _paginate_summaries(query, order, page, page_size, _url_prefix(request))


def _require_live_model(session: Session, uid: str) -> Model:
    """Return the model, or 404 if it doesn't exist or has been soft-deleted.

    The existence check every write/artifact route needs. Folding the deleted
    check in here means a route can't accidentally act on a deleted model by
    checking only for existence.
    """
    model = session.get(Model, uid)
    if model is None or model.deleted_at is not None:
        raise HTTPException(status_code=404, detail="unknown model")
    return model


def _load_summary(uid: str, url_prefix: str = "") -> ModelSummaryOut:
    """Read one live model's current label, or 404. Shared by the GET/PUT routes.

    A soft-deleted model reads as 404 here — the same as a nonexistent one, since
    from a labeler's point of view it is gone. The Deleted view uses a separate
    query that opts *into* deleted rows.
    """
    latest = _latest_labels()
    with session_scope() as session:
        row = session.execute(
            select(*_SUMMARY_COLUMNS, latest.c.class_name, latest.c.source, latest.c.confidence)
            .outerjoin(latest, Model.uid == latest.c.model_uid)
            .where(Model.uid == uid, Model.deleted_at.is_(None))
        ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown model")
    return _summary(build_storage(get_settings()), *row, url_prefix=url_prefix)


@app.get("/models/{uid}", response_model=ModelSummaryOut, dependencies=LOGIN_REQUIRED)
def get_model(request: Request, uid: str) -> ModelSummaryOut:
    return _load_summary(uid, _url_prefix(request))


@app.delete("/models/{uid}", status_code=204)
def delete_model(uid: str, admin: AdminUser) -> Response:
    """Soft-delete a model (admin-only, FR-9): hide it, keep its data.

    Idempotent — deleting an already-deleted model is a no-op, so a double click
    or a retried request doesn't error. The blobs are left in place and the row
    keeps all its labels, so `POST /models/{uid}/restore` fully reverses this
    (server.md#soft-delete).
    """
    with session_scope() as session:
        model = session.get(Model, uid)
        if model is None:
            raise HTTPException(status_code=404, detail="unknown model")
        if model.deleted_at is None:
            model.deleted_at = datetime.now(UTC)
    logger.info("soft-deleted", extra={"uid": uid, "admin": admin.email})
    return Response(status_code=204)


@app.post("/models/{uid}/restore", response_model=ModelSummaryOut)
def restore_model(request: Request, uid: str, admin: AdminUser) -> ModelSummaryOut:
    """Undo a soft delete (admin-only), returning the now-visible model.

    404 if the uid doesn't exist; restoring a model that isn't deleted just
    returns it, so this is idempotent too.
    """
    with session_scope() as session:
        model = session.get(Model, uid)
        if model is None:
            raise HTTPException(status_code=404, detail="unknown model")
        model.deleted_at = None
    logger.info("restored", extra={"uid": uid, "admin": admin.email})
    return _load_summary(uid, _url_prefix(request))


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
    request: Request,
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
    return _load_summary(uid, _url_prefix(request))


@app.get(
    "/models/{uid}/artifacts", response_model=ModelArtifactsOut, dependencies=LOGIN_REQUIRED
)
def get_model_artifacts(request: Request, uid: str) -> ModelArtifactsOut:
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
    url_prefix = _url_prefix(request)

    def resolve(key: str) -> str | None:
        if not storage.exists(key):
            return None
        return storage.signed_url(key, ARTIFACT_URL_TTL) or f"{url_prefix}/artifacts/{key}"

    with session_scope() as session:
        _require_live_model(session, uid)

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


# ── Downloads (web.md#downloads) ────────────────────────────────────────────
# These two routes send the bytes **through the API**, where the viewer's
# artifact URLs are signed so the browser reads GCS directly instead. The
# trade-off inverts here for two reasons.
#
# **Where the filename comes from.** A saved file gets its name from one of two
# places: the `download` attribute on the link the user clicked, or the
# response's own `Content-Disposition` header. The `download` attribute is
# ignored when the link points at another origin — and a signed
# `storage.googleapis.com` URL is another origin — so with signed URLs the name
# can only come from the header, which means the *storage backend* has to send
# it. GCS can (`generate_signed_url(response_disposition=...)` bakes the header
# into the signature), but that would mean widening the `Storage` protocol for a
# GCS-only feature, and local dev can't sign at all — so one button would need
# two divergent code paths. Serving the bytes ourselves sets the header directly
# and behaves identically in both environments.
#
# **How often it happens.** What rules out proxying for artifacts is volume: 12
# view images on every card of a paginated grid. A download is one deliberate
# click on one model, so that argument doesn't reach these routes.


def _attachment(storage: Storage, key: str, filename: str) -> Response:
    """Serve a blob as a browser download, or 404 if it was never produced.

    404 rather than 500 for a missing blob: a model part-way through the
    pipeline legitimately has no normalized mesh yet, which is an absent file
    and not an error.
    """
    if not storage.exists(key):
        raise HTTPException(status_code=404, detail="artifact not available")
    return Response(
        content=storage.get_bytes(key),
        media_type="application/octet-stream",
        headers={
            # Quoted, and the name is sanitised at the call site — a filename is
            # interpolated into a header, so a stray quote or newline would be a
            # header-injection lever.
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "private, no-store",
        },
    )


def _safe_filename(stem: str, suffix: str) -> str:
    """A download filename with everything but `[A-Za-z0-9._-]` replaced.

    Uids are ours (Objaverse hashes, or a uuid4 for an upload), so this never
    fires in practice; it is here so that stays true if the source of a uid ever
    changes.
    """
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)
    return f"{cleaned}{suffix}"


@app.get("/models/{uid}/download/source", dependencies=LOGIN_REQUIRED)
def download_source_mesh(uid: str) -> Response:
    """The original ingested mesh, straight from the raw bucket.

    `raw_key` carries the format, which is not always GLB — an admin upload may
    be STL or OBJ (`artifact_keys.RAW_SUFFIX_TO_FILE_TYPE`), so the suffix is
    read off the stored key rather than assumed.
    """
    with session_scope() as session:
        key = _require_live_model(session, uid).raw_key
    if key is None:
        raise HTTPException(status_code=404, detail="artifact not available")
    storage = build_storage(get_settings())
    return _attachment(storage, key, _safe_filename(uid, Path(key).suffix))


@app.get("/models/{uid}/download/normalized", dependencies=LOGIN_REQUIRED)
def download_normalized_mesh(uid: str) -> Response:
    """The centered, unit-scaled PLY — what the viewer shows and training consumes."""
    with session_scope() as session:
        _require_live_model(session, uid)
    storage = build_storage(get_settings())
    return _attachment(
        storage, normalized_key(uid), _safe_filename(f"{uid}-normalized", MESH_SUFFIX)
    )


@app.get("/training-runs/{run_id}/weights", dependencies=[Depends(require_admin)])
def download_training_weights(run_id: int) -> Response:
    """A finished run's saved weights — the `.pt` checkpoint written by `ml/train.py`.

    **Admin-only**, unlike the mesh downloads and the rest of the training
    dashboard, which any authenticated user may read. A trained model is the one
    artifact NFR-6 names as non-redistributable, so pulling the file down is kept
    to the role that could already launch the run that produced it. Reading the
    dashboard's numbers stays open — it is the weights themselves that are
    restricted, not the fact of the run.

    `weights_uri` holds a storage **key** (`artifact_keys.weights_key`), not a
    URL. It is null until a run checkpoints, so a still-running run — or one that
    failed before its first epoch finished — reads as "not available".
    """
    with session_scope() as session:
        run = session.get(TrainingRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="unknown training run")
        key = run.weights_uri
    if key is None:
        raise HTTPException(status_code=404, detail="artifact not available")
    storage = build_storage(get_settings())
    return _attachment(storage, key, _safe_filename(f"imagegenie-run-{run_id}", ".pt"))


@app.put("/models/{uid}/label", response_model=ModelSummaryOut)
def set_label(
    request: Request, uid: str, body: LabelIn, admin: AdminUser
) -> ModelSummaryOut:
    """Record a **manual** label (confirm keeps the class, correct changes it).

    Admin-only (FR-8); the correction is attributed to the calling admin.
    """
    write_key = f"label:user:{admin.id}"
    if not label_limiter.check(write_key, LABEL_WRITE_PER_USER):
        raise _too_many_requests(label_limiter.retry_after(write_key))
    with session_scope() as session:
        _require_live_model(session, uid)
        session.add(
            Label(
                model_uid=uid,
                class_name=body.class_name,
                source=LabelSource.manual,
                confidence=None,
                annotator=admin.email,
            )
        )
    return _load_summary(uid, _url_prefix(request))


# ── SPA serving (server.md#serving-the-spa) ─────────────────────────────────
# In production the built SPA ships inside this deployment so it and the API
# share one origin — the CSRF scheme rests on that. But the SPA's own client-side
# routes share the API's namespace (`/models/:uid` is both a page and an
# endpoint), so they can't both live at the root. The API therefore mounts under
# `/api` — which is exactly the prefix the frontend already sends and the Vite dev
# server already strips — and the SPA is served at the root.
#
# The public entrypoint is `app.api:root_app`. The API app above is untouched and
# still `app`, so it runs at the root under a direct `uvicorn app.api:app` (local
# backend-only dev) and the whole test suite exercises it that way.

# Hashed asset filenames can be cached forever; the shell must never be, or a
# deploy leaves browsers holding an index.html that points at asset hashes the
# new build no longer serves.
_SPA_ASSET_CACHE = "public, max-age=31536000, immutable"
_SPA_SHELL_CACHE = "no-cache"

# Same lifespan as `app`: mounting a sub-app does not run its lifespan, so the
# schema bootstrap (init_db) has to hang off the app uvicorn actually serves.
root_app = FastAPI(lifespan=lifespan, title="ImageGenie")
root_app.mount("/api", app)


@root_app.get("/{spa_path:path}", include_in_schema=False)
def serve_spa(spa_path: str) -> FileResponse:
    """Serve a built SPA file, or its index shell for client-routed paths.

    A catch-all rather than a StaticFiles mount because the SPA routes client-side:
    a deep link like `/deleted` or `/models/{uid}` is not a file on disk, so an
    unknown path must fall back to `index.html` for the browser router to take
    over — a plain static mount would 404 it. Registered after the `/api` mount,
    so it only ever sees non-API paths.

    404s when no `spa_dir` is configured, so a misconfigured deploy fails loudly
    rather than serving nothing.
    """
    spa_dir = get_settings().spa_dir
    if spa_dir is None:
        raise HTTPException(status_code=404, detail="not found")
    root = Path(spa_dir).resolve()
    index = root / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="not found")

    # Serve a real file when the path names one *inside* the SPA dir — the
    # `root in parents` check rejects `..` traversal, which resolves outside root.
    candidate = (root / spa_path).resolve()
    if spa_path and candidate.is_file() and root in candidate.parents:
        cache = _SPA_ASSET_CACHE if spa_path.startswith("assets/") else _SPA_SHELL_CACHE
        return FileResponse(candidate, headers={"Cache-Control": cache})
    return FileResponse(index, headers={"Cache-Control": _SPA_SHELL_CACHE})
