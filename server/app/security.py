"""Password hashing + server-side session helpers for the auth API.

bcrypt for passwords; opaque random session tokens stored in the ``session`` table
(``LoginSession``) so logout revokes immediately. The API sets the token as an
httpOnly cookie the browser JS can't read (server.md#api-layer, web.md#auth--roles).
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

import bcrypt
from sqlalchemy.orm import Session

from .models import LoginSession, User

SESSION_TTL = timedelta(days=14)
SESSION_COOKIE = "imagegenie_session"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_session(session: Session, user: User) -> str:
    """Mint a session token for `user` and persist it; returns the token."""
    token = secrets.token_urlsafe(32)
    session.add(
        LoginSession(
            token=token,
            user_id=user.id,
            expires_at=datetime.now(UTC) + SESSION_TTL,
        )
    )
    return token


def resolve_session(session: Session, token: str) -> User | None:
    """Return the session's user, or None if the token is unknown or expired."""
    login = session.get(LoginSession, token)
    if login is None or login.expires_at < datetime.now(UTC):
        return None
    return session.get(User, login.user_id)


def delete_session(session: Session, token: str) -> None:
    login = session.get(LoginSession, token)
    if login is not None:
        session.delete(login)
