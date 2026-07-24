"""Tests for the admin bootstrap (`app.create_admin`).

The one way into a freshly provisioned database, so it has to be right: it creates
a verified admin, and re-running promotes/resets in place rather than duplicating.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Engine, func, select, text

from app import db
from app.create_admin import create_admin
from app.models import User, UserRole
from app.security import hash_password, verify_password


@pytest.fixture
def clean_db(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Engine:
    monkeypatch.setattr(db, "get_engine", lambda: pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(text("TRUNCATE app_user RESTART IDENTITY CASCADE"))
    return pg_engine


def test_creates_a_verified_admin(clean_db) -> None:
    assert create_admin("Owner@Example.com", "s3cret-pw") == "created"

    with db.session_scope() as session:
        user = session.scalar(select(User).where(User.email == "owner@example.com"))
        assert user.role == UserRole.admin
        assert user.verified is True
        assert verify_password("s3cret-pw", user.password_hash)
        # Email is normalized, so a later login with any casing resolves the row.
        assert user.email == "owner@example.com"


def test_rerun_promotes_and_resets_in_place(clean_db) -> None:
    """Re-running must not duplicate — it promotes a normal user and resets the pw."""
    with db.session_scope() as session:
        session.add(
            User(
                email="owner@example.com",
                role=UserRole.user,
                password_hash=hash_password("old-pw"),
                verified=False,
            )
        )

    assert create_admin("owner@example.com", "new-pw") == "updated"

    with db.session_scope() as session:
        assert session.scalar(select(func.count(User.id))) == 1
        user = session.scalar(select(User).where(User.email == "owner@example.com"))
        assert user.role == UserRole.admin
        assert user.verified is True
        assert verify_password("new-pw", user.password_hash)
