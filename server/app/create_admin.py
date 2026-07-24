"""Create or promote a verified admin account — the deploy bootstrap.

Signup is invite-gated and invites can only be minted by an admin, so a freshly
provisioned database has no way in. This mints the first admin directly. Run it
once after the schema is in place (server.md#deploying-the-api-to-cloud-run); from
then on that admin invites everyone else through the UI.

    python -m app.create_admin --email you@example.com            # prompts for a password
    IMAGEGENIE_ADMIN_PASSWORD=... python -m app.create_admin --email you@example.com

Idempotent: an existing account with that email is promoted to a verified admin
and (if a password is given) its password reset, rather than duplicated.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from sqlalchemy import select

from .db import session_scope
from .models import User, UserRole
from .security import hash_password


def create_admin(email: str, password: str) -> str:
    """Upsert a verified admin with `email`/`password`. Returns 'created'|'updated'."""
    email = email.strip().lower()
    with session_scope() as session:
        user = session.scalar(select(User).where(User.email == email))
        if user is None:
            session.add(
                User(
                    email=email,
                    role=UserRole.admin,
                    password_hash=hash_password(password),
                    verified=True,
                )
            )
            return "created"
        # Promote + reset in place, so re-running to reset a forgotten password is safe.
        user.role = UserRole.admin
        user.verified = True
        user.password_hash = hash_password(password)
        return "updated"


def _resolve_password() -> str:
    """The admin password, from the env or an interactive prompt (never a CLI arg).

    Kept off the command line so it doesn't land in shell history or the process
    list; env for non-interactive use (the deploy script), prompt otherwise.
    """
    password = os.environ.get("IMAGEGENIE_ADMIN_PASSWORD")
    if password:
        return password
    if not sys.stdin.isatty():
        raise SystemExit(
            "no password: set IMAGEGENIE_ADMIN_PASSWORD or run interactively"
        )
    first = getpass.getpass("Admin password: ")
    if first != getpass.getpass("Confirm password: "):
        raise SystemExit("passwords did not match")
    if len(first) < 8:
        raise SystemExit("password must be at least 8 characters")
    return first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="admin account email")
    args = parser.parse_args()

    outcome = create_admin(args.email, _resolve_password())
    print(f"{outcome} admin {args.email.strip().lower()}")


if __name__ == "__main__":
    main()
