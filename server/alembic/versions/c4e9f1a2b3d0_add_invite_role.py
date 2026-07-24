"""add invite.role so admins choose viewer vs admin at invite time

Revision ID: c4e9f1a2b3d0
Revises: fb0e0e1af523
Create Date: 2026-07-24 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c4e9f1a2b3d0'
down_revision: Union[str, None] = 'fb0e0e1af523'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Reuse the existing userrole enum (created with app_user) rather than
    # recreating the type — create_type=False, else this fails "type already exists".
    userrole = postgresql.ENUM("user", "admin", name="userrole", create_type=False)
    # server_default backfills any existing invite rows to 'user'; drop it after so
    # the column matches app_user.role, whose default is app-level (the ORM), not DB.
    op.add_column(
        "invite",
        sa.Column("role", userrole, nullable=False, server_default="user"),
    )
    op.alter_column("invite", "role", server_default=None)


def downgrade() -> None:
    op.drop_column("invite", "role")
