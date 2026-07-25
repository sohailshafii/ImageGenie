"""add training_run and training_metric for the training milestone (FR-6/NFR-4)

Revision ID: d5a1c8b3e6f2
Revises: c4e9f1a2b3d0
Create Date: 2026-07-24 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'd5a1c8b3e6f2'
down_revision: Union[str, None] = 'c4e9f1a2b3d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "training_run",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "status",
            sa.Enum("running", "completed", "failed", name="trainingstatus"),
            nullable=False,
        ),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("data_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("weights_uri", sa.String(), nullable=True),
        sa.Column("notes", sa.String(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "training_metric",
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("loss", sa.Float(), nullable=False),
        sa.Column("val_loss", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["training_run.id"]),
        sa.PrimaryKeyConstraint("run_id", "step"),
    )


def downgrade() -> None:
    op.drop_table("training_metric")
    op.drop_table("training_run")
    # training_run.status created this enum type; create_table won't drop it.
    sa.Enum(name="trainingstatus").drop(op.get_bind())
