"""add evaluation status and error so a scoring job in flight is visible

An evaluation used to exist only once it had finished, because the row *was* the
report. That is fine for `make evaluate` in a terminal, where the log stream is
the progress bar, and wrong for a button: a job that takes minutes shows nothing
at all until it lands, and a job that dies shows nothing ever. So the row is now
written when scoring starts and finalized when it ends, which is the bargain
`training_run` already makes.

Consequences, in order:

- `status` reuses the existing **trainingstatus** enum rather than declaring a
  parallel one. The values are the same three, the UI renders both with one
  badge, and a second type would be a thing to keep in step for no gain.
  `create_type=False` because the type already exists — creating it again fails
  the migration.
- Existing rows are all finished, by definition: nothing could write a row before
  this. They are backfilled `completed` explicitly rather than by a server
  default, so the column carries no default that a future direct INSERT could
  silently inherit — new rows get `running` from the ORM, on purpose.
- `report` becomes nullable. A running evaluation has no report and a failed one
  never will; a `{}` placeholder would have to be told apart from a real empty
  report by every reader.

Revision ID: 4cbd7fc5f228
Revises: a2f9bcd2c0c0
Create Date: 2026-08-09 08:37:57.004524

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '4cbd7fc5f228'
down_revision: Union[str, None] = 'a2f9bcd2c0c0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TRAINING_STATUS = postgresql.ENUM(
    "running", "completed", "failed", name="trainingstatus", create_type=False
)


def upgrade() -> None:
    # Three steps, because the column deliberately carries no default (see above).
    # Postgres will only add a NOT NULL column to a table that already has rows if
    # it has something to put in them, so the backfill goes between adding the
    # column and constraining it.
    op.add_column("evaluation", sa.Column("status", TRAINING_STATUS, nullable=True))
    op.execute("UPDATE evaluation SET status = 'completed'")
    op.alter_column("evaluation", "status", nullable=False)
    op.add_column("evaluation", sa.Column("error", sa.String(), nullable=True))
    op.alter_column("evaluation", "report", existing_type=postgresql.JSONB, nullable=True)


def downgrade() -> None:
    # A row with no report cannot survive the NOT NULL going back on, and there is
    # no honest value to invent for it — an unfinished evaluation measured
    # nothing. Dropping those rows is the truthful reversal.
    op.execute("DELETE FROM evaluation WHERE report IS NULL")
    op.alter_column("evaluation", "report", existing_type=postgresql.JSONB, nullable=False)
    op.drop_column("evaluation", "error")
    op.drop_column("evaluation", "status")
    # The enum type itself stays: training_run still uses it.
