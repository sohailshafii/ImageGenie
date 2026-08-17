"""add dev_set_member so a reserved model cannot be labeled

The second dev set is only worth scoring against while its objects stay
untrained-on, and a model becomes trainable the moment it has a label. Nothing
enforced that. The gold labels live in a CSV, so no automated path *put* a label
on these models, and the property was mistaken for a structural one — until two
were hand-labeled through the labeling UI on 2026-08-01, which cannot tell a
dev-set model from any other model in the catalog. They were found on 2026-08-16
by an `lvis` evaluation scoring 982 instead of 984, fifteen days later.

This table records the reservation where the server can act on it, so
`PUT /models/{uid}/label` can refuse rather than the contamination guard
reporting it after the fact.

**No foreign key to `model`.** Membership belongs to a uid and is decided when
the set is selected, which is before those objects are ingested and therefore
before any `model` row exists. Requiring the row first would leave open exactly
the window this closes — ingested and visible, but not yet protected.

Revision ID: b3d7e91c4a02
Revises: 4cbd7fc5f228
Create Date: 2026-08-16 17:02:11.882410

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3d7e91c4a02'
down_revision: Union[str, None] = '4cbd7fc5f228'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dev_set_member",
        sa.Column("model_uid", sa.String(), nullable=False),
        sa.Column("dev_set", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # Composite, so one uid can be reserved to more than one dev set and a
        # second push cannot duplicate a membership it already recorded.
        sa.PrimaryKeyConstraint("model_uid", "dev_set"),
    )
    # The hot query is "is this uid reserved at all", asked on every label write;
    # the primary key's leading column serves it, but the label path looks up by
    # uid alone often enough to name the index.
    op.create_index(
        "ix_dev_set_member_model_uid", "dev_set_member", ["model_uid"]
    )


def downgrade() -> None:
    op.drop_index("ix_dev_set_member_model_uid", table_name="dev_set_member")
    op.drop_table("dev_set_member")
