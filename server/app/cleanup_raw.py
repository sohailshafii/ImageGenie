"""Delete raw source meshes for models excluded from the dataset.

CLAUDE.md's cost guardrail: "Delete raw files for models excluded from the
dataset." The raw bucket is the bulk of the storage line, and a mesh that will
never be trained on is paying rent for nothing.

**Deliberately narrow.** It deletes raw *only* for models that are not in the
dataset at all — soft-deleted, unlabeled, or labeled outside the 12-class roster
— plus stray blobs with no model row. Every model still in the dataset keeps its
source mesh, so any stage can be re-run from scratch. It never touches
``processed/``.

**Only raw is removable in the first place.** The processed artifacts are what
the labeling UI and training read; raw is read once, by convert. Deleting raw is
also compatible with the reconciler, which marks a model whose mesh is gone as
``pending`` rather than dropping it (server.md#rebuilding-the-tables-from-storage).

    python -m app.cleanup_raw                 # dry run: report, delete nothing
    python -m app.cleanup_raw --apply         # delete, after a typed confirmation

`make cleanup-raw` wraps the dry run; `make cleanup-raw APPLY=1` the real thing.

**This is an operator tool, not worker-image code.** It is the one module under
``app/`` that imports from ``ml/`` (the class roster, whose single source of
truth is ``ml/taxonomy.py``), which is why the Makefile target puts both on
PYTHONPATH — mirroring how `make train` puts ``server/`` on the path for
``ml/train.py``. Nothing in the worker image imports this module.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session
from taxonomy import ROSTER

from .artifact_keys import RAW_PREFIX, uid_from_key
from .models import Label, Model
from .storage import Storage

logger = logging.getLogger(__name__)

# Why a given mesh is not in the dataset. Reported per-reason so a dry run shows
# *what kind* of exclusion is reclaiming the space, not just a total — a surprise
# in the mix (say, thousands "unlabeled") is the signal to stop and look rather
# than pass --apply.
REASON_ORPHAN = "no model row"
REASON_DELETED = "soft-deleted"
REASON_UNLABELED = "no label"
REASON_OUT_OF_SCOPE = "label outside the roster"


@dataclass(frozen=True)
class Candidate:
    """One raw blob that would be deleted, and why."""

    key: str
    uid: str | None  # None only for a key that parses to no uid
    reason: str
    size_bytes: int


def _excluded_uids(session: Session) -> tuple[set[str], dict[str, str]]:
    """(soft-deleted uids, uid → current class name) for every model row.

    The current label is the most recent one, manual beating weak — the same
    resolution the API uses to answer "what is this model labeled?", so the
    cleanup and the UI can never disagree about what is in the dataset.
    """
    deleted_uids_set = set(
        session.scalars(select(Model.uid).where(Model.deleted_at.is_not(None))).all()
    )
    current_labels = (
        select(Label.model_uid, Label.class_name)
        .distinct(Label.model_uid)
        .order_by(Label.model_uid, Label.created_at.desc(), Label.id.desc())
    )
    uid_to_class_name = {
        model_uid: class_name for model_uid, class_name in session.execute(current_labels)
    }
    return deleted_uids_set, uid_to_class_name


def _classify(
    key: str,
    size_bytes: int,
    known_uids_set: set[str],
    deleted_uids_set: set[str],
    uid_to_class_name: dict[str, str],
    in_scope_set: frozenset[str],
) -> Candidate | None:
    """The exclusion reason for one raw key, or None to keep it."""
    uid = uid_from_key(key)
    if uid is None:
        # An unrecognised key under raw/. Reported, never deleted: the reconciler
        # treats stray objects the same way, and a key we cannot parse is one we
        # cannot claim to own.
        return None
    if uid not in known_uids_set:
        return Candidate(key, uid, REASON_ORPHAN, size_bytes)
    if uid in deleted_uids_set:
        return Candidate(key, uid, REASON_DELETED, size_bytes)
    class_name = uid_to_class_name.get(uid)
    if class_name is None:
        return Candidate(key, uid, REASON_UNLABELED, size_bytes)
    if class_name not in in_scope_set:
        return Candidate(key, uid, REASON_OUT_OF_SCOPE, size_bytes)
    return None


def select_candidates(
    session: Session,
    storage: Storage,
    in_scope: Iterable[str] = ROSTER,
) -> Iterator[Candidate]:
    """Stream the raw blobs that are excluded from the dataset.

    One listing pass over ``raw/`` joined against two in-memory maps of the model
    table. The maps are loaded up front rather than queried per blob: ~12k models
    is a few MB, where a query per object would be ~165k round-trips.

    `in_scope` is a parameter rather than a constant so a test can pin behaviour
    without depending on the live roster.
    """
    in_scope_set = frozenset(in_scope)
    deleted_uids_set, uid_to_class_name = _excluded_uids(session)
    known_uids_set = set(session.scalars(select(Model.uid)).all())

    for key, size_bytes in storage.list_sizes(RAW_PREFIX):
        candidate = _classify(
            key,
            size_bytes,
            known_uids_set,
            deleted_uids_set,
            uid_to_class_name,
            in_scope_set,
        )
        if candidate is not None:
            yield candidate


def summarize(candidates: Iterable[Candidate]) -> tuple[Counter, Counter]:
    """(count per reason, bytes per reason) — what the dry run prints."""
    counts: Counter = Counter()
    total_bytes: Counter = Counter()
    for candidate in candidates:
        counts[candidate.reason] += 1
        total_bytes[candidate.reason] += candidate.size_bytes
    return counts, total_bytes
