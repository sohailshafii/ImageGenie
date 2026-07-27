"""The 12-class roster, as the API knows it.

**This duplicates `ml/taxonomy.py`'s `ROSTER` on purpose.** The API image ships
`server/app` and the built SPA and nothing else (server/Dockerfile), so importing
the ml package here would crash the service at startup in production while
working perfectly in every local test — the worst possible failure shape. The
frontend keeps its own copy for the same reason (web/src/api/types.ts).

`tests/test_roster.py` asserts this list still matches `ml/taxonomy.py`, so the
duplication is checked rather than merely hoped for; that test runs from the repo
root, where both packages are importable.
"""

from __future__ import annotations

# Sorted, matching ml/taxonomy.py's `ROSTER = tuple(sorted(...))`. The *order*
# is not load-bearing here (the API only tests membership), but keeping it
# identical makes the drift test a plain equality check.
CLASS_ROSTER: tuple[str, ...] = (
    "aircraft",
    "animal",
    "building",
    "car",
    "chair",
    "electronics",
    "figure",
    "food",
    "lamp",
    "plant",
    "table",
    "weapon",
)
