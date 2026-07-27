"""The API's roster copy must not drift from ml/taxonomy.py.

`app/roster.py` duplicates the class list because the API image ships only
`server/app` — importing the ml package there would crash the service at startup
in production. This test is the other half of that trade: it runs from the repo
root, where both packages are importable, so the duplication is verified rather
than assumed.
"""

from taxonomy import ROSTER

from app.roster import CLASS_ROSTER


def test_api_roster_matches_the_ml_taxonomy() -> None:
    assert CLASS_ROSTER == ROSTER
