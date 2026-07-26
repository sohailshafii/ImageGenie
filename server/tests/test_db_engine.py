"""How the engine is built (server.md#training-gpu).

Three call sites want three different transports and must not drift: workers and
the API dial Postgres directly, while the Vertex training job reads its URL from
Secret Manager and dials Cloud SQL through the connector. The wiring is what is
tested here — never a real connection.
"""

import pytest

from app import config, db


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    """`get_engine` and `get_settings` are both lru_cached, so a test that
    changed config would otherwise leak into the next one."""
    db.get_engine.cache_clear()
    config.get_settings.cache_clear()
    yield
    db.get_engine.cache_clear()
    config.get_settings.cache_clear()


def test_the_plain_path_uses_database_url_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    """What every worker and the API do — unchanged by the training work."""
    captured = {}

    def fake_create_engine(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return "engine"

    monkeypatch.setattr(db, "create_engine", fake_create_engine)
    monkeypatch.setattr(
        db, "get_settings", lambda: config.Settings(database_url="postgresql+psycopg://u:p@h/d")
    )

    assert db.get_engine() == "engine"
    assert captured["url"] == "postgresql+psycopg://u:p@h/d"
    assert "creator" not in captured["kwargs"]  # no connector involved


def test_the_url_can_come_from_secret_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """The training job passes a secret *name*: a Vertex job's environment is
    visible in its metadata, and the URL carries the DB password."""
    settings = config.Settings(
        database_url="postgresql+psycopg://ignored:ignored@ignored/ignored",
        database_url_secret="projects/1/secrets/db-url/versions/latest",
    )

    class FakeResponse:
        class payload:  # noqa: N801 — mirrors the client's own shape
            data = b"postgresql+psycopg://real:secret@host/imagegenie\n"

    class FakeClient:
        def __init__(self) -> None:
            self.asked_for = None

        def access_secret_version(self, name: str):
            self.asked_for = name
            return FakeResponse()

    fake_client = FakeClient()
    fake_module = type("m", (), {"SecretManagerServiceClient": lambda: fake_client})
    monkeypatch.setitem(
        __import__("sys").modules, "google.cloud.secretmanager", fake_module
    )

    resolved = db._resolve_database_url(settings)

    assert resolved == "postgresql+psycopg://real:secret@host/imagegenie"  # trailing \n stripped
    assert fake_client.asked_for == "projects/1/secrets/db-url/versions/latest"


def test_the_secret_is_only_read_when_configured() -> None:
    """Workers must not pay a Secret Manager call — or need the permission."""
    settings = config.Settings(database_url="postgresql+psycopg://u:p@h/d")
    assert db._resolve_database_url(settings) == "postgresql+psycopg://u:p@h/d"


def test_the_connector_reuses_the_urls_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the transport changes, so the two paths cannot disagree about which
    database — or which user — they mean."""
    captured = {}

    class FakeConnector:
        def connect(self, instance, driver, **kwargs):
            captured["instance"] = instance
            captured["driver"] = driver
            captured.update(kwargs)
            return "dbapi-connection"

    fake_module = type("m", (), {"Connector": FakeConnector})
    monkeypatch.setitem(__import__("sys").modules, "google.cloud.sql.connector", fake_module)

    def fake_create_engine(url, **kwargs):
        captured["url"] = url
        kwargs["creator"]()  # the engine would call this to open a connection
        captured["pre_ping"] = kwargs.get("pool_pre_ping")
        return "engine"

    monkeypatch.setattr(db, "create_engine", fake_create_engine)

    settings = config.Settings(cloudsql_instance="proj:region:instance")
    db._cloudsql_connector_engine(settings, "postgresql+psycopg://alice:pw@h/imagegenie")

    assert captured["instance"] == "proj:region:instance"
    assert captured["driver"] == "pg8000"
    assert captured["user"] == "alice"
    assert captured["password"] == "pw"
    assert captured["db"] == "imagegenie"
    # A long run sits idle on the DB between epoch writes, long enough for Cloud
    # SQL to drop a stale connection.
    assert captured["pre_ping"] is True


def test_a_percent_encoded_password_is_decoded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The URL form escapes reserved characters; the connector takes raw values,
    so a password with an `@` or `/` would otherwise authenticate as garbage."""
    captured = {}

    class FakeConnector:
        def connect(self, instance, driver, **kwargs):
            captured.update(kwargs)
            return "dbapi-connection"

    monkeypatch.setitem(
        __import__("sys").modules,
        "google.cloud.sql.connector",
        type("m", (), {"Connector": FakeConnector}),
    )
    monkeypatch.setattr(db, "create_engine", lambda url, **kw: kw["creator"]() and "engine")

    settings = config.Settings(cloudsql_instance="proj:region:instance")
    db._cloudsql_connector_engine(settings, "postgresql+psycopg://user:p%40ss%2Fword@h/imagegenie")

    assert captured["password"] == "p@ss/word"
