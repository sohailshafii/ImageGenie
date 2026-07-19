import pytest

from app.config import Settings


def test_defaults() -> None:
    settings = Settings()
    assert settings.pubsub_project == "imagegenie-local"
    assert settings.download_topic == "download-jobs"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGEGENIE_PUBSUB_PROJECT", "custom-project")
    assert Settings().pubsub_project == "custom-project"
