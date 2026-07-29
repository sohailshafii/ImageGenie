"""Rebuilding a run's model and classifying with it (M7 C1 spine)."""

import io

import pytest
import torch
from infer import (
    RunNotUsable,
    classify_model,
    evaluate_samples,
    load_run_model,
    predict_probabilities,
    rank_samples,
    rebuild_config,
)
from model import MultiViewCNN
from taxonomy import ROSTER
from train import Config

from app.storage import LocalStorage


def test_rebuild_config_fills_fields_a_run_predates() -> None:
    # Runs 2 and 3 were trained before class_weighting existed; their stored
    # config has no such key, and rebuilding must not fail on that.
    config = rebuild_config({"backbone": "resnet18", "num_classes": 12})

    assert config.backbone == "resnet18"
    assert config.class_weighting == Config().class_weighting


def test_rebuild_config_prefers_the_run_over_current_defaults() -> None:
    """The stored value wins — otherwise a changed default would silently
    rebuild a different architecture than the weights were trained for."""
    config = rebuild_config({"dropout": 0.1, "view_pool": "mean"})

    assert config.dropout == 0.1
    assert config.view_pool == "mean"


def test_rebuild_config_keeps_keys_this_checkout_dropped() -> None:
    config = rebuild_config({"steps_per_epoch": 100})

    assert config.steps_per_epoch == 100


def test_classify_returns_the_whole_roster_best_first(tmp_path) -> None:
    storage = LocalStorage(tmp_path)
    _seed_views(storage, "m1")
    model = MultiViewCNN.from_config(rebuild_config({"pretrained": False}))
    model.eval()

    ranked = classify_model(model, "m1", storage)

    assert len(ranked) == len(ROSTER)
    assert {name for name, _ in ranked} == set(ROSTER)
    probabilities = [probability for _, probability in ranked]
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-5)


def test_probabilities_are_a_distribution_per_item() -> None:
    model = MultiViewCNN.from_config(rebuild_config({"pretrained": False}))
    model.eval()
    views = torch.zeros(2, 12, 3, 224, 224)

    probabilities = predict_probabilities(model, views)

    assert probabilities.shape == (2, len(ROSTER))
    assert probabilities.sum(dim=1).tolist() == pytest.approx([1.0, 1.0], abs=1e-5)


def test_evaluate_samples_reports_against_the_given_split(tmp_path) -> None:
    storage = LocalStorage(tmp_path)
    for uid in ("m1", "m2"):
        _seed_views(storage, uid)
    model = MultiViewCNN.from_config(rebuild_config({"pretrained": False}))
    model.eval()

    report = evaluate_samples(model, [("m1", "chair"), ("m2", "lamp")], storage, "test")

    assert report["split"] == "test"
    assert report["sample_count"] == 2
    assert set(report["per_class"]) == set(ROSTER)


def test_rank_samples_pairs_each_ranking_with_its_own_sample(tmp_path) -> None:
    """Batched scoring must stay aligned with the input, across batch boundaries —
    a shuffled loader would silently attribute one model's prediction to another."""
    storage = LocalStorage(tmp_path)
    samples = [("m1", "chair"), ("m2", "lamp"), ("m3", "car")]
    for uid, _ in samples:
        _seed_views(storage, uid)
    model = MultiViewCNN.from_config(rebuild_config({"pretrained": False}))
    model.eval()

    # batch_size=2 over 3 samples so the last batch is short.
    ranked_samples = list(rank_samples(model, samples, storage, batch_size=2))

    assert [(uid, label) for uid, label, _ in ranked_samples] == samples
    for _, _, ranked in ranked_samples:
        assert len(ranked) == len(ROSTER)
        probabilities = [probability for _, probability in ranked]
        assert probabilities == sorted(probabilities, reverse=True)


def test_a_run_without_weights_is_not_usable(monkeypatch, tmp_path) -> None:
    """A failed or still-running run has no checkpoint — that is a state to
    report, not a crash deep in torch.load."""

    class _Run:
        weights_uri = None
        config: dict = {}
        data_snapshot: dict = {}

        class status:
            value = "running"

    monkeypatch.setattr("infer.session_scope", lambda: _fake_session(_Run()))

    with pytest.raises(RunNotUsable, match="no saved weights"):
        load_run_model(7, LocalStorage(tmp_path))


def test_a_missing_run_is_not_usable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("infer.session_scope", lambda: _fake_session(None))

    with pytest.raises(RunNotUsable, match="no training run"):
        load_run_model(999, LocalStorage(tmp_path))


def _seed_views(storage: LocalStorage, uid: str) -> None:
    """12 valid PNG views, so the loader exercises real decoding."""
    from PIL import Image

    from app.artifact_keys import view_keys

    for key in view_keys(uid):
        buffer = io.BytesIO()
        Image.new("RGB", (224, 224), color=(120, 120, 120)).save(buffer, format="PNG")
        storage.put_bytes(key, buffer.getvalue())


class _fake_session:
    """Minimal `session_scope()` stand-in returning one run from `session.get`."""

    def __init__(self, run):
        self._run = run

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def get(self, _model, _identifier):
        return self._run
