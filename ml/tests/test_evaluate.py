"""Scoring a finished run against a held-out split (M7 C1)."""

from types import SimpleNamespace

import evaluate
import pytest
from splits import stratified_split


def test_the_parser_only_accepts_real_splits() -> None:
    assert evaluate.build_parser().parse_args(["--run", "4"]).split == "test"

    with pytest.raises(SystemExit):
        evaluate.build_parser().parse_args(["--run", "4", "--split", "holdout"])


def test_scores_the_requested_split_using_the_runs_own_seed(monkeypatch, capsys) -> None:
    samples = [(f"m{index}", "chair") for index in range(10)]
    samples += [(f"w{index}", "weapon") for index in range(10)]
    # A non-default seed, so a wrong one produces a different partition and the
    # assertion below catches it.
    config = SimpleNamespace(seed=7, backbone="resnet18")
    expected = stratified_split(samples, 7).test
    scored: dict = {}

    _stub_run(monkeypatch, config, snapshot={"label_hash": None})
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: samples)
    monkeypatch.setattr(
        evaluate,
        "score",
        lambda model, samples, storage, split_name, num_workers=0: scored.update(
            samples=samples, split_name=split_name
        )
        or _REPORT,
    )

    evaluate.evaluate_run(4)

    assert scored["samples"] == expected
    assert scored["split_name"] == "test"


def test_records_the_report_with_the_current_label_hash(monkeypatch) -> None:
    samples = [(f"m{index}", "chair") for index in range(10)]
    recorded: dict = {}

    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), {"label_hash": None})
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: samples)
    monkeypatch.setattr(
        evaluate, "score", lambda *args, **kwargs: _REPORT
    )
    monkeypatch.setattr(
        evaluate,
        "record_evaluation",
        lambda run_id, dev_set, report, label_hash: recorded.update(
            run_id=run_id, dev_set=dev_set, report=report, label_hash=label_hash
        )
        or 1,
    )

    evaluate.evaluate_run(4)

    assert recorded["run_id"] == 4
    assert recorded["dev_set"] == "test"
    assert recorded["report"] is _REPORT
    # The hash is computed now, not copied from the run — that is the whole point
    # of storing it: it describes the data this report was actually scored on.
    assert recorded["label_hash"].startswith("sha256:")


def test_warns_when_the_labels_moved_since_the_run(monkeypatch, capsys) -> None:
    """A label added since the run shifts the partition, so the recomputed
    `test` split is not the one the run held out. Reported, not fatal."""
    samples = [(f"m{index}", "chair") for index in range(10)]

    config = SimpleNamespace(seed=0, backbone="resnet18")
    _stub_run(monkeypatch, config, {"label_hash": "sha256:stale"})
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: samples)
    monkeypatch.setattr(evaluate, "score", lambda *args, **kwargs: _REPORT)

    evaluate.evaluate_run(4)

    assert "WARNING" in capsys.readouterr().out


def test_no_warning_when_the_labels_are_unchanged(monkeypatch, capsys) -> None:
    samples = [(f"m{index}", "chair") for index in range(10)]
    split = stratified_split(samples, 0)
    current = evaluate.data_snapshot(samples, split)["label_hash"]

    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), {"label_hash": current})
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: samples)
    monkeypatch.setattr(evaluate, "score", lambda *args, **kwargs: _REPORT)

    evaluate.evaluate_run(4)

    assert "WARNING" not in capsys.readouterr().out


def test_an_empty_split_is_refused(monkeypatch) -> None:
    """Better than reporting metrics over zero samples, which read as a result."""
    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), {"label_hash": None})
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: [("m0", "chair")])

    with pytest.raises(SystemExit, match="empty"):
        evaluate.evaluate_run(4)


_REPORT = {"accuracy": 0.5, "macro_recall": 0.4, "split": "test", "sample_count": 2}


def _stub_run(monkeypatch, config, snapshot) -> None:
    """Replace the DB/storage-backed pieces so these tests stay hermetic."""
    monkeypatch.setattr(evaluate, "build_storage", lambda _settings: None)
    monkeypatch.setattr(evaluate, "get_settings", lambda: None)
    monkeypatch.setattr(
        evaluate, "load_run_model", lambda run_id, storage: (None, config, snapshot)
    )
    monkeypatch.setattr(evaluate, "record_evaluation", lambda *args, **kwargs: 1)


def test_replays_the_recorded_partition_instead_of_recomputing(monkeypatch) -> None:
    """The case the whole feature exists for: labels added since the run change
    what `stratified_split` produces, but the recorded uids still name the set
    the run actually held out."""
    trained_on = [(f"m{index}", "chair") for index in range(10)]
    # Two labels added since — enough to reshuffle the recomputed partition.
    samples = trained_on + [("new1", "chair"), ("new2", "lamp")]
    snapshot = {"held_out": {"test": ["m3", "m7"]}, "label_hash": "sha256:stale"}

    scored = evaluate.resolve_scored_samples(
        4, "test", snapshot, samples, stratified_split(samples, 0)
    )

    assert scored == [("m3", "chair"), ("m7", "chair")]


def test_replay_uses_current_labels_not_the_ones_trained_on(monkeypatch) -> None:
    """A corrected label is a better answer to "is the model right?" than the one
    the run trained against — and re-scoring after corrections is the M8 loop."""
    samples = [("m1", "table")]  # was `chair` when the run trained
    snapshot = {"held_out": {"test": ["m1"]}}

    scored = evaluate.resolve_scored_samples(
        4, "test", snapshot, samples, stratified_split(samples, 0)
    )

    assert scored == [("m1", "table")]


def test_recorded_models_that_left_the_set_are_skipped_and_reported(
    monkeypatch, capsys
) -> None:
    samples = [("m1", "chair"), ("m2", "lamp")]
    snapshot = {"held_out": {"test": ["m1", "deleted", "m2"]}}

    scored = evaluate.resolve_scored_samples(
        4, "test", snapshot, samples, stratified_split(samples, 0)
    )

    assert scored == [("m1", "chair"), ("m2", "lamp")]
    assert "1 of 3" in capsys.readouterr().out


def test_a_run_without_a_recorded_split_falls_back_and_warns(capsys) -> None:
    """Runs 2-4 predate the field; they still evaluate, with the caveat stated."""
    samples = [(f"m{index}", "chair") for index in range(10)]
    split = stratified_split(samples, 0)
    snapshot = {"label_hash": "sha256:stale"}

    scored = evaluate.resolve_scored_samples(4, "test", snapshot, samples, split)

    assert scored == split.test
    assert "WARNING" in capsys.readouterr().out


def test_train_always_recomputes_since_it_is_never_recorded() -> None:
    samples = [(f"m{index}", "chair") for index in range(10)]
    split = stratified_split(samples, 0)
    snapshot = {"held_out": {"val": ["m1"], "test": ["m2"]}}

    scored = evaluate.resolve_scored_samples(4, "train", snapshot, samples, split)

    assert scored == split.train
