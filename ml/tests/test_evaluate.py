"""Scoring a finished run against a held-out split (M7 C1)."""

from types import SimpleNamespace

import evaluate
import pytest
from splits import stratified_split


def test_the_parser_only_accepts_real_dev_sets() -> None:
    assert evaluate.build_parser().parse_args(["--run", "4"]).dev_set == "test"
    assert evaluate.build_parser().parse_args(
        ["--run", "4", "--dev-set", "lvis"]
    ).dev_set == "lvis"

    with pytest.raises(SystemExit):
        evaluate.build_parser().parse_args(["--run", "4", "--dev-set", "holdout"])


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
        "start_evaluation",
        lambda run_id, dev_set: recorded.update(run_id=run_id, dev_set=dev_set) or 7,
    )
    monkeypatch.setattr(
        evaluate,
        "finish_evaluation",
        lambda evaluation_id, report, label_hash: recorded.update(
            evaluation_id=evaluation_id, report=report, label_hash=label_hash
        ),
    )

    evaluate.evaluate_run(4)

    assert recorded["run_id"] == 4
    assert recorded["dev_set"] == "test"
    # The report lands on the row that was claimed before scoring started, not on
    # a fresh one — otherwise the "scoring…" row would be orphaned.
    assert recorded["evaluation_id"] == 7
    assert recorded["report"] is _REPORT
    # The hash is computed now, not copied from the run — that is the whole point
    # of storing it: it describes the data this report was actually scored on.
    assert recorded["label_hash"].startswith("sha256:")


def test_a_failure_marks_the_evaluation_failed_and_still_raises(monkeypatch) -> None:
    """The case the status column exists for. A job that dies must leave a row
    saying so — and must still fail loudly for the CLI and for Vertex."""
    marked: dict = {}

    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), {})
    monkeypatch.setattr(evaluate, "start_evaluation", lambda run_id, dev_set: 7)
    monkeypatch.setattr(
        evaluate,
        "fail_evaluation",
        lambda evaluation_id, reason: marked.update(
            evaluation_id=evaluation_id, reason=reason
        ),
    )
    # An empty trainable set makes the split empty, which the module refuses with
    # SystemExit — the ordinary way this fails, and not an Exception subclass.
    monkeypatch.setattr(evaluate, "load_trainable_samples", list)

    with pytest.raises(SystemExit):
        evaluate.evaluate_run(4)

    assert marked["evaluation_id"] == 7
    assert "nothing to score" in marked["reason"]


def test_warns_when_the_labels_moved_since_the_run(monkeypatch, capsys) -> None:
    """A run with no recorded split cannot be scored on the partition it held
    out. Reported, not fatal — and the message names the label drift too."""
    samples = [(f"m{index}", "chair") for index in range(10)]

    config = SimpleNamespace(seed=0, backbone="resnet18")
    _stub_run(monkeypatch, config, {"label_hash": "sha256:stale"})
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: samples)
    monkeypatch.setattr(evaluate, "score", lambda *args, **kwargs: _REPORT)

    evaluate.evaluate_run(4)

    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "also changed" in output


def test_still_warns_without_a_recorded_split_even_if_the_labels_held(
    monkeypatch, capsys
) -> None:
    """Unchanged labels are not enough to trust a recomputed partition.

    Runs predating `held_out` also predate hash-bucketed splitting, so the scheme
    that produced their split no longer exists — a matching label_hash says the
    data is the same, not that the partition is (ml.md#dataset-splits).
    """
    samples = [(f"m{index}", "chair") for index in range(10)]
    split = stratified_split(samples, 0)
    current = evaluate.data_snapshot(samples, split)["label_hash"]

    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), {"label_hash": current})
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: samples)
    monkeypatch.setattr(evaluate, "score", lambda *args, **kwargs: _REPORT)

    evaluate.evaluate_run(4)

    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "does not help here" in output


def test_no_warning_when_the_run_recorded_its_split(monkeypatch, capsys) -> None:
    """The replay path is the trustworthy one and must stay quiet — otherwise the
    warning becomes noise that gets ignored on the runs that need it."""
    samples = [(f"m{index}", "chair") for index in range(10)]
    snapshot = {"label_hash": "sha256:whatever", "held_out": {"test": ["m0", "m1", "m2"]}}

    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), snapshot)
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
    monkeypatch.setattr(evaluate, "start_evaluation", lambda run_id, dev_set: 1)
    monkeypatch.setattr(evaluate, "finish_evaluation", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate, "fail_evaluation", lambda *args, **kwargs: None)


def test_replays_the_recorded_partition_instead_of_recomputing(monkeypatch) -> None:
    """The case the whole feature exists for: labels added since the run change
    what `stratified_split` produces, but the recorded uids still name the set
    the run actually held out."""
    trained_on = [(f"m{index}", "chair") for index in range(10)]
    # Two labels added since — enough to reshuffle the recomputed partition.
    samples = trained_on + [("new1", "chair"), ("new2", "lamp")]
    snapshot = {"held_out": {"test": ["m3", "m7"]}, "label_hash": "sha256:stale"}

    selection = evaluate.resolve_scored_samples(
        4, "test", snapshot, samples, samples, stratified_split(samples, 0)
    )

    assert selection.samples == [("m3", "chair"), ("m7", "chair")]


def test_replay_uses_current_labels_not_the_ones_trained_on(monkeypatch) -> None:
    """A corrected label is a better answer to "is the model right?" than the one
    the run trained against — and re-scoring after corrections is the M8 loop."""
    samples = [("m1", "table")]  # was `chair` when the run trained
    snapshot = {"held_out": {"test": ["m1"]}}

    selection = evaluate.resolve_scored_samples(
        4, "test", snapshot, samples, samples, stratified_split(samples, 0)
    )

    assert selection.samples == [("m1", "table")]


def test_recorded_models_that_left_the_set_are_skipped_and_reported(
    monkeypatch, capsys
) -> None:
    samples = [("m1", "chair"), ("m2", "lamp")]
    snapshot = {"held_out": {"test": ["m1", "deleted", "m2"]}}

    selection = evaluate.resolve_scored_samples(
        4, "test", snapshot, samples, samples, stratified_split(samples, 0)
    )

    assert selection.samples == [("m1", "chair"), ("m2", "lamp")]
    assert "1 of 3" in capsys.readouterr().out
    # The denominator is the recorded set, not what survived it — that gap is the
    # whole point of carrying it out of here.
    assert (selection.expected_count, selection.basis) == (3, "held_out")


def test_a_run_without_a_recorded_split_falls_back_and_warns(capsys) -> None:
    """Runs 2-4 predate the field; they still evaluate, with the caveat stated."""
    samples = [(f"m{index}", "chair") for index in range(10)]
    split = stratified_split(samples, 0)
    snapshot = {"label_hash": "sha256:stale"}

    selection = evaluate.resolve_scored_samples(4, "test", snapshot, samples, samples, split)

    assert selection.samples == split.test
    assert "WARNING" in capsys.readouterr().out


def test_train_always_recomputes_since_it_is_never_recorded() -> None:
    samples = [(f"m{index}", "chair") for index in range(10)]
    split = stratified_split(samples, 0)
    snapshot = {"held_out": {"val": ["m1"], "test": ["m2"]}}

    selection = evaluate.resolve_scored_samples(4, "train", snapshot, samples, samples, split)

    assert selection.samples == split.train


def test_a_limited_run_is_scored_against_its_own_subset(monkeypatch) -> None:
    """Regression: evaluating run 4 without reproducing its --limit subsample put
    141 of 1,173 recomputed "test" models into a set the run had trained on."""
    samples = [(f"m{index}", "chair") for index in range(100)]
    samples += [(f"w{index}", "weapon") for index in range(100)]
    # 40 rather than 20 so the recomputed test bucket is non-empty: hashed
    # partitions are proportional in expectation, not exact, so a 20-model subset
    # can legitimately hash to no test models at all.
    snapshot = {"limit": 40}
    config = SimpleNamespace(seed=0, backbone="resnet18")
    scored: dict = {}

    _stub_run(monkeypatch, config, snapshot)
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: samples)
    monkeypatch.setattr(
        evaluate,
        "score",
        lambda model, samples, storage, split_name, num_workers=0: scored.update(
            samples=samples
        )
        or _REPORT,
    )

    evaluate.evaluate_run(4)

    subset = evaluate.subsample(samples, 40, 0)
    expected = stratified_split(subset, 0).test
    assert scored["samples"] == expected
    # And nothing the run trained on leaks into what is scored.
    trained_on = {uid for uid, _ in stratified_split(subset, 0).train}
    assert not [uid for uid, _ in scored["samples"] if uid in trained_on]


def test_a_limited_runs_replay_survives_the_corpus_growing(monkeypatch) -> None:
    """Regression, found on Vertex: run 17 scored 4 of its 45 held-out models.

    Two models gained labels between training and scoring. That is enough to
    change which models a `--limit` run's subset contains, and the replay used to
    look its recorded uids up *in that subset* — so 41 perfectly live, labeled,
    rendered models were dropped and reported as "no longer trainable", leaving a
    0.0% accuracy that rendered on the dashboard as a result.
    """
    trained_on = [(f"m{index}", "chair") for index in range(60)]
    trained_on += [(f"w{index}", "weapon") for index in range(60)]
    held_out = [uid for uid, _ in trained_on[:8]]
    snapshot = {"limit": 20, "held_out": {"test": held_out}}
    # The corpus grows by two, exactly as prod did (11,783 -> 11,785).
    grown = trained_on + [("new1", "lamp"), ("new2", "plant")]
    scored: dict = {}

    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), snapshot)
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: grown)
    monkeypatch.setattr(
        evaluate,
        "score",
        lambda model, samples, storage, split_name, num_workers=0: scored.update(
            samples=samples
        )
        or _REPORT,
    )

    evaluate.evaluate_run(4)

    assert [uid for uid, _ in scored["samples"]] == held_out


# --- How much of the dev set was actually scored ------------------------------
# The blind spot behind backlog item 14. Scoring skips models that left the set,
# which is right, but the report used to carry only the count it managed: an
# evaluation over 4 of run 17's 45 held-out models rendered a full per-class table
# and confusion matrix with nothing anywhere saying the number rested on 4 models.
# These pin the denominator to the report, per path.


def _capture_report(monkeypatch, report: dict) -> dict:
    """Score returning `report`, and hand back what `finish_evaluation` stored."""
    recorded: dict = {}
    monkeypatch.setattr(evaluate, "score", lambda *args, **kwargs: report)
    monkeypatch.setattr(
        evaluate,
        "finish_evaluation",
        lambda evaluation_id, report, label_hash: recorded.update(report=report),
    )
    return recorded


def test_records_what_fraction_of_the_recorded_split_it_scored(monkeypatch) -> None:
    """The run-17 shape: the recorded uids are the denominator, so a replay that
    loses most of them says so instead of reporting a bare count."""
    samples = [("m1", "chair"), ("m2", "lamp")]
    snapshot = {"held_out": {"test": ["m1", "gone", "m2"]}}

    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), snapshot)
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: samples)
    recorded = _capture_report(monkeypatch, dict(_REPORT))

    evaluate.evaluate_run(4)

    assert recorded["report"]["coverage"] == {
        "expected": 3,
        "scored": 2,
        "basis": "held_out",
    }


def test_a_recomputed_partition_is_measured_against_the_recorded_split_size(
    monkeypatch,
) -> None:
    """Runs 2-4 recorded no uids but did record how big each split was. A weaker
    denominator — it says nothing about *which* models — so it is named apart."""
    samples = [(f"m{index}", "chair") for index in range(10)]
    # Sized from the partition this recomputes, so the assertion is about where
    # the denominator came from rather than about a hashed bucket's exact size.
    recomputed = stratified_split(samples, 0)
    snapshot = {
        "label_hash": None,
        "splits": {
            "train": len(recomputed.train),
            "val": len(recomputed.val),
            "test": len(recomputed.test),
        },
    }

    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), snapshot)
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: samples)
    recorded = _capture_report(monkeypatch, dict(_REPORT))

    evaluate.evaluate_run(4)

    coverage = recorded["report"]["coverage"]
    assert coverage["expected"] == len(recomputed.test)
    assert coverage["basis"] == "recorded_split_size"


def test_no_coverage_claim_when_there_is_nothing_to_compare_against(monkeypatch) -> None:
    """Absent, not zero and not 1.0. A run recording neither its held-out uids nor
    its split sizes has no expected count, and inventing one would be a claim the
    data cannot support — "no claim" has to stay distinct from "complete"."""
    samples = [(f"m{index}", "chair") for index in range(10)]

    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), {"label_hash": None})
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: samples)
    recorded = _capture_report(monkeypatch, dict(_REPORT))

    evaluate.evaluate_run(4)

    assert "coverage" not in recorded["report"]


def test_coverage_counts_the_models_the_metrics_were_computed_over(monkeypatch) -> None:
    """`sample_count`, not the length of the resolved list. The resolver's list is
    what scoring was asked for; a model can still drop out below that."""
    samples = [("m1", "chair"), ("m2", "lamp")]
    snapshot = {"held_out": {"test": ["m1", "m2"]}}
    report = {**_REPORT, "sample_count": 1}

    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), snapshot)
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: samples)
    recorded = _capture_report(monkeypatch, report)

    evaluate.evaluate_run(4)

    assert recorded["report"]["coverage"]["scored"] == 1


def test_stamping_coverage_leaves_the_scored_report_alone(monkeypatch) -> None:
    """A new dict, never a mutation: the report belongs to whoever produced it."""
    report = dict(_REPORT)
    selection = evaluate.ScoredSelection([("m1", "chair")], 45, "held_out")

    stamped = evaluate.with_coverage(report, selection)

    assert "coverage" not in report
    assert stamped["coverage"]["expected"] == 45


# --- Refusing an evaluation that lost most of its dev set ---------------------
# Coverage on its own only describes; these decide. Refuse below 50%, mark below
# 90%, and stay quiet above — the thresholds and the evidence behind them are in
# `evaluate.MIN_SCORED_FRACTION`.


def _floor_run(monkeypatch, samples, snapshot) -> dict:
    """A run whose scoring is stubbed, so only the floor decides the outcome."""
    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), snapshot)
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: samples)
    return _capture_report(monkeypatch, dict(_REPORT))


def test_an_evaluation_that_lost_most_of_its_dev_set_is_refused(monkeypatch) -> None:
    """The run-17 shape, at the scale it happened: 4 of 45 recorded models."""
    samples = [(f"m{index}", "chair") for index in range(4)]
    snapshot = {"held_out": {"test": [f"m{index}" for index in range(45)]}}
    marked: dict = {}

    _floor_run(monkeypatch, samples, snapshot)
    monkeypatch.setattr(
        evaluate,
        "fail_evaluation",
        lambda evaluation_id, reason: marked.update(reason=reason),
    )

    with pytest.raises(SystemExit, match="refusing to score"):
        evaluate.evaluate_run(4)

    # The refusal reaches the row, with the arithmetic intact — a job that dies
    # unattended is only legible if the page can say what stopped it.
    assert "4 of the 45 test models" in marked["reason"]
    assert "8.9%" in marked["reason"]


def test_refusing_costs_nothing_to_score(monkeypatch) -> None:
    """Refused before `score()`, not after. Counting on the stored `sample_count`
    would mean paying for the GPU minutes first."""
    samples = [("m0", "chair")]
    snapshot = {"held_out": {"test": [f"m{index}" for index in range(45)]}}
    scored: dict = {}

    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), snapshot)
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: samples)
    monkeypatch.setattr(
        evaluate, "score", lambda *args, **kwargs: scored.update(ran=True) or _REPORT
    )

    with pytest.raises(SystemExit):
        evaluate.evaluate_run(4)

    assert scored == {}


def test_a_small_shortfall_is_scored_and_marked(monkeypatch, capsys) -> None:
    """Between the floor and the marking threshold a partial number is still
    worth having — as long as it says what it is."""
    samples = [(f"m{index}", "chair") for index in range(8)]
    snapshot = {"held_out": {"test": [f"m{index}" for index in range(10)]}}

    recorded = _floor_run(monkeypatch, samples, snapshot)
    evaluate.evaluate_run(4)

    assert "WARNING" in capsys.readouterr().out
    assert recorded["report"]["coverage"]["expected"] == 10


def test_a_dev_set_that_barely_moved_is_scored_quietly(monkeypatch, capsys) -> None:
    """The `lvis` case: 982 of 984 is two models that never finished ingesting,
    and warning about it would train everyone to ignore the warning."""
    dev_set = [(f"m{index}", "chair") for index in range(100)]

    _stub_lvis(monkeypatch, dev_set, rendered=[f"m{index}" for index in range(98)],
               trainable=[])
    evaluate.evaluate_run(4, "lvis")

    assert "WARNING" not in capsys.readouterr().out


def test_min_coverage_lowers_the_floor_but_not_the_mark(monkeypatch, capsys) -> None:
    """The escape hatch: a human with a checkout can score a set they know is
    thin. It removes the refusal, never the record of what was scored."""
    samples = [(f"m{index}", "chair") for index in range(4)]
    snapshot = {"held_out": {"test": [f"m{index}" for index in range(45)]}}

    recorded = _floor_run(monkeypatch, samples, snapshot)
    evaluate.evaluate_run(4, min_coverage=0)

    assert "WARNING" in capsys.readouterr().out
    assert recorded["report"]["coverage"] == {
        "expected": 45,
        "scored": 2,
        "basis": "held_out",
    }


def test_min_coverage_still_applies_when_raised_above_the_mark(monkeypatch) -> None:
    """An override that silently stopped applying past 90% would be worse than no
    override — so the refusal is tested before the marking threshold."""
    samples = [(f"m{index}", "chair") for index in range(19)]
    snapshot = {"held_out": {"test": [f"m{index}" for index in range(20)]}}

    _floor_run(monkeypatch, samples, snapshot)

    with pytest.raises(SystemExit, match="refusing to score"):
        evaluate.evaluate_run(4, min_coverage=0.99)


def test_an_unknown_expected_count_cannot_refuse(monkeypatch) -> None:
    """A refusal has to rest on a denominator the run actually recorded. Runs
    recording neither their held-out uids nor their split sizes still evaluate."""
    samples = [(f"m{index}", "chair") for index in range(10)]

    recorded = _floor_run(monkeypatch, samples, {"label_hash": None})
    evaluate.evaluate_run(4)

    assert "coverage" not in recorded["report"]


def test_coverage_above_one_is_not_a_shortfall(monkeypatch, capsys) -> None:
    """Runs 2-4 are measured against the size their split recorded, and the corpus
    has only grown since — exceeding it is ordinary, not wrong."""
    samples = [(f"m{index}", "chair") for index in range(60)]
    snapshot = {"label_hash": None, "splits": {"train": 8, "val": 1, "test": 1}}

    _floor_run(monkeypatch, samples, snapshot)
    evaluate.evaluate_run(4)

    assert "WARNING: scoring only" not in capsys.readouterr().out


def test_the_parser_carries_the_floor_and_can_be_told_to_drop_it() -> None:
    parser = evaluate.build_parser()
    assert parser.parse_args(["--run", "4"]).min_coverage == evaluate.MIN_SCORED_FRACTION
    assert parser.parse_args(["--run", "4", "--min-coverage", "0"]).min_coverage == 0


# --- The second dev set (FR-7) ----------------------------------------------
# `lvis` is not a partition of our corpus, so none of the split machinery above
# applies to it: no seed, no replay, no recomputation. What it needs instead is
# proof that the two things which would silently invalidate it are handled —
# models that have not rendered yet, and models that have acquired a label.


def _stub_lvis(monkeypatch, dev_set, rendered, trainable) -> dict:
    scored: dict = {}
    _stub_run(monkeypatch, SimpleNamespace(seed=0, backbone="resnet18"), {})
    monkeypatch.setattr(evaluate, "load_dev_set", lambda: dev_set)
    monkeypatch.setattr(evaluate, "load_rendered_uids", lambda uids: set(rendered))
    monkeypatch.setattr(evaluate, "load_trainable_samples", lambda: trainable)
    monkeypatch.setattr(
        evaluate,
        "score",
        lambda model, samples, storage, split_name, num_workers=0: scored.update(
            samples=samples, split_name=split_name
        )
        or _REPORT,
    )
    return scored


def test_lvis_scores_only_the_models_that_have_rendered(monkeypatch, capsys) -> None:
    dev_set = [("a", "chair"), ("b", "lamp"), ("c", "car")]

    scored = _stub_lvis(monkeypatch, dev_set, rendered=["a", "c"], trainable=[])
    evaluate.evaluate_run(4, "lvis")

    assert scored["samples"] == [("a", "chair"), ("c", "car")]
    assert scored["split_name"] == "lvis"
    assert "1 of 3 dev-set models are not rendered" in capsys.readouterr().out


def test_lvis_drops_models_that_acquired_a_label(monkeypatch, capsys) -> None:
    """A labeled dev-set model is a trainable one, and scoring a run on its own
    training data is the exact failure this dev set exists to avoid."""
    dev_set = [("a", "chair"), ("b", "lamp")]

    scored = _stub_lvis(
        monkeypatch, dev_set, rendered=["a", "b"], trainable=[("b", "weapon")]
    )
    evaluate.evaluate_run(4, "lvis")

    assert scored["samples"] == [("a", "chair")]
    assert "carry a label in the database" in capsys.readouterr().out


def test_lvis_refuses_to_score_before_the_data_lands(monkeypatch) -> None:
    _stub_lvis(monkeypatch, [("a", "chair")], rendered=[], trainable=[])
    with pytest.raises(SystemExit, match="are rendered yet"):
        evaluate.evaluate_run(4, "lvis")


def test_lvis_refuses_when_everything_is_contaminated(monkeypatch) -> None:
    _stub_lvis(
        monkeypatch, [("a", "chair")], rendered=["a"], trainable=[("a", "chair")]
    )
    with pytest.raises(SystemExit, match="nothing independent left"):
        evaluate.evaluate_run(4, "lvis")


def test_lvis_fingerprints_the_pairs_it_actually_scored(monkeypatch) -> None:
    """The stored hash must identify the dev set, not the corpus — a partition
    report hashes the trainable set, but `lvis` has no relationship to it."""
    dev_set = [("a", "chair"), ("b", "lamp")]
    recorded: dict = {}

    _stub_lvis(monkeypatch, dev_set, rendered=["a", "b"], trainable=[])
    monkeypatch.setattr(
        evaluate,
        "start_evaluation",
        lambda run_id, dev_set_name: recorded.update(dev_set=dev_set_name) or 1,
    )
    monkeypatch.setattr(
        evaluate,
        "finish_evaluation",
        lambda evaluation_id, report, label_hash: recorded.update(
            label_hash=label_hash
        ),
    )

    evaluate.evaluate_run(4, "lvis")

    assert recorded["dev_set"] == "lvis"
    assert recorded["label_hash"] == evaluate.label_hash(dev_set)


def test_lvis_coverage_is_measured_against_the_whole_selected_dev_set(monkeypatch) -> None:
    """The denominator is the selection, not what survived ingestion: these 1,000
    objects were chosen and balanced deliberately, so a model missing from the
    scoring is one that never arrived rather than one that was never wanted."""
    dev_set = [("a", "chair"), ("b", "lamp"), ("c", "car")]

    _stub_lvis(monkeypatch, dev_set, rendered=["a", "c"], trainable=[])
    recorded = _capture_report(monkeypatch, dict(_REPORT))

    evaluate.evaluate_run(4, "lvis")

    assert recorded["report"]["coverage"] == {
        "expected": 3,
        "scored": 2,
        "basis": "selected_dev_set",
    }
