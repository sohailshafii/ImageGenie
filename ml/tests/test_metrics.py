"""Per-class evaluation metrics (B4, ml.md#metrics).

The load-bearing cases are the degenerate ones: a class the model never predicts,
a class absent from the split, and a model that has collapsed onto the majority
class. Those are what these metrics exist to expose, and what an accuracy number
hides.
"""

import pytest
from metrics import (
    confusion_matrix,
    evaluation_report,
    per_class_metrics,
)

CLASSES = ("chair", "lamp", "table")


def test_confusion_matrix_puts_truth_on_the_row() -> None:
    """Orientation is the easiest thing to invert, and inverting it silently
    swaps precision and recall everywhere downstream."""
    # one true-chair predicted as lamp
    matrix = confusion_matrix([0], [1], num_classes=3)

    assert matrix[0][1] == 1
    assert matrix[1][0] == 0


def test_confusion_matrix_counts_every_example() -> None:
    true = [0, 0, 1, 2, 2, 2]
    predicted = [0, 1, 1, 2, 2, 0]

    matrix = confusion_matrix(true, predicted, num_classes=3)

    assert sum(sum(row) for row in matrix) == len(true)
    assert matrix[0] == [1, 1, 0]
    assert matrix[2] == [1, 0, 2]


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        confusion_matrix([0, 1], [0], num_classes=3)


def test_precision_and_recall_on_a_simple_case() -> None:
    #   true chair x2: one right, one called lamp
    #   true lamp  x1: right
    matrix = confusion_matrix([0, 0, 1], [0, 1, 1], num_classes=3)

    metrics = per_class_metrics(matrix, CLASSES)

    assert metrics["chair"]["recall"] == pytest.approx(0.5)  # 1 of 2 true chairs
    assert metrics["chair"]["precision"] == pytest.approx(1.0)  # 1 of 1 predicted
    assert metrics["lamp"]["recall"] == pytest.approx(1.0)
    assert metrics["lamp"]["precision"] == pytest.approx(0.5)  # 1 of 2 predicted
    assert metrics["chair"]["support"] == 2


def test_a_never_predicted_class_has_undefined_precision_not_zero() -> None:
    """0.0 would claim the model predicted `table` and got them all wrong. It
    never predicted it at all — a different, and less damning, statement."""
    matrix = confusion_matrix([0, 1], [0, 1], num_classes=3)

    metrics = per_class_metrics(matrix, CLASSES)

    assert metrics["table"]["precision"] is None
    assert metrics["table"]["f1"] is None


def test_a_class_absent_from_the_split_has_undefined_recall() -> None:
    matrix = confusion_matrix([0, 1], [0, 1], num_classes=3)

    assert per_class_metrics(matrix, CLASSES)["table"]["recall"] is None
    assert per_class_metrics(matrix, CLASSES)["table"]["support"] == 0


def test_f1_is_zero_when_both_sides_are_defined_but_wrong() -> None:
    """Distinct from undefined: here the model *did* predict the class, and
    missed every time."""
    matrix = confusion_matrix([0, 1], [1, 0], num_classes=3)

    metrics = per_class_metrics(matrix, CLASSES)

    assert metrics["chair"]["precision"] == pytest.approx(0.0)
    assert metrics["chair"]["recall"] == pytest.approx(0.0)
    assert metrics["chair"]["f1"] == pytest.approx(0.0)


def test_macro_average_ignores_undefined_classes() -> None:
    matrix = confusion_matrix([0, 1], [0, 1], num_classes=3)  # table never seen

    report = evaluation_report([0, 1], [0, 1], CLASSES, split="val")

    # Both defined classes are perfect; `table` is undefined and excluded rather
    # than dragging the mean toward zero.
    assert report["macro_precision"] == pytest.approx(1.0)
    assert matrix[2] == [0, 0, 0]


def test_the_collapsed_model_is_what_the_report_exposes() -> None:
    """The case B4 exists for: a model that answers `chair` for everything.

    Accuracy looks like the majority-class share, while macro recall collapses —
    the gap between those two numbers is the tell.
    """
    true = [0] * 8 + [1] * 1 + [2] * 1  # 80% chair, mirroring the real skew
    predicted = [0] * 10  # always chair

    report = evaluation_report(true, predicted, CLASSES, split="val")

    assert report["accuracy"] == pytest.approx(0.8)  # flattering
    assert report["macro_recall"] == pytest.approx(1 / 3)  # honest: 1.0, 0, 0
    assert report["per_class"]["lamp"]["recall"] == pytest.approx(0.0)
    assert report["per_class"]["lamp"]["precision"] is None  # never predicted
    # Every off-diagonal example landed in the `chair` column.
    assert [row[0] for row in report["confusion"]["matrix"]] == [8, 1, 1]


def test_report_is_json_able_and_self_describing() -> None:
    import json

    report = evaluation_report([0, 1, 2], [0, 1, 2], CLASSES, split="val")

    assert json.loads(json.dumps(report)) == report  # goes straight into JSONB
    assert report["split"] == "val"
    assert report["sample_count"] == 3
    assert report["confusion"]["classes"] == list(CLASSES)


def test_an_empty_split_reports_null_accuracy_rather_than_dividing_by_zero() -> None:
    report = evaluation_report([], [], CLASSES, split="val")

    assert report["sample_count"] == 0
    assert report["accuracy"] is None
