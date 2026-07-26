"""Per-class evaluation metrics: confusion matrix, precision, recall (B4, FR-7).

Pure functions over predicted/true class indices — no torch, no sklearn, no I/O.
That keeps them usable from three places without dragging dependencies around:
the training loop's end-of-run report (B4), the M7 evaluation over both dev sets
(ml.md#evaluation), and their own tests.

sklearn would do this in one call, but it is a heavy dependency to add to a CUDA
training image for ~40 lines of counting, and the undefined-metric convention
below is one we want to be explicit about rather than inherit.

**Why per-class and not just accuracy.** The corpus is skewed ~7.7:1 (`weapon`
alone is ~18%), so a model that predicts the majority class for everything scores
~18% accuracy and 0 recall on eleven classes. Only the per-class view — and the
confusion matrix's off-diagonal mass — shows that.
"""

from __future__ import annotations

from collections.abc import Sequence


def confusion_matrix(
    true_indices: Sequence[int], predicted_indices: Sequence[int], num_classes: int
) -> list[list[int]]:
    """``matrix[i][j]`` = examples whose **true** class is i, predicted as j.

    Row-major with truth as the row, matching ml.md#metrics — the diagonal is
    correct predictions and a heavy off-diagonal column means the model is
    dumping other classes into that one.
    """
    if len(true_indices) != len(predicted_indices):
        raise ValueError(
            f"true/predicted length mismatch: {len(true_indices)} vs {len(predicted_indices)}"
        )
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true_index, predicted_index in zip(true_indices, predicted_indices, strict=True):
        matrix[true_index][predicted_index] += 1
    return matrix


def _ratio(numerator: int, denominator: int) -> float | None:
    """``numerator/denominator``, or None when the denominator is zero.

    None rather than 0.0 on purpose. A class the model never predicted has
    *undefined* precision, not zero precision — reporting 0.0 would claim it
    predicted that class and got them all wrong, which is a different and worse
    statement about the model. The distinction matters exactly when a run has
    collapsed onto the majority class, which is the case these metrics exist to
    expose.
    """
    if denominator == 0:
        return None
    return numerator / denominator


def per_class_metrics(
    matrix: list[list[int]], class_names: Sequence[str]
) -> dict[str, dict[str, float | int | None]]:
    """Precision, recall, F1 and support for each class, keyed by class name.

    - **precision** = correct_i / everything predicted as i (column sum)
    - **recall**    = correct_i / everything truly i (row sum, i.e. support)
    - **f1**        = harmonic mean, or None if either side is undefined
    """
    class_to_metrics: dict[str, dict[str, float | int | None]] = {}
    for index, class_name in enumerate(class_names):
        correct = matrix[index][index]
        support = sum(matrix[index])  # true count for this class
        predicted_count = sum(row[index] for row in matrix)

        precision = _ratio(correct, predicted_count)
        recall = _ratio(correct, support)
        if precision is None or recall is None or precision + recall == 0:
            f1 = None if precision is None or recall is None else 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        class_to_metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return class_to_metrics


def _macro_average(
    class_to_metrics: dict[str, dict[str, float | int | None]], metric: str
) -> float | None:
    """Unweighted mean of a metric over the classes where it is defined.

    Macro, not micro: micro-averaging collapses to plain accuracy, which on a
    7.7:1 corpus is dominated by the majority class. The macro average weights a
    278-example class the same as a 2,134-example one, which is what makes a
    collapsed model look as bad as it is.
    """
    values = [
        metrics[metric]
        for metrics in class_to_metrics.values()
        if isinstance(metrics[metric], float)
    ]
    if not values:
        return None
    return sum(values) / len(values)


def evaluation_report(
    true_indices: Sequence[int],
    predicted_indices: Sequence[int],
    class_names: Sequence[str],
    split: str,
) -> dict:
    """The full report stored in ``training_run.metrics`` (NFR-4, FR-7).

    A plain JSON-able dict rather than a typed object: it goes straight into a
    JSONB column, and the dashboard renders it generically, so adding a metric
    later needs no migration and no frontend change (config-over-code).
    """
    matrix = confusion_matrix(true_indices, predicted_indices, len(class_names))
    class_to_metrics = per_class_metrics(matrix, class_names)
    total = len(true_indices)
    correct = sum(matrix[index][index] for index in range(len(class_names)))
    return {
        "split": split,
        "sample_count": total,
        "accuracy": _ratio(correct, total),
        # Reported alongside accuracy because they disagree loudly on an
        # imbalanced corpus, and that disagreement is the signal.
        "macro_precision": _macro_average(class_to_metrics, "precision"),
        "macro_recall": _macro_average(class_to_metrics, "recall"),
        "macro_f1": _macro_average(class_to_metrics, "f1"),
        "per_class": class_to_metrics,
        "confusion": {"classes": list(class_names), "matrix": matrix},
    }
