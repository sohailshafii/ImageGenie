// Shape of the B4 metrics blob and the guard that recognises it, split out of
// MetricsReport.tsx: a .tsx exporting both a component and a helper breaks fast
// refresh, and this half is plain data anyway.

type ClassMetrics = {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  support: number;
};

/**
 * How much of its dev set an evaluation actually covered (ml.md#how-much-of-the-
 * dev-set-a-report-covers). Optional because it is: every evaluation stored
 * before this shipped has no such key, and one whose run recorded no expected
 * count deliberately omits it — "makes no claim" is not the same as "complete".
 */
export type ReportCoverage = {
  expected: number;
  scored: number;
  basis: string;
};

export type EvaluationReport = {
  split: string;
  sample_count: number;
  coverage?: ReportCoverage;
  accuracy: number | null;
  macro_precision: number | null;
  macro_recall: number | null;
  macro_f1: number | null;
  per_class: Record<string, ClassMetrics>;
  confusion: { classes: string[]; matrix: number[][] };
};

/**
 * Below this share of the dev set, a report is drawn with the shortfall called
 * out rather than merely stated.
 *
 * A mirror of `MARK_SCORED_FRACTION` in `ml/evaluate.py`, which is where the
 * evidence for the number lives, and unpinned by anything: nothing here can
 * import Python, and there are no frontend tests to assert the copy. The drift
 * it risks is cosmetic — a banner appearing at a slightly different shortfall —
 * so the alternative, freezing the verdict into the stored report at scoring
 * time, buys less than it costs. A stored row keeps its numbers; how loudly they
 * are read is a decision for whoever is reading them now.
 */
export const MARK_SCORED_FRACTION = 0.9;

/**
 * Below this many models, the per-class table is noise rather than measurement.
 *
 * A different failure from a shortfall, and invisible to coverage: 45 of 45 is
 * complete and still spreads 45 models across 12 classes, leaving most of them
 * with one or two examples. Perfect coverage of a tiny dev set reads exactly
 * like a real result too.
 */
export const THIN_SAMPLE_COUNT = 30;

/**
 * The report's coverage, or null when it makes no usable claim.
 *
 * Defensive because the blob is JSONB written by another codebase entirely: an
 * older row has no coverage at all, and a malformed one must render the report
 * without it rather than take the page down. Deliberately *not* part of
 * `isEvaluationReport` — coverage is something a report may say, never something
 * it must say to be drawable.
 */
export function readCoverage(report: EvaluationReport): ReportCoverage | null {
  const coverage = report.coverage as Partial<ReportCoverage> | undefined;
  if (
    typeof coverage?.expected !== 'number' ||
    typeof coverage?.scored !== 'number' ||
    coverage.expected <= 0
  ) {
    return null;
  }
  return {
    expected: coverage.expected,
    scored: coverage.scored,
    basis: typeof coverage.basis === 'string' ? coverage.basis : '',
  };
}

/**
 * True when the blob is a B4 report this component knows how to draw.
 *
 * Accepts null so callers need no separate check: an evaluation has no report
 * while it is running or after it has failed, and "nothing to draw" is the same
 * answer as "not a shape I recognise" at every call site.
 */
export function isEvaluationReport(
  blob: Record<string, unknown> | null,
): blob is EvaluationReport {
  if (blob === null) return false;
  const confusion = blob.confusion as { classes?: unknown; matrix?: unknown } | undefined;
  return (
    typeof blob.split === 'string' &&
    typeof blob.per_class === 'object' &&
    blob.per_class !== null &&
    Array.isArray(confusion?.classes) &&
    Array.isArray(confusion?.matrix)
  );
}
