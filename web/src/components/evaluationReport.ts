// Shape of the B4 metrics blob and the guard that recognises it, split out of
// MetricsReport.tsx: a .tsx exporting both a component and a helper breaks fast
// refresh, and this half is plain data anyway.

type ClassMetrics = {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  support: number;
};

export type EvaluationReport = {
  split: string;
  sample_count: number;
  accuracy: number | null;
  macro_precision: number | null;
  macro_recall: number | null;
  macro_f1: number | null;
  per_class: Record<string, ClassMetrics>;
  confusion: { classes: string[]; matrix: number[][] };
};

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
