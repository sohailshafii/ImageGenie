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

/** True when the blob is a B4 report this component knows how to draw. */
export function isEvaluationReport(
  blob: Record<string, unknown>,
): blob is EvaluationReport {
  const confusion = blob.confusion as { classes?: unknown; matrix?: unknown } | undefined;
  return (
    typeof blob.split === 'string' &&
    typeof blob.per_class === 'object' &&
    blob.per_class !== null &&
    Array.isArray(confusion?.classes) &&
    Array.isArray(confusion?.matrix)
  );
}
