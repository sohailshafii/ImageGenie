// The B4 dev-set report on a run's detail page: headline numbers, per-class
// precision/recall/F1, and the confusion matrix (ml.md#metrics).
//
// The blob is JSONB written by ml/metrics.py, so this narrows it defensively and
// the caller falls back to a generic key/value dump if the shape doesn't match —
// runs predating B4, or a future M7 report with a different layout, must still
// render something rather than crash the page.

import type { CSSProperties } from 'react';

import type { EvaluationReport, ReportCoverage } from './evaluationReport';
import { MARK_SCORED_FRACTION, THIN_SAMPLE_COUNT, readCoverage } from './evaluationReport';

/** `0.8123` → `81.2%`; null → `—`. Null means *undefined*, not zero — a class the
 *  model never predicted has no precision, which is a different claim from 0%. */
function percent(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(1)}%`;
}

/** What the denominator was, said in words — the three are not equally strong. */
function expectedPhrase(coverage: ReportCoverage): string {
  if (coverage.basis === 'selected_dev_set') return 'models selected for this dev set';
  if (coverage.basis === 'recorded_split_size') return 'models this run recorded in its split';
  return 'models this run held out';
}

export function MetricsReport({ report }: { report: EvaluationReport }) {
  const { classes, matrix } = report.confusion;
  const coverage = readCoverage(report);
  // Only a *shortfall* is worth a denominator. Coverage can legitimately exceed
  // 1.0 on a recomputed partition — the corpus has grown since the run recorded
  // its split size — and "1,200 of 1,173 models" reads as an error rather than
  // as the ordinary thing it is.
  const shortfall = coverage !== null && coverage.scored < coverage.expected ? coverage : null;
  const isThin = report.sample_count < THIN_SAMPLE_COUNT;

  return (
    <div className="metrics-report">
      <dl className="kv-list">
        <div className="kv-row">
          <dt>split</dt>
          <dd className="kv-value">
            {report.split} (
            {shortfall === null
              ? `${report.sample_count} models`
              : `${shortfall.scored} of ${shortfall.expected} models`}
            )
          </dd>
        </div>
        <div className="kv-row">
          <dt>accuracy</dt>
          <dd className="kv-value">{percent(report.accuracy)}</dd>
        </div>
        {/* Macro sits next to accuracy on purpose: the two disagree loudly on an
            imbalanced corpus, and that disagreement is the finding. */}
        <div className="kv-row">
          <dt>macro recall</dt>
          <dd className="kv-value">{percent(report.macro_recall)}</dd>
        </div>
        <div className="kv-row">
          <dt>macro precision</dt>
          <dd className="kv-value">{percent(report.macro_precision)}</dd>
        </div>
      </dl>

      {/* Stated where the numbers are read, not only in the job log. An
          evaluation over 4 of 45 models drew a full per-class table and
          confusion matrix with nothing anywhere saying what it rested on
          (ml.md#how-much-of-the-dev-set-a-report-covers). */}
      {shortfall !== null && shortfall.scored / shortfall.expected < MARK_SCORED_FRACTION && (
        <p className="form-error">
          This report covers {shortfall.scored} of the {shortfall.expected}{' '}
          {expectedPhrase(shortfall)} (
          {((shortfall.scored / shortfall.expected) * 100).toFixed(1)}%). Every number below
          describes that subset, not the split it names.
        </p>
      )}

      <h3 className="metrics-heading">Per class</h3>
      {/* A different failure from a shortfall, and invisible to coverage: a
          complete report over a small dev set is just as thin. */}
      {isThin && (
        <p className="page-lead metrics-note">
          {report.sample_count} models across {classes.length} classes leaves most classes with a
          handful of examples or none, so the rates below move by whole percentage points per
          model. Read them as a direction, not a measurement.
        </p>
      )}
      <div className="table-wrap">
        <table className="runs-table">
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(report.per_class).map(([className, metrics]) => (
              <tr key={className}>
                <td>{className}</td>
                <td>{percent(metrics.precision)}</td>
                <td>{percent(metrics.recall)}</td>
                <td>{percent(metrics.f1)}</td>
                <td>{metrics.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="metrics-heading">Confusion matrix</h3>
      <p className="page-lead metrics-note">
        Rows are the true class, columns the prediction — so the diagonal is correct and a heavy
        off-diagonal <em>column</em> is a class the model dumps everything into. Shading is
        relative to each row, so a small class reads as clearly as a large one.
      </p>
      <div className="table-wrap">
        <table className="confusion">
          <thead>
            <tr>
              <th />
              {classes.map((className) => (
                <th key={className} className="confusion-head">
                  {className}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIndex) => {
              // Normalise within the row: with a ~7.7:1 skew, shading by the
              // global maximum would leave every small class looking empty.
              const rowTotal = row.reduce((sum, count) => sum + count, 0);
              return (
                <tr key={classes[rowIndex]}>
                  <th className="confusion-head">{classes[rowIndex]}</th>
                  {row.map((count, columnIndex) => {
                    const share = rowTotal === 0 ? 0 : count / rowTotal;
                    const isDiagonal = rowIndex === columnIndex;
                    return (
                      <td
                        key={classes[columnIndex]}
                        className={`confusion-cell${isDiagonal ? ' is-diagonal' : ''}`}
                        style={{ '--cell-share': share } as CSSProperties}
                        title={`true ${classes[rowIndex]} → predicted ${classes[columnIndex]}: ${count}`}
                      >
                        {count === 0 ? '' : count}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
