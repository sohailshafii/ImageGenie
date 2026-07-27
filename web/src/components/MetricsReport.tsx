// The B4 dev-set report on a run's detail page: headline numbers, per-class
// precision/recall/F1, and the confusion matrix (ml.md#metrics).
//
// The blob is JSONB written by ml/metrics.py, so this narrows it defensively and
// the caller falls back to a generic key/value dump if the shape doesn't match —
// runs predating B4, or a future M7 report with a different layout, must still
// render something rather than crash the page.

import type { CSSProperties } from 'react';

import type { EvaluationReport } from './evaluationReport';

/** `0.8123` → `81.2%`; null → `—`. Null means *undefined*, not zero — a class the
 *  model never predicted has no precision, which is a different claim from 0%. */
function percent(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(1)}%`;
}

export function MetricsReport({ report }: { report: EvaluationReport }) {
  const { classes, matrix } = report.confusion;

  return (
    <div className="metrics-report">
      <dl className="kv-list">
        <div className="kv-row">
          <dt>split</dt>
          <dd className="kv-value">
            {report.split} ({report.sample_count} models)
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

      <h3 className="metrics-heading">Per class</h3>
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
