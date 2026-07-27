import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  downloadTrainingWeights,
  getTrainingRun,
  getTrainingRunMetrics,
} from '../api/training';
import type { TrainingMetricPoint, TrainingRunDetail } from '../api/types';
import { useAuth } from '../auth/AuthContext';
import { AppLayout } from '../components/AppLayout';
import { CostCurve } from '../components/CostCurve';
import { DownloadButton } from '../components/DownloadButton';
import { isEvaluationReport } from '../components/evaluationReport';
import { MetricsReport } from '../components/MetricsReport';

// Training-run detail (FR-6 / B3): the cost curve plus the run's bookkeeping —
// the config it ran with, the data snapshot it trained on, and (once evaluated)
// its dev-set metrics. Distinct from the per-model DetailPage. Read-only.

/** Render a config/snapshot value: arrays as CSV, objects as JSON, else as text. */
function formatValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(', ');
  if (value !== null && typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function KeyValues({ record }: { record: Record<string, unknown> }) {
  return (
    <dl className="kv-list">
      {Object.entries(record).map(([key, value]) => (
        <div className="kv-row" key={key}>
          <dt>{key}</dt>
          <dd className="kv-value">{formatValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

export function TrainingRunDetailPage() {
  const params = useParams();
  const runId = Number(params.id);
  const { user } = useAuth();
  // Weights are the one admin-only corner of an otherwise open dashboard
  // (NFR-6). The server enforces it; hiding the button just avoids offering a
  // viewer an action that can only fail.
  const canDownloadWeights = user?.role === 'admin';

  const [run, setRun] = useState<TrainingRunDetail | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetricPoint[] | null>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'not-found'>('loading');

  useEffect(() => {
    let active = true;
    setStatus('loading');
    getTrainingRun(runId)
      .then((result) => {
        if (!active) return;
        setRun(result);
        setStatus('ready');
      })
      .catch(() => {
        if (active) setStatus('not-found');
      });
    return () => {
      active = false;
    };
  }, [runId]);

  // Fetched separately from the run, mirroring the model/artifacts split: the
  // curve can be long, and the header/config shouldn't wait on it.
  useEffect(() => {
    let active = true;
    setMetrics(null);
    getTrainingRunMetrics(runId)
      .then((result) => {
        if (active) setMetrics(result);
      })
      .catch(() => {
        if (active) setMetrics([]);
      });
    return () => {
      active = false;
    };
  }, [runId]);

  return (
    <AppLayout>
      <p className="form-note" style={{ marginTop: 0 }}>
        <Link to="/training">← Back to training runs</Link>
      </p>

      {status === 'loading' && <p className="page-lead">Loading run…</p>}
      {status === 'not-found' && <p className="page-lead">That training run wasn’t found.</p>}

      {status === 'ready' && run && (
        <>
          <div className="run-header">
            <h1>Training run #{run.id}</h1>
            <span className={`status-badge is-${run.status}`}>{run.status}</span>
            {canDownloadWeights && run.weightsUri && (
              <DownloadButton
                label="Download weights"
                title={`The run's saved checkpoint (${run.weightsUri})`}
                missingLabel="Checkpoint missing"
                onDownload={() => downloadTrainingWeights(run.id)}
              />
            )}
          </div>
          {run.notes && <p className="page-lead">{run.notes}</p>}

          <section className="run-section">
            <h2>Cost curve</h2>
            {metrics === null ? (
              <p className="page-lead">Loading metrics…</p>
            ) : (
              <CostCurve points={metrics} />
            )}
          </section>

          <div className="run-columns">
            <section className="run-section">
              <h2>Configuration</h2>
              <KeyValues record={run.config} />
            </section>
            <section className="run-section">
              <h2>Data snapshot</h2>
              <KeyValues record={run.dataSnapshot} />
            </section>
          </div>

          <section className="run-section">
            <h2>Dev-set metrics</h2>
            {run.metrics === null ? (
              <p className="page-lead">
                Not evaluated yet — a run records this when it finishes (B4), so a run still
                training, or one that failed, has none.
              </p>
            ) : isEvaluationReport(run.metrics) ? (
              <MetricsReport report={run.metrics} />
            ) : (
              // An unrecognised blob still renders: runs predating B4, and
              // whatever shape M7's two-dev-set report turns out to have.
              <KeyValues record={run.metrics} />
            )}
          </section>

          <p className="run-timestamps">
            Started {new Date(run.startedAt).toLocaleString()}
            {run.finishedAt && <> · finished {new Date(run.finishedAt).toLocaleString()}</>}
          </p>
        </>
      )}
    </AppLayout>
  );
}
