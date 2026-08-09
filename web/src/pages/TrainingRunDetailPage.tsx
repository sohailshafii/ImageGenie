import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  downloadTrainingWeights,
  getTrainingRun,
  getTrainingRunEvaluations,
  getTrainingRunMetrics,
  launchEvaluation,
} from '../api/training';
import { ApiError } from '../api/errors';
import {
  EVALUATION_DEV_SETS,
  type Evaluation,
  type TrainingMetricPoint,
  type TrainingRunDetail,
} from '../api/types';
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

// Past this, a value is collapsed behind its summary. `data_snapshot.held_out`
// records every uid a run held out — ~2,300 of them at full scale — and printed
// inline it buries every other field on the page under a wall of hex. The list
// is still bookkeeping worth reading (NFR-4), so it folds rather than truncates.
const COLLAPSE_OVER_CHARACTERS = 240;

/** A one-line stand-in for a value too long to show inline: how much is in there. */
function summarizeValue(value: unknown): string {
  if (Array.isArray(value)) return `${value.length.toLocaleString()} items`;
  if (value !== null && typeof value === 'object') {
    return Object.entries(value)
      .map(([key, entry]) => (Array.isArray(entry) ? `${key}: ${entry.length.toLocaleString()}` : key))
      .join(' · ');
  }
  return `${String(value).length.toLocaleString()} characters`;
}

function KeyValues({ record }: { record: Record<string, unknown> }) {
  return (
    <dl className="kv-list">
      {Object.entries(record).map(([key, value]) => {
        const formatted = formatValue(value);
        return (
          <div className="kv-row" key={key}>
            <dt>{key}</dt>
            <dd className="kv-value">
              {formatted.length <= COLLAPSE_OVER_CHARACTERS ? (
                formatted
              ) : (
                <details className="kv-details">
                  <summary>{summarizeValue(value)}</summary>
                  <div className="kv-overflow">{formatted}</div>
                </details>
              )}
            </dd>
          </div>
        );
      })}
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

  // Scoring spends GPU time, so it is admin-only like starting a run. The server
  // enforces it either way; hiding the control avoids offering an action that can
  // only 403.
  const canEvaluate = user?.role === 'admin';

  const [run, setRun] = useState<TrainingRunDetail | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetricPoint[] | null>(null);
  const [evaluations, setEvaluations] = useState<Evaluation[] | null>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'not-found'>('loading');
  const [devSet, setDevSet] = useState<string>(EVALUATION_DEV_SETS[0].name);
  const [evaluating, setEvaluating] = useState(false);
  const [evaluateError, setEvaluateError] = useState<string | null>(null);
  const [requested, setRequested] = useState(false);
  // Bumped to re-fetch the list. Nothing arrives for minutes — the row is written
  // when the container starts — so this is a manual refresh rather than a poll
  // that would spend a request a second discovering nothing has changed.
  const [reloadToken, setReloadToken] = useState(0);

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

  // Dev-set reports, likewise separate: most runs have none, and a failure to
  // load them should not cost the reader the config and the curve.
  useEffect(() => {
    let active = true;
    setEvaluations(null);
    getTrainingRunEvaluations(runId)
      .then((result) => {
        if (active) setEvaluations(result);
      })
      .catch(() => {
        if (active) setEvaluations([]);
      });
    return () => {
      active = false;
    };
  }, [runId, reloadToken]);

  async function onEvaluate() {
    setEvaluating(true);
    setEvaluateError(null);
    try {
      await launchEvaluation(runId, devSet);
      setRequested(true);
      setReloadToken((token) => token + 1);
    } catch (caught) {
      setEvaluateError(
        caught instanceof ApiError ? caught.message : 'Could not start the evaluation.',
      );
    } finally {
      setEvaluating(false);
    }
  }

  // The set the run *trained* on, to compare against what each evaluation was
  // *scored* on. The split is recomputed rather than stored, so these differing
  // is what turns a dev-set number into an indicative one.
  const runLabelHash =
    typeof run?.dataSnapshot?.label_hash === 'string' ? run.dataSnapshot.label_hash : null;

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

          {/* Held-out scores (M7), kept separate from the section above rather
              than merged into it: that one is the run's own report on `val`, a
              split it consulted every epoch, and these are scored afterwards on
              data it never saw. Same shape, different standing — presenting them
              together would invite reading the optimistic number as the honest
              one. */}
          {(canEvaluate || (evaluations !== null && evaluations.length > 0)) && (
            <section className="run-section">
              <h2>Held-out evaluation</h2>

              {/* Scoring is a Vertex job, not something a request can do: the API
                  image carries no torch. So this asks, and the report arrives
                  minutes later — which is why the row shows up as `running`
                  rather than the page pretending to wait. */}
              {canEvaluate && (
                <div className="evaluate-controls">
                  <label htmlFor="evaluate-dev-set">Score against</label>
                  <select
                    id="evaluate-dev-set"
                    value={devSet}
                    onChange={(event) => setDevSet(event.target.value)}
                  >
                    {EVALUATION_DEV_SETS.map((option) => (
                      <option key={option.name} value={option.name}>
                        {option.name}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={onEvaluate}
                    disabled={evaluating || !run.weightsUri}
                  >
                    {evaluating ? 'Starting…' : 'Evaluate'}
                  </button>
                  <span className="field-hint">
                    {EVALUATION_DEV_SETS.find((option) => option.name === devSet)?.hint}
                  </span>
                  {!run.weightsUri && (
                    <span className="field-hint">
                      This run saved no weights, so there is nothing to score.
                    </span>
                  )}
                  {evaluateError !== null && <p className="form-error">{evaluateError}</p>}
                  {requested && (
                    <p className="form-note">
                      Vertex accepted the job. It waits for a spot GPU and pulls the training
                      image first, so the evaluation appears here in roughly 12 minutes —{' '}
                      <button
                        type="button"
                        className="link-button"
                        onClick={() => setReloadToken((token) => token + 1)}
                      >
                        refresh
                      </button>
                      .
                    </p>
                  )}
                </div>
              )}

              {evaluations !== null && evaluations.length === 0 && (
                <p className="page-lead">
                  Not scored yet. Training reports on <code>val</code>, a split it consulted
                  every epoch; this is where a number from data the run never saw appears.
                </p>
              )}

              {evaluations?.map((evaluation) => (
                <div key={evaluation.id} className="evaluation-block">
                  <h3>
                    {evaluation.devSet}
                    <span className={`status-badge is-${evaluation.status}`}>
                      {evaluation.status}
                    </span>
                    <span className="form-note">
                      {evaluation.status === 'completed' ? 'scored' : 'started'}{' '}
                      {new Date(evaluation.createdAt).toLocaleString()}
                    </span>
                  </h3>
                  {runLabelHash !== null &&
                    evaluation.labelHash !== null &&
                    evaluation.labelHash !== runLabelHash && (
                      <p className="form-error">
                        The labeled set changed between training and scoring, so this split is
                        not the one the run held out. Treat the numbers as indicative.
                      </p>
                    )}
                  {evaluation.status === 'running' && (
                    <p className="page-lead">
                      Scoring — a forward pass over every model in the dev set, reading each
                      one's renders from the bucket. Refresh in a few minutes.
                    </p>
                  )}
                  {evaluation.status === 'failed' && (
                    // The reason, not a shrug: the alternative is reading Vertex's
                    // log stream to find out why something in the app didn't work.
                    <p className="form-error">{evaluation.error ?? 'The scoring job failed.'}</p>
                  )}
                  {isEvaluationReport(evaluation.report) ? (
                    <MetricsReport report={evaluation.report} />
                  ) : (
                    // An unfamiliar shape still renders rather than vanishing —
                    // a report is worth showing raw if it cannot be shown well.
                    evaluation.report !== null && <KeyValues record={evaluation.report} />
                  )}
                </div>
              ))}
            </section>
          )}

          <p className="run-timestamps">
            Started {new Date(run.startedAt).toLocaleString()}
            {run.finishedAt && <> · finished {new Date(run.finishedAt).toLocaleString()}</>}
          </p>
        </>
      )}
    </AppLayout>
  );
}
