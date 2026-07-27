import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { getTrainingLaunchConfig, launchTrainingRun } from '../api/training';
import { ApiError } from '../api/errors';
import type { TrainingLaunchConfig } from '../api/types';
import { AppLayout } from '../components/AppLayout';

// Start a training run (web.md#starting-a-training-run). Admin-only, and the one
// page in the app where clicking a button spends money on a GPU — so it shows
// what the run will cost *before* the button, not after.
//
// The defaults are recommendations, not limits: they start small so the
// expensive choice is a deliberate edit rather than the path of least
// resistance, but nothing here stops an admin who means it.

const DEFAULT_EPOCHS = 5;
const DEFAULT_LIMIT = 500;

// Measured, not guessed: a real spot-T4 run did 500 models × 1 epoch in 144s
// (~0.29s per model per epoch). Provisioning is the fixed cost — that same run
// waited ~12 min for a node and image pull before training started.
const SECONDS_PER_MODEL_EPOCH = 0.288;
const PROVISION_SECONDS = 12 * 60;
// Spot T4 + n1-standard-8, us-central1, roughly. Billed while the container
// runs, not while the job queues for capacity.
const DOLLARS_PER_HOUR = 0.16;

function formatDuration(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)}s`;
  const minutes = seconds / 60;
  if (minutes < 90) return `${Math.round(minutes)} min`;
  const hours = minutes / 60;
  return hours < 10 ? `${hours.toFixed(1)} hours` : `${Math.round(hours)} hours`;
}

export function StartTrainingRunPage() {
  const navigate = useNavigate();
  const [config, setConfig] = useState<TrainingLaunchConfig | null>(null);
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);
  const [limit, setLimit] = useState<number | null>(DEFAULT_LIMIT);
  const [notes, setNotes] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [launched, setLaunched] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getTrainingLaunchConfig()
      .then((result) => {
        if (active) setConfig(result);
      })
      .catch(() => {
        if (active) setError('Could not read the launch configuration.');
      });
    return () => {
      active = false;
    };
  }, []);

  const trainable = config?.trainableCount ?? 0;
  // A limit above the trainable count trains on everything, matching
  // ml/train.py's `_subsample`, so the estimate must not claim otherwise.
  const modelCount = limit === null ? trainable : Math.min(limit, trainable);
  const trainSeconds = modelCount * epochs * SECONDS_PER_MODEL_EPOCH;
  const dollars = (trainSeconds / 3600) * DOLLARS_PER_HOUR;

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = await launchTrainingRun({ epochs, limit, notes: notes || null });
      setLaunched(result.jobName);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not start the run.');
      setBusy(false);
    }
  }

  if (launched !== null) {
    return (
      <AppLayout>
        <h1>Training run requested</h1>
        <p className="page-lead">
          Vertex accepted the job. It waits for a spot GPU and pulls the training image before
          anything runs, so the run appears on the dashboard in roughly{' '}
          {formatDuration(PROVISION_SECONDS)} — not immediately.
        </p>
        <p className="form-note dlq-uid">{launched}</p>
        <p className="form-note">
          <button type="button" className="btn-primary" onClick={() => navigate('/training')}>
            Back to training runs
          </button>
        </p>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <p className="form-note" style={{ marginTop: 0 }}>
        <Link to="/training">← Back to training runs</Link>
      </p>
      <h1>Start a training run</h1>

      {config !== null && !config.configured ? (
        <p className="page-lead">
          This deployment can't launch training runs — it has no Vertex AI target configured.
          Runs can still be started from the command line with <code>make train-cloud</code>.
        </p>
      ) : (
        <form className="start-run-form" onSubmit={onSubmit}>
          <div className="field">
            <label htmlFor="run-models">Models</label>
            <input
              id="run-models"
              type="number"
              min={1}
              value={limit ?? ''}
              placeholder={`all ${trainable.toLocaleString()}`}
              onChange={(event) =>
                setLimit(event.target.value === '' ? null : Number(event.target.value))
              }
            />
            <span className="field-hint">
              of {trainable.toLocaleString()} labeled and rendered. Blank trains on all of them.
            </span>
          </div>

          <div className="field">
            <label htmlFor="run-epochs">Epochs</label>
            <input
              id="run-epochs"
              type="number"
              min={1}
              value={epochs}
              onChange={(event) => setEpochs(Number(event.target.value))}
            />
            <span className="field-hint">Passes over the training split.</span>
          </div>

          <div className="field">
            <label htmlFor="run-notes">Notes</label>
            <input
              id="run-notes"
              type="text"
              value={notes}
              placeholder="what this run is testing"
              onChange={(event) => setNotes(event.target.value)}
            />
            <span className="field-hint">Shown on the dashboard next to the run.</span>
          </div>

          {/* The disclaimer, made concrete: it moves as the inputs move, so the
              cost of "just train on everything" is visible before the click. */}
          <div className="cost-estimate">
            <h2>What this will do</h2>
            <dl className="kv-list">
              <div className="kv-row">
                <dt>trains on</dt>
                <dd className="kv-value">
                  {modelCount.toLocaleString()} models × {epochs}{' '}
                  {epochs === 1 ? 'epoch' : 'epochs'}
                </dd>
              </div>
              <div className="kv-row">
                <dt>GPU time</dt>
                <dd className="kv-value">
                  ~{formatDuration(trainSeconds)} (plus ~{formatDuration(PROVISION_SECONDS)}{' '}
                  waiting for a spot GPU and the image)
                </dd>
              </div>
              <div className="kv-row">
                <dt>rough cost</dt>
                <dd className="kv-value">
                  ~${dollars < 0.01 ? dollars.toFixed(3) : dollars.toFixed(2)} on a spot T4
                </dd>
              </div>
              <div className="kv-row">
                <dt>image</dt>
                <dd className="kv-value dlq-uid">{config?.image ?? '—'}</dd>
              </div>
            </dl>
            <p className="form-note">
              Estimated from a measured run, so treat it as an order of magnitude. Spot capacity
              is not guaranteed — the job may queue, and a preemption ends the run (each epoch is
              checkpointed, so the weights survive). The image above is pinned to a commit: if it
              predates a change you are testing, rebuild it with <code>make train-image</code>{' '}
              first.
            </p>
          </div>

          {error !== null && <p className="form-error">{error}</p>}

          <button type="submit" className="btn-primary" disabled={busy || config === null}>
            {busy ? 'Starting…' : 'Start run'}
          </button>
        </form>
      )}
    </AppLayout>
  );
}
