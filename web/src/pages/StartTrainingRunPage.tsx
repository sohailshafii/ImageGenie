import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { getTrainingLaunchConfig, launchTrainingRun } from '../api/training';
import { ApiError } from '../api/errors';
import { TRAINING_DEFAULTS, type TrainingLaunchConfig } from '../api/types';
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

// The tunable knobs, in the order they appear. Held as strings so "" reads as
// "leave the trainer's default alone" — see TrainingHyperparameters.
type AdvancedField =
  | 'learningRate'
  | 'batchSize'
  | 'optimizer'
  | 'momentum'
  | 'dropout'
  | 'weightDecay'
  | 'labelSmoothing'
  | 'classWeighting';

const EMPTY_TUNING: Record<AdvancedField, string> = {
  learningRate: '',
  batchSize: '',
  optimizer: '',
  momentum: '',
  dropout: '',
  weightDecay: '',
  labelSmoothing: '',
  classWeighting: '',
};

/** "" -> undefined (not sent), otherwise the number. */
function asNumber(value: string): number | undefined {
  return value.trim() === '' ? undefined : Number(value);
}

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
  // Advanced knobs are held as *strings*, so "" can mean "not set" — distinct
  // from 0, which is a real value for weight decay and label smoothing. They are
  // converted at submit, and an unset one is never sent (api/training.ts).
  const [tuning, setTuning] = useState<Record<AdvancedField, string>>(EMPTY_TUNING);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const changedCount = Object.values(tuning).filter((value) => value !== '').length;

  function setField(field: AdvancedField, value: string) {
    setTuning((current) => ({ ...current, [field]: value }));
  }

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

  // Batch size is the one knob whose real cost is invisible at the point it is
  // typed: a model is `viewsPerModel` rendered images and ml/model.py folds them
  // into the batch, so the GPU is asked for 12x the number in the box. Runs
  // 18-20 died of that ~45s into a billed job. Both figures come from the API
  // (server/app/training_jobs.py), so this page states the arithmetic rather
  // than keeping its own idea of the ceiling.
  const viewsPerModel = config?.viewsPerModel ?? null;
  const maxBatchSize = config?.maxBatchSize ?? null;
  const resolvedBatchSize = asNumber(tuning.batchSize) ?? TRAINING_DEFAULTS.batchSize;
  const imagesPerForward =
    viewsPerModel === null || !Number.isFinite(resolvedBatchSize)
      ? null
      : resolvedBatchSize * viewsPerModel;
  const batchTooLarge =
    maxBatchSize !== null && Number.isFinite(resolvedBatchSize) && resolvedBatchSize > maxBatchSize;
  // Said once, shown wherever the admin is looking: in the field while tuning is
  // open, and next to the dead submit button when it is folded away — a disabled
  // button with its explanation hidden behind a disclosure is just a broken page.
  const batchTooLargeMessage =
    `Batch size ${resolvedBatchSize} is too large for the training GPU — it runs out of ` +
    `memory about a minute in, after the job has queued and billed. ` +
    `Use ${maxBatchSize} or fewer.`;

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
      const result = await launchTrainingRun({
        epochs,
        limit,
        notes: notes || null,
        hyperparameters: {
          learningRate: asNumber(tuning.learningRate),
          batchSize: asNumber(tuning.batchSize),
          optimizer: (tuning.optimizer || undefined) as 'adam' | 'sgd' | undefined,
          momentum: asNumber(tuning.momentum),
          dropout: asNumber(tuning.dropout),
          weightDecay: asNumber(tuning.weightDecay),
          labelSmoothing: asNumber(tuning.labelSmoothing),
          classWeighting: (tuning.classWeighting || undefined) as
            | 'none'
            | 'balanced'
            | undefined,
        },
      });
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

          {/* Tuning is folded away because the common run changes none of it.
              Every box empty === the trainer's own defaults, so the short form
              stays the honest default path rather than a stripped-down one. */}
          <div className="advanced-tuning">
            <button
              type="button"
              className="disclosure-toggle"
              aria-expanded={showAdvanced}
              onClick={() => setShowAdvanced((open) => !open)}
            >
              {showAdvanced ? '▾' : '▸'} Tuning
              {changedCount > 0 && <span className="pill">{changedCount} changed</span>}
            </button>

            {showAdvanced && (
              <>
                <p className="field-hint">
                  Leave a box empty to use the trainer's default, shown as the placeholder.
                  Whatever a run resolves to is recorded on its detail page.
                </p>

                <div className="field">
                  <label htmlFor="run-class-weighting">Class weighting</label>
                  <select
                    id="run-class-weighting"
                    value={tuning.classWeighting}
                    onChange={(event) => setField('classWeighting', event.target.value)}
                  >
                    <option value="">default ({TRAINING_DEFAULTS.classWeighting})</option>
                    <option value="none">none</option>
                    <option value="balanced">balanced</option>
                  </select>
                  <span className="field-hint">
                    The roster is skewed ~7.7:1 (weapon 2,134 vs aircraft 278).{' '}
                    <strong>balanced</strong> makes one rare-class mistake cost as much as
                    several common-class ones, so the tail stops being ignored — at some
                    precision on the big classes.
                  </span>
                </div>

                <div className="field">
                  <label htmlFor="run-learning-rate">Learning rate</label>
                  <input
                    id="run-learning-rate"
                    type="number"
                    step="0.0001"
                    min={0}
                    value={tuning.learningRate}
                    placeholder={String(TRAINING_DEFAULTS.learningRate)}
                    onChange={(event) => setField('learningRate', event.target.value)}
                  />
                  <span className="field-hint">Step size. Too high diverges, too low crawls.</span>
                </div>

                <div className="field">
                  <label htmlFor="run-batch-size">Batch size</label>
                  <input
                    id="run-batch-size"
                    type="number"
                    min={1}
                    max={maxBatchSize ?? undefined}
                    value={tuning.batchSize}
                    placeholder={String(TRAINING_DEFAULTS.batchSize)}
                    onChange={(event) => setField('batchSize', event.target.value)}
                  />
                  <span className="field-hint">
                    Models per weight update. Doubling it halves the updates per epoch, so the
                    same epoch count learns less.
                    {imagesPerForward !== null && viewsPerModel !== null && (
                      <>
                        {' '}
                        A model is {viewsPerModel} rendered views and they all go through the
                        network together, so this asks the GPU for{' '}
                        <strong>{imagesPerForward.toLocaleString()} images per pass</strong>
                        {maxBatchSize !== null && <> — {maxBatchSize} models is the most it fits</>}
                        .
                      </>
                    )}
                  </span>
                  {batchTooLarge && <span className="form-error">{batchTooLargeMessage}</span>}
                </div>

                <div className="field">
                  <label htmlFor="run-optimizer">Optimizer</label>
                  <select
                    id="run-optimizer"
                    value={tuning.optimizer}
                    onChange={(event) => setField('optimizer', event.target.value)}
                  >
                    <option value="">default ({TRAINING_DEFAULTS.optimizer})</option>
                    <option value="adam">adam</option>
                    <option value="sgd">sgd</option>
                  </select>
                </div>

                {tuning.optimizer === 'sgd' && (
                  <div className="field">
                    <label htmlFor="run-momentum">Momentum</label>
                    <input
                      id="run-momentum"
                      type="number"
                      step="0.05"
                      min={0}
                      max={1}
                      value={tuning.momentum}
                      placeholder={String(TRAINING_DEFAULTS.momentum)}
                      onChange={(event) => setField('momentum', event.target.value)}
                    />
                    <span className="field-hint">SGD only — Adam ignores it.</span>
                  </div>
                )}

                <div className="field">
                  <label htmlFor="run-dropout">Dropout</label>
                  <input
                    id="run-dropout"
                    type="number"
                    step="0.05"
                    min={0}
                    max={0.95}
                    value={tuning.dropout}
                    placeholder={String(TRAINING_DEFAULTS.dropout)}
                    onChange={(event) => setField('dropout', event.target.value)}
                  />
                  <span className="field-hint">
                    Fraction of head units dropped each step, to curb overfitting. Must be under 1.
                  </span>
                </div>

                <div className="field">
                  <label htmlFor="run-weight-decay">Weight decay</label>
                  <input
                    id="run-weight-decay"
                    type="number"
                    step="0.0001"
                    min={0}
                    value={tuning.weightDecay}
                    placeholder={String(TRAINING_DEFAULTS.weightDecay)}
                    onChange={(event) => setField('weightDecay', event.target.value)}
                  />
                  <span className="field-hint">Penalty on large weights. 0 disables it.</span>
                </div>

                <div className="field">
                  <label htmlFor="run-label-smoothing">Label smoothing</label>
                  <input
                    id="run-label-smoothing"
                    type="number"
                    step="0.05"
                    min={0}
                    max={0.95}
                    value={tuning.labelSmoothing}
                    placeholder={String(TRAINING_DEFAULTS.labelSmoothing)}
                    onChange={(event) => setField('labelSmoothing', event.target.value)}
                  />
                  <span className="field-hint">
                    Softens the target so the model is less certain — useful against weak-label
                    noise. 0 disables it.
                  </span>
                </div>

                {changedCount > 0 && (
                  <p className="form-note">
                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={() => setTuning(EMPTY_TUNING)}
                    >
                      Reset to defaults
                    </button>
                  </p>
                )}
              </>
            )}
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
          {batchTooLarge && !showAdvanced && <p className="form-error">{batchTooLargeMessage}</p>}

          <button
            type="submit"
            className="btn-primary"
            disabled={busy || config === null || batchTooLarge}
          >
            {busy ? 'Starting…' : 'Start run'}
          </button>
        </form>
      )}
    </AppLayout>
  );
}
