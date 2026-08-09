import { download, request } from './client';
import type {
  Evaluation,
  TrainingHyperparameters,
  TrainingLaunch,
  TrainingLaunchConfig,
  TrainingMetricPoint,
  TrainingRunDetail,
  TrainingRunSummary,
  TrainingStatus,
} from './types';

// Training-dashboard client (FR-6, web.md#training-dashboard). Read-only: the
// training script writes these rows directly to the DB (ml/train.py), so there
// is nothing to POST here for *recording* a run — this module lists, drills into,
// and downloads the artifacts of runs. It can, however, ASK for one to be started
// (web.md#starting-a-training-run): the job writes its own row when it begins.

/** Wire shapes — snake_case, mirroring the FastAPI response models. */
interface TrainingRunSummaryResponse {
  id: number;
  status: TrainingStatus;
  arch: string | null;
  label_count: number | null;
  final_loss: number | null;
  final_accuracy: number | null;
  started_at: string;
  finished_at: string | null;
}

interface TrainingRunDetailResponse {
  id: number;
  status: TrainingStatus;
  config: Record<string, unknown>;
  data_snapshot: Record<string, unknown>;
  metrics: Record<string, unknown> | null;
  weights_uri: string | null;
  notes: string | null;
  started_at: string;
  finished_at: string | null;
}

interface TrainingMetricResponse {
  step: number;
  loss: number;
  val_loss: number | null;
  val_accuracy: number | null;
}

interface EvaluationResponse {
  id: number;
  dev_set: string;
  status: TrainingStatus;
  report: Record<string, unknown> | null;
  error: string | null;
  label_hash: string | null;
  created_at: string;
}

function toSummary(run: TrainingRunSummaryResponse): TrainingRunSummary {
  return {
    id: run.id,
    status: run.status,
    arch: run.arch,
    labelCount: run.label_count,
    finalLoss: run.final_loss,
    finalAccuracy: run.final_accuracy,
    startedAt: run.started_at,
    finishedAt: run.finished_at,
  };
}

/** GET /training-runs — all runs, newest first. */
export async function listTrainingRuns(): Promise<TrainingRunSummary[]> {
  const rows = await request<TrainingRunSummaryResponse[]>('GET', '/training-runs');
  return rows.map(toSummary);
}

/** GET /training-runs/{id} — one run's config / data-snapshot / metrics blobs. */
export async function getTrainingRun(id: number): Promise<TrainingRunDetail> {
  const run = await request<TrainingRunDetailResponse>('GET', `/training-runs/${id}`);
  return {
    id: run.id,
    status: run.status,
    config: run.config,
    dataSnapshot: run.data_snapshot,
    metrics: run.metrics,
    weightsUri: run.weights_uri,
    notes: run.notes,
    startedAt: run.started_at,
    finishedAt: run.finished_at,
  };
}

/** GET /training-runs/{id}/metrics — the loss curve, in step order. */
export async function getTrainingRunMetrics(id: number): Promise<TrainingMetricPoint[]> {
  const rows = await request<TrainingMetricResponse[]>(
    'GET',
    `/training-runs/${id}/metrics`,
  );
  return rows.map((point) => ({
    step: point.step,
    loss: point.loss,
    valLoss: point.val_loss,
    valAccuracy: point.val_accuracy,
  }));
}

/** GET /training-runs/{id}/evaluations — dev-set reports, newest first (M7). */
export async function getTrainingRunEvaluations(id: number): Promise<Evaluation[]> {
  const rows = await request<EvaluationResponse[]>(
    'GET',
    `/training-runs/${id}/evaluations`,
  );
  return rows.map((row) => ({
    id: row.id,
    devSet: row.dev_set,
    status: row.status,
    report: row.report,
    error: row.error,
    labelHash: row.label_hash,
    createdAt: row.created_at,
  }));
}

/**
 * POST /training-runs/{id}/evaluations — **admin-only**: score a finished run.
 *
 * Resolves once Vertex has *accepted* the job (202), not once a report exists.
 * The `evaluation` row is written by the job itself when the container starts,
 * minutes later, and appears as `running` from that point — so the caller should
 * tell the user to come back rather than wait on a number.
 */
export async function launchEvaluation(
  id: number,
  devSet: string,
): Promise<TrainingLaunch> {
  const body = await request<TrainingLaunchResponse>(
    'POST',
    `/training-runs/${id}/evaluations`,
    { dev_set: devSet },
  );
  return { jobName: body.job_name, image: body.image, args: body.args };
}

/**
 * GET /training-runs/{id}/weights — **admin-only**: the run's saved `.pt`
 * checkpoint.
 *
 * The one part of the dashboard a viewer can't reach (a trained model is what
 * NFR-6 calls non-redistributable), so a non-admin caller gets a `forbidden`
 * ApiError. A run with no `weightsUri` has nothing to download and 404s.
 */
export async function downloadTrainingWeights(id: number): Promise<void> {
  await download(`/training-runs/${id}/weights`, `imagegenie-run-${id}.pt`);
}

/** What the launch form needs before anything is spent. */
interface TrainingLaunchConfigResponse {
  configured: boolean;
  image: string | null;
  region: string | null;
  trainable_count: number;
  views_per_model: number;
  max_batch_size: number;
}

interface TrainingLaunchResponse {
  job_name: string;
  image: string;
  args: string[];
}

/** GET /training-launch — **admin-only**: is a launch possible here, and on what. */
export async function getTrainingLaunchConfig(): Promise<TrainingLaunchConfig> {
  const body = await request<TrainingLaunchConfigResponse>('GET', '/training-launch');
  return {
    configured: body.configured,
    image: body.image,
    region: body.region,
    trainableCount: body.trainable_count,
    viewsPerModel: body.views_per_model,
    maxBatchSize: body.max_batch_size,
  };
}

/**
 * POST /training-runs — **admin-only**: start a Vertex spot-GPU run.
 *
 * Resolves once Vertex has *accepted* the job (202), not once a run exists: the
 * `training_run` row is written by the training container minutes later, after
 * the GPU is provisioned and the image pulled. So the caller should send the
 * user to the run list to wait, not expect a run id back.
 */
export async function launchTrainingRun(params: {
  epochs: number;
  limit: number | null;
  notes: string | null;
  hyperparameters?: TrainingHyperparameters;
}): Promise<TrainingLaunch> {
  const { hyperparameters = {} } = params;
  // Only *set* knobs are sent. An absent key leaves ml/train.py's Config default
  // in charge, so the request records what the admin chose rather than
  // re-asserting a copy of the defaults that could drift from the trainer's.
  const overrides: Record<string, number | string> = {};
  for (const [name, value] of Object.entries(hyperparameters)) {
    if (value !== undefined) overrides[HYPERPARAMETER_KEYS[name as HyperparameterName]] = value;
  }

  const body = await request<TrainingLaunchResponse>('POST', '/training-runs', {
    epochs: params.epochs,
    limit: params.limit,
    notes: params.notes,
    ...overrides,
  });
  return { jobName: body.job_name, image: body.image, args: body.args };
}

type HyperparameterName = keyof TrainingHyperparameters;

/** camelCase field -> the snake_case name `TrainingLaunchIn` expects. */
const HYPERPARAMETER_KEYS: Record<HyperparameterName, string> = {
  learningRate: 'learning_rate',
  batchSize: 'batch_size',
  optimizer: 'optimizer',
  momentum: 'momentum',
  dropout: 'dropout',
  weightDecay: 'weight_decay',
  labelSmoothing: 'label_smoothing',
  classWeighting: 'class_weighting',
};
