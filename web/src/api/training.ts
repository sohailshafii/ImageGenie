import { download, request } from './client';
import type {
  TrainingMetricPoint,
  TrainingRunDetail,
  TrainingRunSummary,
  TrainingStatus,
} from './types';

// Training-dashboard client (FR-6, web.md#training-dashboard). Read-only: the
// training script writes these rows directly to the DB (ml/train.py), so there
// is nothing to POST here — this module only lists, drills into, and downloads
// the artifacts of runs.

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
