import { request, upload } from './client';
import type {
  ClassName,
  DeadLetter,
  LabelSource,
  ModelArtifacts,
  ModelPage,
  ModelSort,
  ModelSummary,
  PipelineStage,
} from './types';

// Model catalog client for the FastAPI backend (server.md#api-layer).
// Every call here is real — no mocks remain in this app.

/** Wire shape of a model summary — snake_case, with nullable label fields. */
interface ModelSummaryResponse {
  uid: string;
  title: string;
  tags: string[];
  class_name: string | null;
  source: string | null;
  confidence: number | null;
  thumbnail: string | null;
}

interface ModelArtifactsResponse {
  uid: string;
  views: string[];
  mesh: string | null;
}

interface ModelPageResponse {
  items: ModelSummaryResponse[];
  total: number;
  page: number;
  page_size: number;
}

function toModelSummary(model: ModelSummaryResponse): ModelSummary {
  return {
    uid: model.uid,
    title: model.title,
    tags: model.tags,
    className: (model.class_name as ClassName | null) ?? null,
    source: (model.source as LabelSource | null) ?? null,
    confidence: model.confidence,
    thumbnail: model.thumbnail,
  };
}

/**
 * GET /models/{uid}/artifacts — the model's rendered views and normalized mesh.
 *
 * Separate from the summary on purpose: this checks each blob exists, so it
 * costs a round-trip per artifact and is only worth paying on the detail view.
 * The grid uses `ModelSummary.thumbnail` instead.
 */
export async function getModelArtifacts(uid: string): Promise<ModelArtifacts> {
  return request<ModelArtifactsResponse>(
    'GET',
    `/models/${encodeURIComponent(uid)}/artifacts`,
  );
}

/** GET /models — a page of models, optionally filtered by class and/or source. */
export async function listModels(params: {
  page: number;
  pageSize: number;
  className?: ClassName;
  source?: LabelSource;
  sort?: ModelSort;
}): Promise<ModelPage> {
  const query = new URLSearchParams({
    page: String(params.page),
    page_size: String(params.pageSize),
  });
  if (params.className !== undefined) query.set('class_name', params.className);
  if (params.source !== undefined) query.set('source', params.source);
  if (params.sort !== undefined) query.set('sort', params.sort);

  const page = await request<ModelPageResponse>('GET', `/models?${query}`);
  return {
    items: page.items.map(toModelSummary),
    total: page.total,
    page: page.page,
    pageSize: page.page_size,
  };
}

/** GET /models/{uid} — a single model (for the detail view). */
export async function getModel(uid: string): Promise<ModelSummary> {
  return toModelSummary(
    await request<ModelSummaryResponse>('GET', `/models/${encodeURIComponent(uid)}`),
  );
}

/**
 * PUT /models/{uid}/label — admin-only. Records the class as a **manual** label
 * (confirm = keep the class, correct = change it), which becomes the model's
 * current label. Returns the updated model.
 */
export async function setLabel(uid: string, className: ClassName): Promise<ModelSummary> {
  return toModelSummary(
    await request<ModelSummaryResponse>('PUT', `/models/${encodeURIComponent(uid)}/label`, {
      class_name: className,
    }),
  );
}

// ── Dead letters ────────────────────────────────────────────────────────────
interface DeadLetterResponse {
  id: number;
  uid: string;
  stage: PipelineStage;
  error: string;
  delivery_attempt: number | null;
  failed_at: string;
  replayed_at: string | null;
}

/** GET /dead-letters — admin-only. Outstanding failures, most recent first. */
export async function listDeadLetters(): Promise<DeadLetter[]> {
  const rows = await request<DeadLetterResponse[]>('GET', '/dead-letters');
  return rows.map((row) => ({
    id: row.id,
    uid: row.uid,
    stage: row.stage,
    error: row.error,
    deliveryAttempt: row.delivery_attempt,
    failedAt: row.failed_at,
    replayedAt: row.replayed_at,
  }));
}

/**
 * POST /dead-letters/{id}/retry — admin-only: re-enqueue the job on its stage
 * topic. Safe to press freely; every stage is idempotent (NFR-2), so replaying
 * something that already succeeded is a no-op rather than duplicate work.
 */
export async function retryDeadLetter(id: number): Promise<void> {
  await request<void>('POST', `/dead-letters/${id}/retry`);
}

/**
 * POST /models/upload — admin-only (FR-9). Sends the mesh and returns the model
 * it created, already queued for preprocessing.
 *
 * The upload replaces the *download* stage, so the model enters the pipeline at
 * convert and is unlabeled until a human labels it — there is no store metadata
 * to derive a weak label from.
 *
 * Rejections carry the server's own explanation in `ApiError.message`
 * (`unsupported_media_type`, `payload_too_large`, `validation_error` for an empty
 * or unreadable file), which is worth showing verbatim: it names the offending
 * format or the actual limit.
 */
export async function uploadModel(file: File): Promise<ModelSummary> {
  const created = await upload<ModelSummaryResponse>('/models/upload', file);
  return toModelSummary(created);
}

/**
 * DELETE /models/{uid} — admin-only soft delete (server.md#soft-delete). The
 * model's data is kept; it just disappears from browse until restored.
 * Idempotent, so a double click is harmless.
 */
export async function deleteModel(uid: string): Promise<void> {
  await request<void>('DELETE', `/models/${encodeURIComponent(uid)}`);
}

/** POST /models/{uid}/restore — admin-only: undo a soft delete. */
export async function restoreModel(uid: string): Promise<ModelSummary> {
  const restored = await request<ModelSummaryResponse>(
    'POST',
    `/models/${encodeURIComponent(uid)}/restore`,
  );
  return toModelSummary(restored);
}

/** GET /models/deleted — admin-only: the restore queue, most-recent first. */
export async function listDeletedModels(params: {
  page: number;
  pageSize: number;
}): Promise<ModelPage> {
  const query = new URLSearchParams({
    page: String(params.page),
    page_size: String(params.pageSize),
  });
  const body = await request<ModelPageResponse>('GET', `/models/deleted?${query}`);
  return {
    items: body.items.map(toModelSummary),
    total: body.total,
    page: body.page,
    pageSize: body.page_size,
  };
}
