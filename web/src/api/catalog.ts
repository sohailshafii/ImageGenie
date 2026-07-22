import { request } from './client';
import type {
  ClassName,
  DeadLetter,
  LabelSource,
  ModelArtifacts,
  ModelPage,
  ModelSummary,
  PipelineStage,
} from './types';

// Model catalog client for the FastAPI backend (server.md#api-layer).
//
// The model/label calls are real. The **dead-letter calls are still mocked** —
// the backend has no DLQ endpoints yet (`app/replay_dlq.py` is a CLI tool, not an
// API), so DeadLettersPage would have nothing to talk to. That is the last mock
// left in this app; see the marker below.

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
}): Promise<ModelPage> {
  const query = new URLSearchParams({
    page: String(params.page),
    page_size: String(params.pageSize),
  });
  if (params.className !== undefined) query.set('class_name', params.className);
  if (params.source !== undefined) query.set('source', params.source);

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

// ── Dead letters: STILL MOCKED ──────────────────────────────────────────────
// TODO(dlq-api): the backend has no dead-letter endpoints yet. Until it does,
// this serves fixed sample rows so the admin page renders; nothing here reflects
// the real DLQs. Replace both functions with `request(...)` calls when the
// endpoints land — the signatures are already the ones they'll use.
const MOCK_LATENCY_MS = 250;
const delay = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, MOCK_LATENCY_MS));

const deadLetters: DeadLetter[] = [
  {
    uid: 'a1b2c3d4e5f60001',
    stage: 'download',
    error: 'ReadTimeout fetching mesh',
    failedAt: '2026-07-19T17:40:00Z',
  },
  {
    uid: 'a1b2c3d4e5f60002',
    stage: 'download',
    error: 'SSLError on mirror connection',
    failedAt: '2026-07-19T17:41:12Z',
  },
  {
    uid: 'a1b2c3d4e5f60003',
    stage: 'convert',
    error: 'ValueError: mesh has no faces',
    failedAt: '2026-07-19T17:38:05Z',
  },
  {
    uid: 'a1b2c3d4e5f60004',
    stage: 'download',
    error: 'Memory limit exceeded (>2Gi)',
    failedAt: '2026-07-19T17:45:33Z',
  },
  {
    uid: 'a1b2c3d4e5f60005',
    stage: 'render',
    error: 'OSMesa context creation failed',
    failedAt: '2026-07-19T17:44:20Z',
  },
];

/** GET /dead-letters — admin-only. **Mock data** (see the TODO above). */
export async function listDeadLetters(): Promise<DeadLetter[]> {
  await delay();
  return [...deadLetters];
}

/** POST /dead-letters/{uid}/retry — admin-only. **Mock** (see the TODO above). */
export async function retryDeadLetter(uid: string, stage: PipelineStage): Promise<void> {
  await delay();
  const index = deadLetters.findIndex((item) => item.uid === uid && item.stage === stage);
  if (index !== -1) deadLetters.splice(index, 1);
}
