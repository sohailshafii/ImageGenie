import { ApiError } from './errors';
import { getSessionAccountId, findAccountById } from './mockDb';
import {
  CLASS_NAMES,
  type ClassName,
  type DeadLetter,
  type LabelSource,
  type ModelPage,
  type ModelSummary,
  type PipelineStage,
} from './types';

// Mock model catalog — same call surface the real FastAPI client will expose, so
// the browse/detail views never touch the backend shape (web.md: "one typed client
// module"). Data is generated deterministically (no backend, no GCS reads from the
// browser yet); real renders/GLBs and DB-backed labels arrive with the API. NOT a
// security boundary.

const LATENCY_MS = 250;
const delay = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, LATENCY_MS));

// A handful of representative tags per class, to make the metadata realistic.
const CLASS_TAGS: Record<ClassName, string[]> = {
  animal: ['creature', 'wildlife', 'sculpt'],
  food: ['kitchen', 'meal', 'lowpoly'],
  car: ['vehicle', 'sedan', 'wheels'],
  chair: ['furniture', 'seat', 'interior'],
  weapon: ['military', 'blade', 'prop'],
  electronics: ['gadget', 'device', 'tech'],
  figure: ['character', 'humanoid', 'rigged'],
  lamp: ['light', 'furniture', 'interior'],
  aircraft: ['plane', 'jet', 'flight'],
  building: ['architecture', 'exterior', 'structure'],
  table: ['furniture', 'desk', 'interior'],
  plant: ['nature', 'foliage', 'pot'],
};

// Deterministic pseudo-hex uid so the same index always yields the same model.
function fakeUid(index: number): string {
  return (index * 2654435761).toString(16).padStart(8, '0').slice(0, 8) + 'a1b2c3d4';
}

// A stable "confidence" derived from the index (weak labels look uncertain).
function fakeConfidence(index: number): number {
  return Math.round((0.55 + ((index * 37) % 45) / 100) * 100) / 100;
}

function buildCatalog(count: number): ModelSummary[] {
  return Array.from({ length: count }, (_, index) => {
    const className = CLASS_NAMES[index % CLASS_NAMES.length];
    return {
      uid: fakeUid(index),
      title: `${className} model ${index + 1}`,
      tags: CLASS_TAGS[className],
      className,
      source: 'weak' as LabelSource,
      confidence: fakeConfidence(index),
    };
  });
}

// In-memory catalog + dead-letter list. Module-scoped so edits (confirm/correct,
// retry) persist across navigations within a session (a page reload resets them,
// like mockDb).
const catalog = buildCatalog(96);

const deadLetters: DeadLetter[] = [
  { uid: fakeUid(900), stage: 'download', error: 'ReadTimeout fetching mesh', failedAt: '2026-07-20T17:40:00Z' },
  { uid: fakeUid(901), stage: 'download', error: 'SSLError on mirror connection', failedAt: '2026-07-20T17:41:12Z' },
  { uid: fakeUid(902), stage: 'convert', error: 'ValueError: mesh has no faces', failedAt: '2026-07-20T17:38:05Z' },
  { uid: fakeUid(903), stage: 'download', error: 'Memory limit exceeded (>2Gi)', failedAt: '2026-07-20T17:45:33Z' },
  { uid: fakeUid(904), stage: 'render', error: 'OSMesa context creation failed', failedAt: '2026-07-20T17:44:20Z' },
];

function requireAdmin(): void {
  const accountId = getSessionAccountId();
  const caller = accountId ? findAccountById(accountId) : undefined;
  if (!caller) throw new ApiError('unauthorized');
  if (caller.role !== 'admin') throw new ApiError('forbidden');
}

/** GET /models — a page of models, optionally filtered by class and/or source. */
export async function listModels(params: {
  page: number;
  pageSize: number;
  className?: ClassName;
  source?: LabelSource;
}): Promise<ModelPage> {
  await delay();
  const filtered = catalog.filter(
    (model) =>
      (params.className === undefined || model.className === params.className) &&
      (params.source === undefined || model.source === params.source),
  );
  const start = (params.page - 1) * params.pageSize;
  return {
    items: filtered.slice(start, start + params.pageSize),
    total: filtered.length,
    page: params.page,
    pageSize: params.pageSize,
  };
}

/**
 * PUT /models/{uid}/label — admin-only. Set the model's class as a **manual**
 * label (confirm = keep the class, correct = change it); confidence becomes 1.
 * Mirrors the DB `label` row with source='manual' (server.md#database).
 */
export async function setLabel(uid: string, className: ClassName): Promise<ModelSummary> {
  await delay();
  requireAdmin();
  const model = catalog.find((candidate) => candidate.uid === uid);
  if (!model) throw new ApiError('validation_error', 'unknown model');
  model.className = className;
  model.source = 'manual';
  model.confidence = 1;
  return model;
}

/** GET /dead-letters — admin-only: models that failed a stage and were quarantined. */
export async function listDeadLetters(): Promise<DeadLetter[]> {
  await delay();
  requireAdmin();
  return [...deadLetters];
}

/**
 * POST /dead-letters/{uid}/retry — admin-only: re-enqueue a dead-lettered model
 * into its stage. In the real backend this republishes the DLQ message to the
 * stage topic; here it just drops it from the list (assume it re-runs).
 */
export async function retryDeadLetter(uid: string, stage: PipelineStage): Promise<void> {
  await delay();
  requireAdmin();
  const index = deadLetters.findIndex((item) => item.uid === uid && item.stage === stage);
  if (index === -1) throw new ApiError('validation_error', 'unknown dead-letter');
  deadLetters.splice(index, 1);
}
