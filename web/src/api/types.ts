// Shared API types for the labeling frontend. Modeled on the ChatApp reference
// auth flow (login / invite-gated signup / email verification + resend). When the
// FastAPI backend lands, these move to a shared contract and the mock client
// (see ./auth.ts) is swapped for a real one — the components don't change.

/** Authorization role. Admins can correct labels + upload + invite (web.md). */
export type Role = 'user' | 'admin';

/** The authenticated account, as returned by the session (`getMe`). */
export interface AuthUser {
  email: string;
  role: Role;
}

/** Typed failure codes the auth flows branch on (mirrors ChatApp's error codes). */
export type ApiErrorCode =
  | 'invalid_credentials'
  | 'unverified' // account exists but email not confirmed → offer resend
  | 'email_taken'
  | 'invite_required' // signup is invite-only; no open invite for this email
  | 'invalid_token'
  | 'expired_token'
  | 'rate_limited'
  | 'validation_error'
  | 'unsupported_media_type' // upload: a format the pipeline can't ingest
  | 'payload_too_large' // upload: over the server's size cap
  | 'unauthorized' // no valid session
  | 'forbidden' // authenticated but lacking the role (e.g. non-admin inviting)
  | 'not_found' // no such resource — or, for a download, an artifact not produced yet
  | 'unavailable' // the server can't answer *yet* (e.g. nothing has trained) — not a fault
  | 'csrf_failure' // missing/mismatched CSRF token (server.md#csrf)
  | 'network_error' // the request never reached the server
  | 'server_error'; // an unrecognized non-2xx response

export interface LoginRequest {
  email: string;
  password: string;
}

export interface SignupRequest {
  email: string;
  password: string;
}

/** An operator/admin-minted, email-bound signup invitation (web.md invite flow). */
export interface Invite {
  email: string;
  expiresAt: string; // ISO 8601
  accepted: boolean;
  role: Role; // the role the account is created with — viewer (user) or admin
}

// ── Model catalog (labeling) ────────────────────────────────────────────────
/** The locked 12-class roster (ml/taxonomy.py). */
export const CLASS_NAMES = [
  'animal',
  'food',
  'car',
  'chair',
  'weapon',
  'electronics',
  'figure',
  'lamp',
  'aircraft',
  'building',
  'table',
  'plant',
] as const;

export type ClassName = (typeof CLASS_NAMES)[number];

/** Whether a label came from the weak-labeling rules or a human correction. */
export type LabelSource = 'weak' | 'manual';

/**
 * Browse ordering. `confidence` is least-confident-first — the review queue,
 * and the order the active-learning loop wants (ml.md). Models with no
 * confidence (manual labels, or none at all) sort last.
 */
export type ModelSort = 'uid' | 'confidence';

/**
 * A model in the browse grid: its current label plus metadata to aid labeling.
 *
 * The label fields are **nullable**: a model has no label until weak labeling or
 * a human assigns one, and the API reports that honestly rather than inventing a
 * class. The UI renders those as "unlabeled" — which is also the state every
 * model is in until the weak-label backfill runs.
 */
export interface ModelSummary {
  uid: string;
  title: string;
  tags: string[];
  className: ClassName | null;
  source: LabelSource | null;
  confidence: number | null; // 0..1, weak labels only; null once manual
  /**
   * First rendered view, for the grid. The server emits this without checking
   * the blob exists (a 24-card page would otherwise cost 24 round-trips to
   * object storage), so it can 404 for a model the pipeline hasn't rendered —
   * treat a load error as "no preview", not as a bug.
   */
  thumbnail: string | null;
}

/**
 * What the classifier makes of one model (server.md#predicting-a-class).
 *
 * `runId` is part of the answer rather than metadata: the prediction comes from
 * whichever run trained most recently, so it is only interpretable alongside
 * which model produced it.
 */
export interface Prediction {
  runId: number;
  /** The whole roster, best first — a near-tie is the case worth seeing. */
  predictions: { className: ClassName; probability: number }[];
}

/** A model's rendered views and mesh, for the detail view. */
export interface ModelArtifacts {
  uid: string;
  views: string[]; // in view order; empty until the render stage runs
  mesh: string | null; // normalized PLY, or null if not yet normalized
}

export interface ModelPage {
  items: ModelSummary[];
  total: number;
  page: number; // 1-based
  pageSize: number;
}

/** A preprocessing stage — used to attribute a dead-lettered failure. */
export type PipelineStage = 'download' | 'convert' | 'normalize' | 'render';

/** A job that failed a pipeline stage, as recorded by the worker that nacked it. */
export interface DeadLetter {
  id: number;
  uid: string;
  stage: PipelineStage;
  error: string;
  /** Pub/Sub's delivery count; null for failures recorded outside push delivery. */
  deliveryAttempt: number | null;
  failedAt: string; // ISO 8601
  /** Set once an admin re-enqueued it; such rows are hidden from the list. */
  replayedAt: string | null;
}

// ── Training dashboard (FR-6) ────────────────────────────────────────────────
/** Lifecycle of a training run (mirrors the server's TrainingStatus). */
export type TrainingStatus = 'running' | 'completed' | 'failed';

/** A training run as it appears in the dashboard list. */
export interface TrainingRunSummary {
  id: number;
  status: TrainingStatus;
  arch: string | null; // config summary for the row; null if config lacks it
  labelCount: number | null; // how many labels the run trained on (from data_snapshot)
  finalLoss: number | null; // training loss at the last logged step; null if no metrics
  // Top-1 val accuracy at the last step that measured it. Shown beside the loss
  // because on this skewed corpus a falling loss can hide a collapsed model.
  finalAccuracy: number | null;
  startedAt: string; // ISO 8601
  finishedAt: string | null;
}

/**
 * One run's full bookkeeping for the detail page — the three NFR-4 blobs
 * (config, data snapshot, dev-set metrics) verbatim plus status/timestamps.
 * The blobs are open records, hence `unknown`-valued: the config's keys grow
 * with the hyperparameters and the page renders them generically.
 */
export interface TrainingRunDetail {
  id: number;
  status: TrainingStatus;
  config: Record<string, unknown>;
  dataSnapshot: Record<string, unknown>;
  metrics: Record<string, unknown> | null; // dev-set eval; null until evaluated (B4/M7)
  weightsUri: string | null;
  notes: string | null;
  startedAt: string;
  finishedAt: string | null;
}

/** One point on a run's loss curve — the data behind the cost graph. */
export interface TrainingMetricPoint {
  step: number;
  loss: number;
  valLoss: number | null; // null on steps where validation was not evaluated
  valAccuracy: number | null; // top-1 on the val split, 0..1; null when not evaluated
}

/**
 * One run scored against one dev set (FR-7, M7).
 *
 * Distinct from `TrainingRunDetail.metrics`, which is the run's own end-of-run
 * report on `val` — a split it consulted every epoch. These are produced
 * afterwards by `make evaluate` against data the run never saw, and there can be
 * several: one per dev set, plus any re-scores.
 */
export interface Evaluation {
  id: number;
  devSet: string; // "test" today; a second, independently-annotated set later
  // A JSON object, always; narrowed to a drawable report with isEvaluationReport.
  report: Record<string, unknown>;
  /**
   * The labeled set as it stood when this was scored — *not* when the run
   * trained. The split is recomputed rather than stored, so a label added since
   * the run shifts it; a hash differing from the run's own means this number
   * describes a different held-out set than the one intended.
   */
  labelHash: string | null;
  createdAt: string;
}

/** Whether this deployment can launch a run, and what it would launch. */
export interface TrainingLaunchConfig {
  configured: boolean; // false where there is no Vertex to submit to (local dev)
  image: string | null; // the exact commit-tagged image that will run
  region: string | null;
  trainableCount: number; // models both labeled and rendered — the full-set size
}

/**
 * Optional per-run overrides. **Every field is optional on purpose**: an omitted
 * one is not sent, and the trainer keeps its own default (ml/train.py's
 * `Config`). So leaving a box empty means "whatever the trainer thinks is best",
 * not zero — which matters for `weightDecay` and `labelSmoothing`, where 0 is a
 * meaningful value an admin might deliberately choose.
 *
 * Architecture is deliberately absent: changing the backbone or head shape
 * changes the checkpoint's shape, so a run's weights only load back into the
 * architecture that produced them (server.md#api-layer).
 */
export interface TrainingHyperparameters {
  learningRate?: number;
  batchSize?: number;
  optimizer?: 'adam' | 'sgd';
  momentum?: number; // SGD only; ignored by Adam
  dropout?: number;
  weightDecay?: number;
  labelSmoothing?: number;
  classWeighting?: 'none' | 'balanced';
}

/**
 * The trainer's defaults, shown as placeholders so an empty box still tells the
 * admin what they are about to get.
 *
 * Mirrors `Config` in ml/train.py, and can drift from it — the API cannot serve
 * these (its image ships without the ml package, see server/app/roster.py), so
 * duplication is the available option. Kept honest by scope: these are display
 * hints only. A stale value here shows a misleading hint; it never changes what
 * the run actually does, because an untouched field is never sent.
 */
export const TRAINING_DEFAULTS = {
  learningRate: 0.0003,
  batchSize: 32,
  optimizer: 'adam',
  momentum: 0.9,
  dropout: 0.5,
  weightDecay: 0,
  labelSmoothing: 0,
  classWeighting: 'none',
} as const;

/** The accepted job. No run id: the row appears when the container starts. */
export interface TrainingLaunch {
  jobName: string;
  image: string;
  args: string[];
}
