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
  | 'unauthorized' // no valid session
  | 'forbidden' // authenticated but lacking the role (e.g. non-admin inviting)
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
