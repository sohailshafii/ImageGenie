// Shared API types for the labeling frontend. Modeled on the ChatApp reference
// auth flow (login / invite-gated signup / email verification + resend). When the
// FastAPI backend lands, these move to a shared contract and the mock client
// (see ./auth.ts) is swapped for a real one — the components don't change.

/** Authorization role. Admins can correct labels + upload + invite (web.md). */
export type Role = 'user' | 'admin';

/** The authenticated account, as returned by the session (`getMe`). */
export interface AuthUser {
  id: string;
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
  | 'forbidden'; // authenticated but lacking the role (e.g. non-admin inviting)

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
