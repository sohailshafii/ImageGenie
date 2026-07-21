import { request } from './client';
import type { AuthUser, Invite, LoginRequest, SignupRequest } from './types';

// Auth client for the FastAPI backend (server.md#api-layer). Replaces the former
// in-memory mock; the call surface is unchanged, so components did not move.
//
// The session lives in an httpOnly cookie the browser attaches automatically —
// there is no token to store, and nothing here reads it. Authorization shown in
// the UI is a hint only; the server is the boundary (NFR-7, web.md).

/** The account shape the API returns. */
interface MeResponse {
  email: string;
  role: AuthUser['role'];
}

/** Invites come back snake_case on the wire. */
interface InviteResponse {
  email: string;
  expires_at: string;
  accepted: boolean;
}

/** POST /auth/login — throws `invalid_credentials`, or `unverified`. */
export async function login(input: LoginRequest): Promise<AuthUser> {
  return request<MeResponse>('POST', '/auth/login', {
    email: input.email,
    password: input.password,
  });
}

/**
 * POST /auth/signup — invite-gated. Throws `invite_required` when the address has
 * no open invite, `email_taken` if it already has an account, or
 * `validation_error` for a too-short password.
 */
export async function signup(input: SignupRequest): Promise<void> {
  await request<void>('POST', '/auth/signup', {
    email: input.email,
    password: input.password,
  });
}

/** POST /auth/verify-email — throws `invalid_token` or `expired_token`. */
export async function verifyEmail(token: string): Promise<void> {
  await request<void>('POST', '/auth/verify-email', { token });
}

/**
 * POST /auth/verify-email/resend — always succeeds for a well-formed request.
 * The server answers identically whether or not the address exists, so this
 * cannot be used to discover which addresses have accounts.
 */
export async function resendVerification(email: string): Promise<void> {
  await request<void>('POST', '/auth/verify-email/resend', { email });
}

/** GET /auth/me — the current session's account, or throws `unauthorized`. */
export async function getMe(): Promise<AuthUser> {
  return request<MeResponse>('GET', '/auth/me');
}

/** POST /auth/logout — revokes the session server-side. */
export async function logout(): Promise<void> {
  await request<void>('POST', '/auth/logout');
}

/**
 * POST /auth/invites — admin-only: mint an email-bound invite. Idempotent per
 * address (re-inviting refreshes it). Throws `unauthorized`/`forbidden` for a
 * non-admin caller, or `validation_error` on a malformed address.
 */
export async function createInvite(email: string): Promise<Invite> {
  const invite = await request<InviteResponse>('POST', '/auth/invites', { email });
  return {
    email: invite.email,
    expiresAt: invite.expires_at,
    accepted: invite.accepted,
  };
}
