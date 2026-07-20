import { ApiError } from './errors';
import {
  accountsByEmail,
  clearSession,
  createAccount,
  emailByVerificationToken,
  findAccountById,
  getSessionAccountId,
  invitesByEmail,
  issueVerificationToken,
  setSessionAccountId,
} from './mockDb';
import type { AuthUser, Invite, LoginRequest, SignupRequest } from './types';

// Mock auth client — same call surface the real FastAPI client will expose, so
// components import these and never touch the backend shape directly (web.md:
// "one typed client module"). Flows mirror the ChatApp reference: invite-gated
// signup → email verification (+ resend) → login. Simulated latency makes the
// pending/disabled UI states real. NOT a security boundary (web.md).

const LATENCY_MS = 300;
const delay = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, LATENCY_MS));

const toAuthUser = (account: { id: string; email: string; role: AuthUser['role'] }): AuthUser => ({
  id: account.id,
  email: account.email,
  role: account.role,
});

const normalizeEmail = (email: string): string => email.trim().toLowerCase();

/** POST /auth/login — throws `invalid_credentials`, or `unverified` if the email isn't confirmed. */
export async function login(input: LoginRequest): Promise<AuthUser> {
  await delay();
  const account = accountsByEmail.get(normalizeEmail(input.email));
  if (!account || account.password !== input.password) {
    throw new ApiError('invalid_credentials');
  }
  if (!account.verified) throw new ApiError('unverified');
  setSessionAccountId(account.id);
  return toAuthUser(account);
}

/**
 * POST /auth/signup — invite-gated: the email must have an open invite. Creates
 * an unverified account and "sends" a verification email. Throws `invite_required`,
 * `email_taken`, or `validation_error`.
 */
export async function signup(input: SignupRequest): Promise<void> {
  await delay();
  const email = normalizeEmail(input.email);
  if (input.password.length < 8) throw new ApiError('validation_error', 'password too short');
  if (accountsByEmail.has(email)) throw new ApiError('email_taken');

  const invite = invitesByEmail.get(email);
  const open = invite && !invite.accepted && new Date(invite.expiresAt) > new Date();
  if (!open) throw new ApiError('invite_required');

  createAccount(email, input.password, 'user');
  invitesByEmail.set(email, { ...invite, accepted: true });
  issueVerificationToken(email);
}

/** POST /auth/verify-email — consumes a one-time token. Throws `invalid_token`. */
export async function verifyEmail(token: string): Promise<void> {
  await delay();
  const email = emailByVerificationToken.get(token);
  const account = email ? accountsByEmail.get(email) : undefined;
  if (!email || !account) throw new ApiError('invalid_token');
  account.verified = true;
  emailByVerificationToken.delete(token);
}

/**
 * POST /auth/verify-email/resend — re-issues a verification link. Responds
 * generically (never reveals whether the email has a pending account), so it
 * always resolves for a well-formed request.
 */
export async function resendVerification(email: string): Promise<void> {
  await delay();
  const account = accountsByEmail.get(normalizeEmail(email));
  if (account && !account.verified) issueVerificationToken(account.email);
  // else: intentionally silent — no account enumeration.
}

/** GET /auth/me — the current session's account, or throws `unauthorized`. */
export async function getMe(): Promise<AuthUser> {
  await delay();
  const accountId = getSessionAccountId();
  const account = accountId ? findAccountById(accountId) : undefined;
  if (!account) throw new ApiError('unauthorized');
  return toAuthUser(account);
}

/** POST /auth/logout — clears the session. */
export async function logout(): Promise<void> {
  await delay();
  clearSession();
}

/**
 * POST /auth/invites — admin-only: mint an email-bound invite (idempotent per
 * email; re-inviting refreshes it). Throws `unauthorized`/`forbidden` if the
 * caller isn't a signed-in admin, or `validation_error` on a bad email.
 */
export async function createInvite(email: string): Promise<Invite> {
  await delay();
  const accountId = getSessionAccountId();
  const caller = accountId ? findAccountById(accountId) : undefined;
  if (!caller) throw new ApiError('unauthorized');
  if (caller.role !== 'admin') throw new ApiError('forbidden');

  const normalized = normalizeEmail(email);
  if (!normalized.includes('@')) throw new ApiError('validation_error', 'invalid email');

  const invite: Invite = {
    email: normalized,
    expiresAt: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString(),
    accepted: false,
  };
  invitesByEmail.set(normalized, invite);
  console.info(`[mock email] invite ${normalized}: /signup?email=${encodeURIComponent(normalized)}`);
  return invite;
}
