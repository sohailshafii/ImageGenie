import type { Invite, Role } from './types';

// In-memory stand-in for the (not-yet-built) FastAPI backend + Postgres. It holds
// accounts, email-bound invites, and pending verification tokens, and persists the
// "session" to localStorage so a reload rehydrates like the real httpOnly cookie
// would. Everything here is a simulation for building the UI — NOT a security
// boundary (web.md). Swapping ./auth.ts for a real client retires this file.

export interface MockAccount {
  id: string;
  email: string;
  password: string;
  role: Role;
  verified: boolean;
}

const SESSION_KEY = 'imagegenie.session';
let idCounter = 0;
const nextId = (prefix: string): string => `${prefix}_${(idCounter += 1)}`;

export const accountsByEmail = new Map<string, MockAccount>();
export const invitesByEmail = new Map<string, Invite>();
export const emailByVerificationToken = new Map<string, string>();

function seed(): void {
  // Demo admin so you can log in immediately and exercise the invite flow.
  accountsByEmail.set('admin@imagegenie.dev', {
    id: nextId('acct'),
    email: 'admin@imagegenie.dev',
    password: 'genie-admin',
    role: 'admin',
    verified: true,
  });
  // One open invite so the invite-gated signup flow is demoable out of the box.
  const inYearDays = new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString();
  invitesByEmail.set('labeler@imagegenie.dev', {
    email: 'labeler@imagegenie.dev',
    expiresAt: inYearDays,
    accepted: false,
  });
}
seed();

export function createAccount(email: string, password: string, role: Role): MockAccount {
  const account: MockAccount = { id: nextId('acct'), email, password, role, verified: false };
  accountsByEmail.set(email, account);
  return account;
}

/** Mint a verification token for an email and "send" it (logged for the demo). */
export function issueVerificationToken(email: string): string {
  const token = nextId('verify');
  emailByVerificationToken.set(token, email);
  // Stand-in for the verification email: the link the recipient would click.
  console.info(`[mock email] verify ${email}: /verify-email?token=${token}`);
  return token;
}

export function getSessionAccountId(): string | null {
  return localStorage.getItem(SESSION_KEY);
}

export function setSessionAccountId(accountId: string): void {
  localStorage.setItem(SESSION_KEY, accountId);
}

export function clearSession(): void {
  localStorage.removeItem(SESSION_KEY);
}

export function findAccountById(accountId: string): MockAccount | undefined {
  for (const account of accountsByEmail.values()) {
    if (account.id === accountId) return account;
  }
  return undefined;
}
