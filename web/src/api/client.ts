import { ApiError } from './errors';
import type { ApiErrorCode } from './types';

// The single fetch wrapper every API module goes through (web.md: "one typed
// client module; no fetch calls scattered through components"). It owns three
// things the callers shouldn't repeat: cookie credentials, the CSRF header, and
// turning a non-2xx body into a typed ApiError.

// Same-origin path, proxied to the FastAPI app in dev (see vite.config.ts) and
// served behind the same host in production. Same-origin is not incidental — the
// CSRF defense (server.md#csrf) rests on it, and a cross-origin API would mean
// enabling CORS and weakening exactly that.
const API_BASE = '/api';

const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS']);

// Must match server/app/security.py.
const CSRF_COOKIE = 'imagegenie_csrf';
const CSRF_HEADER = 'X-CSRF-Token';

/** Read a cookie by name. The CSRF cookie is deliberately not httpOnly. */
function readCookie(name: string): string | null {
  const prefix = `${name}=`;
  for (const part of document.cookie.split('; ')) {
    if (part.startsWith(prefix)) return decodeURIComponent(part.slice(prefix.length));
  }
  return null;
}

// Codes the backend sends in `detail`. Anything outside this set is mapped by
// status instead, so an unrecognized body can never surface as a bogus code.
const KNOWN_CODES = new Set<string>([
  'invalid_credentials',
  'unverified',
  'email_taken',
  'invite_required',
  'invalid_token',
  'expired_token',
  'rate_limited',
  'validation_error',
  'unauthorized',
  'forbidden',
  'csrf_failure',
]);

const STATUS_CODES: Record<number, ApiErrorCode> = {
  400: 'validation_error',
  401: 'unauthorized',
  403: 'forbidden',
  409: 'email_taken',
  413: 'payload_too_large', // upload over the server's cap
  415: 'unsupported_media_type', // upload of a format the pipeline can't ingest
  422: 'validation_error', // FastAPI request-validation failures
  429: 'rate_limited',
};

/** Map an error response to a typed code, preferring the body over the status. */
async function toApiError(response: Response): Promise<ApiError> {
  let detail: unknown;
  try {
    detail = (await response.json())?.detail;
  } catch {
    detail = undefined; // empty or non-JSON body (a proxy error page, say)
  }
  if (typeof detail === 'string' && KNOWN_CODES.has(detail)) {
    return new ApiError(detail as ApiErrorCode, detail);
  }
  const code = STATUS_CODES[response.status] ?? 'server_error';
  // Keep a human-readable `detail` as the message even when it isn't a known
  // code. Upload rejections explain *why* ("unsupported format '.fbx' — upload
  // one of: ...") and that sentence is far more useful to show than the code.
  // The code still drives branching; only the display text comes from the server.
  const message = typeof detail === 'string' && detail ? detail : `HTTP ${response.status}`;
  return new ApiError(code, message);
}

/**
 * Issue a request against the API.
 *
 * `T` is the caller's expected response shape; 204s resolve to `undefined`, so
 * call those as `request<void>(...)`.
 */
export async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers['Content-Type'] = 'application/json';

  if (!SAFE_METHODS.has(method)) {
    // Echo the double-submit token. A cross-site page can cause the cookie to be
    // sent but cannot read it, so it cannot set this header (server.md#csrf).
    const csrf = readCookie(CSRF_COOKIE);
    if (csrf) headers[CSRF_HEADER] = csrf;
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      credentials: 'same-origin', // send the session cookie
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    // fetch only rejects on a transport failure, never on an HTTP error status.
    throw new ApiError('network_error', 'Could not reach the server');
  }

  if (!response.ok) throw await toApiError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/**
 * POST a file as `multipart/form-data`.
 *
 * Separate from `request` because the two disagree on one detail: `request`
 * serializes to JSON and sets `Content-Type`, whereas multipart's header carries
 * a generated boundary, so it must be left for the browser to set. Setting it by
 * hand produces a body the server cannot parse. Everything else — credentials,
 * the CSRF header, typed errors — is deliberately identical.
 */
export async function upload<T>(path: string, file: File, field = 'file'): Promise<T> {
  const headers: Record<string, string> = {};
  const csrf = readCookie(CSRF_COOKIE);
  if (csrf) headers[CSRF_HEADER] = csrf;

  const form = new FormData();
  form.append(field, file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers,
      credentials: 'same-origin',
      body: form,
    });
  } catch {
    throw new ApiError('network_error', 'Could not reach the server');
  }

  if (!response.ok) throw await toApiError(response);
  return (await response.json()) as T;
}
