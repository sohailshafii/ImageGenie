import type { ApiErrorCode } from './types';

// A typed error the API layer throws so UI can branch on `code` (e.g. show the
// resend prompt on 'unverified') rather than string-matching messages. The real
// backend will return the same shape as a JSON error body; the mock throws it
// directly.
export class ApiError extends Error {
  readonly code: ApiErrorCode;

  constructor(code: ApiErrorCode, message?: string) {
    super(message ?? code);
    this.name = 'ApiError';
    this.code = code;
  }
}

/** Narrow an unknown caught value to an ApiError with a specific code. */
export function isApiError(error: unknown, code?: ApiErrorCode): error is ApiError {
  return error instanceof ApiError && (code === undefined || error.code === code);
}
