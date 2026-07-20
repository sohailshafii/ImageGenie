import { useEffect, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { verifyEmail } from '../api/auth';
import { isApiError } from '../api/errors';

// Consumes the one-time token from the emailed link (?token=…) on mount.
type VerifyStatus = 'verifying' | 'verified' | 'invalid' | 'missing';

export function VerifyEmailPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token');
  const [status, setStatus] = useState<VerifyStatus>(token ? 'verifying' : 'missing');
  // StrictMode double-invokes effects in dev; guard so we consume the token once.
  const consumed = useRef(false);

  useEffect(() => {
    if (!token || consumed.current) return;
    consumed.current = true;
    verifyEmail(token)
      .then(() => setStatus('verified'))
      .catch((caught) => setStatus(isApiError(caught) ? 'invalid' : 'invalid'));
  }, [token]);

  return (
    <section style={{ maxWidth: 360 }}>
      <h1>Email confirmation</h1>
      {status === 'verifying' && <p>Confirming your email…</p>}
      {status === 'verified' && (
        <p>
          Your email is confirmed. <Link to="/login">Sign in</Link>
        </p>
      )}
      {status === 'missing' && <p role="alert">This link is missing its token.</p>}
      {status === 'invalid' && (
        <p role="alert">
          This confirmation link is invalid or expired.{' '}
          <Link to="/verify-email/resend">Request a new one</Link>
        </p>
      )}
    </section>
  );
}
