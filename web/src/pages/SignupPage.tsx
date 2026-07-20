import { useState, type FormEvent } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { signup } from '../api/auth';
import { isApiError } from '../api/errors';

// Invite-gated signup: the email must have an open invite (web.md). On success we
// don't log in — the account is unverified until the emailed link is clicked — so
// we show a "check your email" state with a resend fallback.
export function SignupPage() {
  const [searchParams] = useSearchParams();
  const [email, setEmail] = useState(searchParams.get('email') ?? '');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [done, setDone] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setPending(true);
    try {
      await signup({ email, password });
      setDone(true);
    } catch (caught) {
      if (isApiError(caught, 'invite_required')) {
        setError('That email doesn’t have an open invite. Ask an admin to invite you.');
      } else if (isApiError(caught, 'email_taken')) {
        setError('An account with that email already exists.');
      } else if (isApiError(caught, 'validation_error')) {
        setError('Password must be at least 8 characters.');
      } else {
        setError('Something went wrong. Please try again.');
      }
    } finally {
      setPending(false);
    }
  }

  if (done) {
    return (
      <section style={{ maxWidth: 360 }}>
        <h1>Check your email</h1>
        <p>
          We sent a confirmation link to <strong>{email}</strong>. Click it to activate your
          account.
        </p>
        <p>
          Didn’t get it?{' '}
          <Link to={`/verify-email/resend?email=${encodeURIComponent(email)}`}>Resend it</Link>
        </p>
      </section>
    );
  }

  return (
    <section style={{ maxWidth: 360 }}>
      <h1>Create your account</h1>
      <form onSubmit={onSubmit}>
        <label>
          Email
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            required
          />
        </label>
        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="new-password"
            minLength={8}
            required
          />
        </label>
        <button type="submit" disabled={pending}>
          {pending ? 'Creating…' : 'Create account'}
        </button>
      </form>
      {error && <p role="alert">{error}</p>}
      <p>
        Already have an account? <Link to="/login">Sign in</Link>
      </p>
    </section>
  );
}
