import { useState, type FormEvent } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { login } from '../api/auth';
import { isApiError } from '../api/errors';
import { useAuth } from '../auth/AuthContext';

// Login: email + password → session. On `unverified` (account exists but email
// not confirmed) we surface a resend link instead of a dead-end error, mirroring
// the ChatApp reference flow.
export function LoginPage() {
  const { setUser } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const returnTo = (location.state as { from?: string } | null)?.from ?? '/';

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [unverified, setUnverified] = useState(false);
  const [pending, setPending] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setUnverified(false);
    setPending(true);
    try {
      const user = await login({ email, password });
      setUser(user);
      navigate(returnTo, { replace: true });
    } catch (caught) {
      if (isApiError(caught, 'unverified')) {
        setUnverified(true);
      } else if (isApiError(caught, 'invalid_credentials')) {
        setError('Incorrect email or password.');
      } else {
        setError('Something went wrong. Please try again.');
      }
    } finally {
      setPending(false);
    }
  }

  return (
    <section style={{ maxWidth: 360 }}>
      <h1>Sign in</h1>
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
            autoComplete="current-password"
            required
          />
        </label>
        <button type="submit" disabled={pending}>
          {pending ? 'Signing in…' : 'Sign in'}
        </button>
      </form>

      {error && <p role="alert">{error}</p>}
      {unverified && (
        <p role="alert">
          Your email isn’t confirmed yet.{' '}
          <Link to={`/verify-email/resend?email=${encodeURIComponent(email)}`}>
            Resend confirmation
          </Link>
        </p>
      )}

      <p>
        Have an invite? <Link to="/signup">Create your account</Link>
      </p>
    </section>
  );
}
