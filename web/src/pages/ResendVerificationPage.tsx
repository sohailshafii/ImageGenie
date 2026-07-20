import { useState, type FormEvent } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { resendVerification } from '../api/auth';

// Resend the confirmation email. The API responds generically (never reveals
// whether the address has a pending account), so the UI always shows the same
// success state — no account enumeration.
export function ResendVerificationPage() {
  const [searchParams] = useSearchParams();
  const [email, setEmail] = useState(searchParams.get('email') ?? '');
  const [sent, setSent] = useState(false);
  const [pending, setPending] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setPending(true);
    try {
      await resendVerification(email);
      setSent(true);
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="auth-layout">
      <section className="card">
        <p className="brand">
          <span className="brand-mark">🧞</span> ImageGenie
        </p>
        <h1>Resend confirmation</h1>
        {sent ? (
          <>
            <p className="form-success">
              If <strong>{email}</strong> has an unconfirmed account, a new confirmation link is on
              its way.
            </p>
            <p className="form-note">
              <Link to="/login">Back to sign in</Link>
            </p>
          </>
        ) : (
          <form className="form" onSubmit={onSubmit}>
            <div className="field">
              <label htmlFor="email">Email</label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
                required
              />
            </div>
            <button className="btn-primary" type="submit" disabled={pending}>
              {pending ? 'Sending…' : 'Resend confirmation'}
            </button>
          </form>
        )}
      </section>
    </div>
  );
}
