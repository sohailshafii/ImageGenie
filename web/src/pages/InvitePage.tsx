import { useState, type FormEvent } from 'react';
import { createInvite } from '../api/auth';
import { isApiError } from '../api/errors';
import { AppLayout } from '../components/AppLayout';
import type { Invite } from '../api/types';

// Admin-only: mint email-bound invites (web.md invite flow). Reachable only via the
// admin-gated /invite route, but createInvite also re-checks the caller's role — the
// server is the real boundary. Lists the invites opened this session for reference.
export function InvitePage() {
  const [email, setEmail] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [invites, setInvites] = useState<Invite[]>([]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setPending(true);
    try {
      const invite = await createInvite(email);
      // De-dupe by email (re-inviting refreshes the same invite).
      setInvites((prev) => [invite, ...prev.filter((existing) => existing.email !== invite.email)]);
      setEmail('');
    } catch (caught) {
      if (isApiError(caught, 'validation_error')) {
        setError('Enter a valid email address.');
      } else if (isApiError(caught, 'forbidden')) {
        setError('Only admins can send invites.');
      } else {
        setError('Something went wrong. Please try again.');
      }
    } finally {
      setPending(false);
    }
  }

  return (
    <AppLayout>
      <h1>Invite a labeler</h1>
      <p className="page-lead">
        Signup is invite-only. Enter an email to open an invite — they can then create an account
        with that address and confirm it by email.
      </p>

      <form className="form form-inline" onSubmit={onSubmit}>
        <div className="field">
          <label htmlFor="invite-email">Email</label>
          <input
            id="invite-email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="off"
            required
          />
        </div>
        <button className="btn-primary" type="submit" disabled={pending}>
          {pending ? 'Inviting…' : 'Send invite'}
        </button>
      </form>

      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}

      {invites.length > 0 && (
        <>
          <h2 className="section-title">Invited this session</h2>
          <ul className="invite-list">
            {invites.map((invite) => (
              <li key={invite.email} className="invite-row">
                <span className="invite-email">{invite.email}</span>
                <span className="invite-expiry">
                  expires {new Date(invite.expiresAt).toLocaleDateString()}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </AppLayout>
  );
}
