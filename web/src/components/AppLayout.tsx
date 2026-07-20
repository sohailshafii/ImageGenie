import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';

// Shared shell for authenticated pages: the app header (brand, admin nav, user +
// sign out) above the page content. Admin-only links are hidden for normal users
// — a UX affordance; the server API is the real authorization boundary (web.md).
export function AppLayout({ children }: { children: ReactNode }) {
  const { user, logout } = useAuth();

  return (
    <>
      <header className="app-header">
        <Link to="/" className="app-logo">
          <span aria-hidden="true">🧞</span> ImageGenie
        </Link>
        <nav className="app-user">
          {user?.role === 'admin' && <Link to="/invite">Invite</Link>}
          <span>
            {user?.email} <span className="pill">{user?.role}</span>
          </span>
          <button className="btn-secondary" type="button" onClick={() => void logout()}>
            Sign out
          </button>
        </nav>
      </header>
      <div className="app-content">{children}</div>
    </>
  );
}
