import { useAuth } from '../auth/AuthContext';

// Authenticated landing. A placeholder for now — the browse view (paginated
// thumbnail grid + inline confirm/correct) lands in a later chunk.
export function HomePage() {
  const { user, logout } = useAuth();

  return (
    <>
      <header className="app-header">
        <span className="app-logo">
          <span aria-hidden="true">🧞</span> ImageGenie
        </span>
        <div className="app-user">
          <span>
            {user?.email} <span className="pill">{user?.role}</span>
          </span>
          <button className="btn-secondary" type="button" onClick={() => void logout()}>
            Sign out
          </button>
        </div>
      </header>
      <div className="app-content">
        <h1>Labeling</h1>
        <p style={{ color: 'var(--text-muted)' }}>
          Browse and detail views land in the next chunks.
        </p>
      </div>
    </>
  );
}
