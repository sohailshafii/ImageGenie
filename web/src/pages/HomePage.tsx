import { useAuth } from '../auth/AuthContext';

// Authenticated landing. A placeholder for now — the browse view (paginated
// thumbnail grid + inline confirm/correct) lands in a later chunk.
export function HomePage() {
  const { user, logout } = useAuth();

  return (
    <section>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1>ImageGenie — Labeling</h1>
        <div>
          <span>
            {user?.email} ({user?.role})
          </span>{' '}
          <button type="button" onClick={() => void logout()}>
            Sign out
          </button>
        </div>
      </header>
      <p>Browse and detail views land in the next chunks.</p>
    </section>
  );
}
