import { AppLayout } from '../components/AppLayout';

// Authenticated landing. A placeholder for now — the browse view (paginated
// thumbnail grid + inline confirm/correct) lands in a later chunk.
export function HomePage() {
  return (
    <AppLayout>
      <h1>Labeling</h1>
      <p style={{ color: 'var(--text-muted)' }}>Browse and detail views land in the next chunks.</p>
    </AppLayout>
  );
}
