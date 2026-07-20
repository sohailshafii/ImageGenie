import { Navigate, useLocation } from 'react-router-dom';
import type { ReactNode } from 'react';
import { useAuth } from './AuthContext';

// Route guard: gate children behind login, and optionally behind the admin role.
// This is a UX gate only — the server API is the real authorization boundary
// (web.md / NFR-7); a non-admin who forges their way here still can't mutate
// anything the backend rejects.
export function ProtectedRoute({
  children,
  requireAdmin = false,
}: {
  children: ReactNode;
  requireAdmin?: boolean;
}) {
  const { status, user } = useAuth();
  const location = useLocation();

  if (status === 'loading') return <p style={{ padding: '1.5rem' }}>Loading…</p>;

  if (status === 'unauthenticated' || !user) {
    // Remember where they were headed so login can send them back.
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  if (requireAdmin && user.role !== 'admin') {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
}
