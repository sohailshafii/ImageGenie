import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { getMe, logout as logoutRequest } from '../api/auth';
import type { AuthUser } from '../api/types';

// Current session for the app. On load we rehydrate from getMe() (in the real
// backend the session lives in an httpOnly cookie JS can't read), so status starts
// 'loading' until that resolves — modeled on the ChatApp reference AuthContext.
type AuthStatus = 'loading' | 'authenticated' | 'unauthenticated';

interface AuthContextValue {
  status: AuthStatus;
  user: AuthUser | null;
  /** Seed the session after a successful login. */
  setUser: (user: AuthUser) => void;
  /** Clear the session server-side and locally. */
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUserState] = useState<AuthUser | null>(null);
  const [status, setStatus] = useState<AuthStatus>('loading');

  useEffect(() => {
    let active = true;
    getMe()
      .then((account) => {
        if (!active) return;
        setUserState(account);
        setStatus('authenticated');
      })
      .catch(() => {
        // 'unauthorized' (no session) or any failure: treat as logged out.
        if (!active) return;
        setUserState(null);
        setStatus('unauthenticated');
      });
    return () => {
      active = false;
    };
  }, []);

  function setUser(next: AuthUser) {
    setUserState(next);
    setStatus('authenticated');
  }

  async function logout() {
    try {
      await logoutRequest();
    } finally {
      // Clear locally regardless — never leave the UI authenticated after logout.
      setUserState(null);
      setStatus('unauthenticated');
    }
  }

  return (
    <AuthContext.Provider value={{ status, user, setUser, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components -- hook co-located with its provider
export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider');
  return ctx;
}
