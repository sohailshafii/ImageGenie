import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy the API through the dev server so the browser sees ONE origin.
      // That isn't just convenience: the auth cookies are SameSite=Lax and the
      // CSRF defense rests on the same-origin policy (server.md#csrf), so a
      // cross-origin dev setup would need CORS and would weaken exactly the
      // thing being tested. The backend has no /api prefix, so strip it here.
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
