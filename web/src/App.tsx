import { lazy, Suspense } from 'react';
import { Route, Routes } from 'react-router-dom';
import { ProtectedRoute } from './auth/ProtectedRoute';
import { BrowsePage } from './pages/BrowsePage';
import { DeadLettersPage } from './pages/DeadLettersPage';
import { DeletedModelsPage } from './pages/DeletedModelsPage';
import { InvitePage } from './pages/InvitePage';
import { TrainingRunDetailPage } from './pages/TrainingRunDetailPage';
import { StartTrainingRunPage } from './pages/StartTrainingRunPage';
import { TrainingRunsPage } from './pages/TrainingRunsPage';
import { ClassifyPage } from './pages/ClassifyPage';
import { UploadPage } from './pages/UploadPage';

// Code-split the detail view: it pulls in three.js (~600 KB), which no other
// route needs, so it downloads only when a model is opened.
const DetailPage = lazy(() =>
  import('./pages/DetailPage').then((module) => ({ default: module.DetailPage })),
);
import { LoginPage } from './pages/LoginPage';
import { ResendVerificationPage } from './pages/ResendVerificationPage';
import { SignupPage } from './pages/SignupPage';
import { VerifyEmailPage } from './pages/VerifyEmailPage';

export default function App() {
  return (
    <Routes>
      {/* Public auth routes */}
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />
      <Route path="/verify-email" element={<VerifyEmailPage />} />
      <Route path="/verify-email/resend" element={<ResendVerificationPage />} />

      {/* Everything else requires a session (web.md: login-gated) */}
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <BrowsePage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/models/:uid"
        element={
          <ProtectedRoute>
            <Suspense fallback={<p style={{ padding: '1.5rem' }}>Loading viewer…</p>}>
              <DetailPage />
            </Suspense>
          </ProtectedRoute>
        }
      />
      <Route
        path="/training"
        element={
          <ProtectedRoute>
            <TrainingRunsPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/training/new"
        element={
          <ProtectedRoute requireAdmin>
            <StartTrainingRunPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/training/:id"
        element={
          <ProtectedRoute>
            <TrainingRunDetailPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/invite"
        element={
          <ProtectedRoute requireAdmin>
            <InvitePage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/classify"
        element={
          <ProtectedRoute>
            <ClassifyPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/upload"
        element={
          <ProtectedRoute requireAdmin>
            <UploadPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/deleted"
        element={
          <ProtectedRoute requireAdmin>
            <DeletedModelsPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/dead-letters"
        element={
          <ProtectedRoute requireAdmin>
            <DeadLettersPage />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}
