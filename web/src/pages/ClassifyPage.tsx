import { useRef, useState, type ChangeEvent, type FormEvent } from 'react';
import { predictUploadedMesh } from '../api/catalog';
import { ApiError, isApiError } from '../api/errors';
import type { Prediction } from '../api/types';
import { AppLayout } from '../components/AppLayout';

// "What is this?" for a mesh that is not in the catalog (web.md#predicting-a-class).
//
// Open to viewers as well as admins, unlike /upload — because nothing is stored.
// The file is rendered in memory, classified, and forgotten, so this is not a
// route around the admin gate on ingestion (FR-9). That is also why the two live
// on separate pages rather than as two buttons on one: they look alike and mean
// entirely different things, and the expensive mistake is uploading into the
// catalog when you only meant to ask a question.

// Mirrors PREDICTABLE_TYPES on the server: the three ingest formats plus PLY,
// which the pipeline produces but never accepts — so the normalized mesh a
// detail page hands out can be fed straight back in. Duplicated as the `accept`
// hint; the server re-validates, so a stale list here degrades to a clear 415.
const ACCEPTED_EXTENSIONS = ['.glb', '.stl', '.obj', '.ply'];

function describeFailure(caught: unknown): string {
  if (isApiError(caught, 'unavailable')) return 'No trained model yet — run a training job first.';
  if (isApiError(caught, 'network_error')) return 'Could not reach the server. Try again.';
  // unsupported_media_type, payload_too_large and the 422 for an unusable mesh
  // each name the offending format, the real limit, or what trimesh objected to
  // — all better than anything generic written here.
  if (caught instanceof ApiError && caught.message) return caught.message;
  return 'Could not classify that file.';
}

export function ClassifyPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function onChoose(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
    setPrediction(null);
    setError(null);
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (file === null) return;
    setBusy(true);
    setError(null);
    try {
      setPrediction(await predictUploadedMesh(file));
    } catch (caught) {
      setError(describeFailure(caught));
    } finally {
      setBusy(false);
    }
  }

  return (
    <AppLayout>
      <h1>Classify a mesh</h1>
      <p className="page-lead">
        Upload a mesh and the classifier will say what it thinks it is. Nothing is saved — the
        file is rendered, classified, and discarded. To add a model to the catalog instead, use
        Upload.
      </p>

      <form className="start-run-form" onSubmit={onSubmit}>
        <div className="field">
          <label htmlFor="classify-file">Mesh</label>
          <input
            id="classify-file"
            ref={inputRef}
            type="file"
            accept={ACCEPTED_EXTENSIONS.join(',')}
            onChange={onChoose}
          />
          <span className="field-hint">
            {ACCEPTED_EXTENSIONS.join(', ')} — including the normalized PLY you can download from
            a model's page.
          </span>
        </div>

        <button type="submit" className="btn-primary" disabled={busy || file === null}>
          {busy ? 'Rendering and classifying…' : 'Classify'}
        </button>
        {busy && (
          // Worth saying: this renders twelve views server-side before it can
          // answer, so a few seconds of apparent nothing is expected.
          <p className="form-note">Rendering twelve views — this takes a few seconds.</p>
        )}
        {error !== null && <p className="form-error">{error}</p>}
      </form>

      {prediction !== null && (
        <section className="run-section">
          <h2>{file?.name}</h2>
          <ol className="predict-list">
            {prediction.predictions.slice(0, 5).map((row) => (
              <li key={row.className}>
                <span className="predict-class">{row.className}</span>
                <span className="predict-bar" aria-hidden="true">
                  <span style={{ width: `${Math.round(row.probability * 100)}%` }} />
                </span>
                <span className="predict-probability">{(row.probability * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ol>
          <p className="form-note">from training run #{prediction.runId}</p>
        </section>
      )}
    </AppLayout>
  );
}
