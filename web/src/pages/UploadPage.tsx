import { useRef, useState, type ChangeEvent, type DragEvent, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { uploadModel } from '../api/catalog';
import { ApiError, isApiError } from '../api/errors';
import { AppLayout } from '../components/AppLayout';
import type { ModelSummary } from '../api/types';

// Admin-only: feed extra meshes into the pipeline (FR-9, web.md#data-upload).
// Reachable only via the admin-gated /upload route, but the endpoint re-checks
// the role — the server is the real boundary, this is UX.

// Mirrors RAW_SUFFIX_TO_FILE_TYPE on the server. Duplicated deliberately: it is
// the `accept` hint and the message shown before a request is made. The server
// re-validates, so a stale list here degrades to a clear 415 rather than a bad
// upload.
const ACCEPTED_EXTENSIONS = ['.glb', '.stl', '.obj'];

/** Server rejections already explain themselves; only map what it can't know. */
function describeFailure(caught: unknown): string {
  if (isApiError(caught, 'forbidden')) return 'Only admins can upload models.';
  if (isApiError(caught, 'network_error')) return 'Could not reach the server. Try again.';
  if (isApiError(caught, 'rate_limited')) return 'Too many uploads just now — wait a moment.';
  // unsupported_media_type / payload_too_large / validation_error all carry a
  // specific sentence naming the format or the limit, which beats anything
  // generic written here.
  if (caught instanceof ApiError && caught.message) return caught.message;
  return 'Something went wrong. Please try again.';
}

export function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState<ModelSummary[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function select(chosen: File | null) {
    setError(null);
    setFile(chosen);
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    select(event.dataTransfer.files[0] ?? null);
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    setError(null);
    setPending(true);
    try {
      const model = await uploadModel(file);
      setUploaded((prev) => [model, ...prev]);
      setFile(null);
      // The input keeps its own value, so clearing state alone would leave the
      // same filename showing and block re-selecting that file.
      if (inputRef.current) inputRef.current.value = '';
    } catch (caught) {
      setError(describeFailure(caught));
    } finally {
      setPending(false);
    }
  }

  return (
    <AppLayout>
      <h1>Upload a model</h1>
      <p className="page-lead">
        Add a mesh to the pipeline. It is converted, normalized and rendered like any ingested
        model, then appears in the browse grid ready to label. Accepted formats:{' '}
        {ACCEPTED_EXTENSIONS.join(', ')}.
      </p>

      <form className="form" onSubmit={onSubmit}>
        <label
          className={`dropzone${dragging ? ' dropzone-active' : ''}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_EXTENSIONS.join(',')}
            onChange={(event: ChangeEvent<HTMLInputElement>) =>
              select(event.target.files?.[0] ?? null)
            }
          />
          <span className="dropzone-text">
            {file ? file.name : 'Choose a file or drag one here'}
          </span>
        </label>

        <button className="btn-primary" type="submit" disabled={!file || pending}>
          {pending ? 'Uploading…' : 'Upload'}
        </button>
      </form>

      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}

      {uploaded.length > 0 && (
        <>
          <h2 className="section-title">Uploaded this session</h2>
          <p className="page-lead">
            Preprocessing runs in the background, so previews appear once the render stage
            finishes — usually within a minute.
          </p>
          <ul className="invite-list">
            {uploaded.map((model) => (
              <li key={model.uid} className="invite-row">
                <Link to={`/models/${model.uid}`}>{model.title}</Link>
                <span className="invite-expiry">queued for preprocessing</span>
              </li>
            ))}
          </ul>
        </>
      )}
    </AppLayout>
  );
}
