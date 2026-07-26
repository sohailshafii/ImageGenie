import { useState } from 'react';
import { ApiError, isApiError } from '../api/errors';

// A button that runs a download and reports what went wrong inline.
//
// Downloads go through fetch rather than a plain `<a href>` (api/client.ts), so
// the failures are real errors rather than a JSON body saved to disk — and the
// common one is not a failure at all: an artifact that the pipeline simply
// hasn't produced yet comes back 404. That reads as "not available yet", not as
// something broken, which is the distinction this component exists to draw.
export function DownloadButton({
  label,
  onDownload,
  missingLabel = 'Not available yet',
  className = 'btn-secondary',
  title,
}: {
  label: string;
  onDownload: () => Promise<void>;
  /** Shown for a 404 — the artifact hasn't been produced (or was removed). */
  missingLabel?: string;
  className?: string;
  title?: string;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      await onDownload();
    } catch (caught) {
      if (isApiError(caught, 'not_found')) {
        setError(missingLabel);
      } else if (isApiError(caught, 'forbidden')) {
        setError('Admins only');
      } else {
        setError(caught instanceof ApiError ? caught.message : 'Download failed');
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <span className="download-action">
      <button type="button" className={className} disabled={busy} title={title} onClick={run}>
        {busy ? 'Downloading…' : label}
      </button>
      {error && <span className="download-error">{error}</span>}
    </span>
  );
}
