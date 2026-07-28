import { useState } from 'react';
import { predictModelClass } from '../api/catalog';
import { ApiError, isApiError } from '../api/errors';
import type { ClassName, Prediction } from '../api/types';

// "What does the classifier think this is?" on the model detail page
// (web.md#predicting-a-class). Available to viewers and admins alike: the answer
// is a statement about the catalog, which any authenticated user can already
// read. The trained model itself stays admin-only (NFR-6).

// How many of the twelve to show. The full ranking is fetched — the tail is what
// reveals a near-tie — but a list of twelve buries the answer, and everything
// past the first few is noise on a model that is confident.
const SHOWN = 3;

export function PredictPanel({ uid, currentLabel }: { uid: string; currentLabel: ClassName | null }) {
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onPredict() {
    setBusy(true);
    setError(null);
    try {
      setPrediction(await predictModelClass(uid));
    } catch (caught) {
      // Two of the three outcomes are states, not faults, and saying so is the
      // whole point: a freshly uploaded model has not been rendered yet (404),
      // and a deployment with no completed run has nothing to answer with (503).
      // Reporting either as "could not classify" invites debugging a system that
      // is working correctly.
      if (isApiError(caught, 'not_found')) {
        setError('Not rendered yet — this model is still going through the pipeline.');
      } else if (isApiError(caught, 'unavailable')) {
        setError('No trained model yet — run a training job first.');
      } else {
        setError(caught instanceof ApiError ? caught.message : 'Could not classify this model.');
      }
    } finally {
      setBusy(false);
    }
  }

  const top = prediction?.predictions[0];
  // Worth calling out: the classifier disagreeing with the stored label is the
  // signal the M8 review loop is looking for, so it should not need spotting.
  const disagrees = top !== undefined && currentLabel !== null && top.className !== currentLabel;

  return (
    <div className="detail-field">
      <span className="detail-label">Classifier</span>
      <div className="predict-panel">
        {prediction === null ? (
          <button type="button" className="btn-secondary" disabled={busy} onClick={onPredict}>
            {busy ? 'Classifying…' : 'What is this?'}
          </button>
        ) : (
          <>
            <ol className="predict-list">
              {prediction.predictions.slice(0, SHOWN).map((row) => (
                <li key={row.className}>
                  <span className="predict-class">{row.className}</span>
                  {/* A bar as well as a number: the gap between first and second
                      is the reading that matters, and it is far easier to see
                      than to compute from two percentages. */}
                  <span className="predict-bar" aria-hidden="true">
                    <span style={{ width: `${Math.round(row.probability * 100)}%` }} />
                  </span>
                  <span className="predict-probability">
                    {(row.probability * 100).toFixed(1)}%
                  </span>
                </li>
              ))}
            </ol>
            {disagrees && (
              <p className="form-note">
                Disagrees with the current label ({currentLabel}).
              </p>
            )}
            <p className="form-note">
              from training run #{prediction.runId} ·{' '}
              <button type="button" className="link-button" onClick={onPredict} disabled={busy}>
                {busy ? 'classifying…' : 'again'}
              </button>
            </p>
          </>
        )}
        {error !== null && <p className="form-error">{error}</p>}
      </div>
    </div>
  );
}
