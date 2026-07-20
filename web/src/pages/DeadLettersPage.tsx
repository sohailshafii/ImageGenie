import { useEffect, useState } from 'react';
import { listDeadLetters, retryDeadLetter } from '../api/catalog';
import type { DeadLetter, PipelineStage } from '../api/types';
import { AppLayout } from '../components/AppLayout';

// Admin-only "failed ingestion" surface: models that failed a preprocessing stage
// and landed in that stage's dead-letter queue, with a per-row retry. Against the
// mock this drops the row (assume re-run); the real backend reads the Pub/Sub DLQs
// and a retry republishes the message to the stage topic (see server.md#queue).
const retryKey = (uid: string, stage: PipelineStage) => `${uid}:${stage}`;

export function DeadLettersPage() {
  const [items, setItems] = useState<DeadLetter[] | null>(null);
  const [retrying, setRetrying] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    listDeadLetters().then((result) => {
      if (active) setItems(result);
    });
    return () => {
      active = false;
    };
  }, []);

  async function onRetry(item: DeadLetter) {
    setRetrying(retryKey(item.uid, item.stage));
    try {
      await retryDeadLetter(item.uid, item.stage);
      setItems((prev) =>
        prev ? prev.filter((row) => !(row.uid === item.uid && row.stage === item.stage)) : prev,
      );
    } finally {
      setRetrying(null);
    }
  }

  return (
    <AppLayout>
      <h1>Failed ingestion</h1>
      <p className="page-lead">
        Models that failed a preprocessing stage and were quarantined in its dead-letter queue.
        Retry re-enqueues one into its stage — transient failures (timeouts, SSL) usually clear;
        genuine ones (empty meshes, oversized models) will fail again.
      </p>

      {items === null ? (
        <p className="page-lead">Loading…</p>
      ) : items.length === 0 ? (
        <p className="page-lead">No failed models. 🎉</p>
      ) : (
        <div className="table-wrap">
          <table className="dlq-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Stage</th>
                <th>Error</th>
                <th>Failed</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={retryKey(item.uid, item.stage)}>
                  <td className="dlq-uid">{item.uid.slice(0, 10)}…</td>
                  <td>
                    <span className={`stage-badge is-${item.stage}`}>{item.stage}</span>
                  </td>
                  <td className="dlq-error">{item.error}</td>
                  <td className="dlq-time">{new Date(item.failedAt).toLocaleString()}</td>
                  <td>
                    <button
                      type="button"
                      className="btn-secondary"
                      disabled={retrying === retryKey(item.uid, item.stage)}
                      onClick={() => onRetry(item)}
                    >
                      {retrying === retryKey(item.uid, item.stage) ? 'Retrying…' : 'Retry'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </AppLayout>
  );
}
