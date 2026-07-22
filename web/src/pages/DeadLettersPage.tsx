import { useEffect, useState } from 'react';
import { listDeadLetters, retryDeadLetter } from '../api/catalog';
import type { DeadLetter } from '../api/types';
import { AppLayout } from '../components/AppLayout';

// Admin-only "failed ingestion" surface: jobs that failed a preprocessing stage,
// recorded by the worker that nacked them, with a per-row retry that republishes
// to the stage topic (server.md#dead-letters). Retried rows are kept server-side
// but drop out of this list, which shows what is still outstanding.
export function DeadLettersPage() {
  const [items, setItems] = useState<DeadLetter[] | null>(null);
  const [retrying, setRetrying] = useState<number | null>(null);

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
    setRetrying(item.id);
    try {
      await retryDeadLetter(item.id);
      setItems((prev) => (prev ? prev.filter((row) => row.id !== item.id) : prev));
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
                <tr key={item.id}>
                  <td className="dlq-uid">{item.uid.slice(0, 10)}…</td>
                  <td>
                    <span className={`stage-badge is-${item.stage}`}>{item.stage}</span>
                  </td>
                  <td className="dlq-error">
                    {/* Inner element because max-width on a <td> is ignored by
                        the auto table layout — the cell would just grow. */}
                    <div className="dlq-error-text" title={item.error}>
                      {item.error}
                      {item.deliveryAttempt !== null && (
                        <span className="dlq-attempts"> · attempt {item.deliveryAttempt}</span>
                      )}
                    </div>
                  </td>
                  <td className="dlq-time">{new Date(item.failedAt).toLocaleString()}</td>
                  <td>
                    <button
                      type="button"
                      className="btn-secondary"
                      disabled={retrying === item.id}
                      onClick={() => onRetry(item)}
                    >
                      {retrying === item.id ? 'Retrying…' : 'Retry'}
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
