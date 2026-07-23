import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { listDeletedModels, restoreModel } from '../api/catalog';
import type { ModelPage } from '../api/types';
import { AppLayout } from '../components/AppLayout';

const PAGE_SIZE = 24;

// Admin-only restore queue (web.md#data-upload): soft-deleted models, newest
// first, each restorable. Reachable only via the admin-gated /deleted route; the
// endpoint re-checks the role.
export function DeletedModelsPage() {
  const [page, setPage] = useState(1);
  const [data, setData] = useState<ModelPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [restoringUid, setRestoringUid] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    listDeletedModels({ page, pageSize: PAGE_SIZE })
      .then((result) => {
        if (active) setData(result);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [page]);

  async function onRestore(uid: string) {
    setRestoringUid(uid);
    try {
      await restoreModel(uid);
      // Gone from the deleted queue now — drop it from the page in place.
      setData((prev) =>
        prev
          ? { ...prev, items: prev.items.filter((m) => m.uid !== uid), total: prev.total - 1 }
          : prev,
      );
    } finally {
      setRestoringUid(null);
    }
  }

  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.pageSize)) : 1;

  return (
    <AppLayout>
      <h1>Deleted models</h1>
      <p className="page-lead">
        Soft-deleted models, most recently deleted first. Restoring one returns it to the browse
        grid with its labels intact. Their files are never removed, so this is always reversible.
      </p>

      {loading && !data ? (
        <p className="page-lead">Loading…</p>
      ) : data && data.items.length === 0 ? (
        <p className="page-lead">Nothing deleted.</p>
      ) : (
        <ul className="invite-list">
          {data?.items.map((model) => (
            <li key={model.uid} className="invite-row">
              <Link to={`/models/${model.uid}`} className="invite-email">
                {model.title}
              </Link>
              <button
                type="button"
                className="btn-secondary"
                disabled={restoringUid === model.uid}
                onClick={() => onRestore(model.uid)}
              >
                {restoringUid === model.uid ? 'Restoring…' : 'Restore'}
              </button>
            </li>
          ))}
        </ul>
      )}

      {data && data.total > 0 && (
        <nav className="pager" aria-label="Pagination">
          <button
            type="button"
            className="btn-secondary"
            disabled={page <= 1 || loading}
            onClick={() => setPage((current) => current - 1)}
          >
            ← Prev
          </button>
          <span className="pager-status">
            Page {data.page} of {totalPages} · {data.total} deleted
          </span>
          <button
            type="button"
            className="btn-secondary"
            disabled={page >= totalPages || loading}
            onClick={() => setPage((current) => current + 1)}
          >
            Next →
          </button>
        </nav>
      )}
    </AppLayout>
  );
}
