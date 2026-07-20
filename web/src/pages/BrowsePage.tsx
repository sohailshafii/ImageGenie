import { useEffect, useState } from 'react';
import { listModels, setLabel } from '../api/catalog';
import { CLASS_NAMES, type ClassName, type LabelSource, type ModelPage } from '../api/types';
import { useAuth } from '../auth/AuthContext';
import { AppLayout } from '../components/AppLayout';
import { ModelCard } from '../components/ModelCard';

const PAGE_SIZE = 24;

// Browse view (web.md): a paginated grid of models with inline confirm/correct.
// Corrections are admin-only (normal users view) — the server enforces this too.
export function BrowsePage() {
  const { user } = useAuth();
  const canEdit = user?.role === 'admin';

  const [classFilter, setClassFilter] = useState<ClassName | 'all'>('all');
  const [sourceFilter, setSourceFilter] = useState<LabelSource | 'all'>('all');
  const [page, setPage] = useState(1);
  const [data, setData] = useState<ModelPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [savingUid, setSavingUid] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    listModels({
      page,
      pageSize: PAGE_SIZE,
      className: classFilter === 'all' ? undefined : classFilter,
      source: sourceFilter === 'all' ? undefined : sourceFilter,
    })
      .then((result) => {
        if (active) setData(result);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [page, classFilter, sourceFilter]);

  async function onSetLabel(uid: string, className: ClassName) {
    setSavingUid(uid);
    try {
      const updated = await setLabel(uid, className);
      setData((prev) =>
        prev
          ? { ...prev, items: prev.items.map((m) => (m.uid === uid ? updated : m)) }
          : prev,
      );
    } finally {
      setSavingUid(null);
    }
  }

  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.pageSize)) : 1;

  return (
    <AppLayout>
      <div className="page-head">
        <h1>Browse</h1>
        <div className="filters">
          <label>
            Class
            <select
              value={classFilter}
              onChange={(e) => {
                setClassFilter(e.target.value as ClassName | 'all');
                setPage(1);
              }}
            >
              <option value="all">all</option>
              {CLASS_NAMES.map((className) => (
                <option key={className} value={className}>
                  {className}
                </option>
              ))}
            </select>
          </label>
          <label>
            Source
            <select
              value={sourceFilter}
              onChange={(e) => {
                setSourceFilter(e.target.value as LabelSource | 'all');
                setPage(1);
              }}
            >
              <option value="all">all</option>
              <option value="weak">weak</option>
              <option value="manual">manual</option>
            </select>
          </label>
        </div>
      </div>

      {loading && !data ? (
        <p className="page-lead">Loading models…</p>
      ) : data && data.items.length === 0 ? (
        <p className="page-lead">No models match these filters.</p>
      ) : (
        <div className="model-grid" aria-busy={loading}>
          {data?.items.map((model) => (
            <ModelCard
              key={model.uid}
              model={model}
              canEdit={canEdit}
              saving={savingUid === model.uid}
              onSetLabel={onSetLabel}
            />
          ))}
        </div>
      )}

      {data && data.total > 0 && (
        <nav className="pager" aria-label="Pagination">
          <button
            type="button"
            className="btn-secondary"
            disabled={page <= 1 || loading}
            onClick={() => setPage((p) => p - 1)}
          >
            ← Prev
          </button>
          <span className="pager-status">
            Page {data.page} of {totalPages} · {data.total} models
          </span>
          <button
            type="button"
            className="btn-secondary"
            disabled={page >= totalPages || loading}
            onClick={() => setPage((p) => p + 1)}
          >
            Next →
          </button>
        </nav>
      )}
    </AppLayout>
  );
}
