import { useEffect, useRef, useState } from 'react';
import { deleteModel, listModels, setLabel } from '../api/catalog';
import {
  CLASS_NAMES,
  type ClassName,
  type LabelSource,
  type ModelPage,
  type ModelSort,
} from '../api/types';
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
  const [sort, setSort] = useState<ModelSort>('uid');
  const [page, setPage] = useState(1);
  const [data, setData] = useState<ModelPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [savingUid, setSavingUid] = useState<string | null>(null);
  const [deletingUid, setDeletingUid] = useState<string | null>(null);
  const [focusedIndex, setFocusedIndex] = useState(0);
  const cardRefs = useRef<(HTMLElement | null)[]>([]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    listModels({
      page,
      pageSize: PAGE_SIZE,
      className: classFilter === 'all' ? undefined : classFilter,
      source: sourceFilter === 'all' ? undefined : sourceFilter,
      sort,
    })
      .then((result) => {
        if (!active) return;
        setData(result);
        // A new page of models means the old index points at a different card.
        cardRefs.current = [];
        setFocusedIndex(0);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [page, classFilter, sourceFilter, sort]);

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

  async function onDelete(uid: string) {
    setDeletingUid(uid);
    try {
      await deleteModel(uid);
      // Drop it from the current page in place and decrement the count, rather
      // than refetching — a refetch would reflow the grid and jump the admin's
      // position mid-cleanup. The page can end up one short until the next fetch
      // refills it, which is fine for an occasional admin action.
      setData((prev) =>
        prev
          ? { ...prev, items: prev.items.filter((m) => m.uid !== uid), total: prev.total - 1 }
          : prev,
      );
    } finally {
      setDeletingUid(null);
    }
  }

  /** Move keyboard focus to a card, clamped to the page. */
  function focusCard(index: number) {
    const items = data?.items ?? [];
    const clamped = Math.max(0, Math.min(index, items.length - 1));
    cardRefs.current[clamped]?.focus();
  }

  /**
   * Cards per row, measured rather than assumed — the grid is responsive CSS, so
   * the column count changes with the viewport and can't be hard-coded.
   */
  function columnsPerRow(): number {
    const cards = cardRefs.current.filter(Boolean) as HTMLElement[];
    if (cards.length === 0) return 1;
    const firstTop = cards[0].offsetTop;
    const inFirstRow = cards.filter((card) => card.offsetTop === firstTop).length;
    return Math.max(1, inFirstRow);
  }

  function onGridKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    // Once the dropdown has focus it owns its own keys — arrows change the
    // selection and letters do native type-ahead. Never fight that.
    const tag = (event.target as HTMLElement).tagName;
    if (tag === 'SELECT' || tag === 'INPUT' || tag === 'TEXTAREA') return;

    const items = data?.items ?? [];
    if (items.length === 0) return;
    const current = focusedIndex;

    switch (event.key) {
      case 'ArrowRight':
      case 'j':
        event.preventDefault();
        focusCard(current + 1);
        break;
      case 'ArrowLeft':
      case 'k':
        event.preventDefault();
        focusCard(current - 1);
        break;
      case 'ArrowDown':
        event.preventDefault();
        focusCard(current + columnsPerRow());
        break;
      case 'ArrowUp':
        event.preventDefault();
        focusCard(current - columnsPerRow());
        break;
      case 'Enter':
      case ' ': {
        // Confirm-and-advance: the sweep action. Weak labels are right ~90% of
        // the time (that's what the precision figure means), so this is the key
        // that gets pressed most.
        if (!canEdit) break;
        const model = items[current];
        if (!model?.className) break; // nothing to confirm on an unlabeled model
        event.preventDefault();
        void onSetLabel(model.uid, model.className);
        focusCard(current + 1);
        break;
      }
      case 'c':
        // Hand off to the class dropdown; its native type-ahead does the rest,
        // which beats inventing a 12-key mnemonic map to memorize.
        if (!canEdit) break;
        event.preventDefault();
        cardRefs.current[current]?.querySelector('select')?.focus();
        break;
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
          <label>
            Sort
            <select
              value={sort}
              onChange={(e) => {
                setSort(e.target.value as ModelSort);
                setPage(1); // a new order makes the current page meaningless
              }}
            >
              <option value="uid">default</option>
              <option value="confidence">least confident</option>
            </select>
          </label>
        </div>
      </div>

      {loading && !data ? (
        <p className="page-lead">Loading models…</p>
      ) : data && data.items.length === 0 ? (
        <p className="page-lead">No models match these filters.</p>
      ) : (
        <div
          className="model-grid"
          aria-busy={loading}
          onKeyDown={onGridKeyDown}
          role="group"
          aria-label="Models"
        >
          {data?.items.map((model, index) => (
            <ModelCard
              key={model.uid}
              model={model}
              canEdit={canEdit}
              saving={savingUid === model.uid}
              onSetLabel={onSetLabel}
              onDelete={canEdit ? onDelete : undefined}
              deleting={deletingUid === model.uid}
              cardRef={(element) => {
                cardRefs.current[index] = element;
              }}
              onFocusCard={() => setFocusedIndex(index)}
              tabIndex={index === focusedIndex ? 0 : -1}
            />
          ))}
        </div>
      )}

      {canEdit && data && data.items.length > 0 && (
        <p className="keyboard-hint">
          <kbd>Tab</kbd> into the grid, then <kbd>←</kbd> <kbd>→</kbd> <kbd>↑</kbd> <kbd>↓</kbd> to
          move · <kbd>Enter</kbd> to confirm and advance · <kbd>c</kbd> to change the class
        </p>
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
