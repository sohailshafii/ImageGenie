import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { listTrainingRuns } from '../api/training';
import type { TrainingRunSummary } from '../api/types';
import { AppLayout } from '../components/AppLayout';

// Training dashboard list (FR-6): every training run with its status and
// headline (final training loss). Each row links to the run's detail page
// (config, data snapshot, cost curve). Read-only — runs are produced by
// ml/train.py writing directly to the DB, not by the app.
//
// Sorting is **client-side**. `GET /training-runs` returns every run in one
// unpaginated response and runs are minted one per `make train`, so there is
// nothing to page through and re-sorting costs a re-render rather than a
// round-trip. If the endpoint ever grows pagination, the ordering has to move to
// the server with it — sorting one page of many client-side would silently sort
// only that page.

type SortColumn = 'id' | 'status' | 'arch' | 'labelCount' | 'finalLoss' | 'startedAt';
type SortDirection = 'asc' | 'desc';

const COLUMNS: { key: SortColumn; label: string }[] = [
  { key: 'id', label: 'Run' },
  { key: 'status', label: 'Status' },
  { key: 'arch', label: 'Arch' },
  { key: 'labelCount', label: 'Labels' },
  { key: 'finalLoss', label: 'Final loss' },
  { key: 'startedAt', label: 'Started' },
];

/** The value a column sorts on. `startedAt` compares as a timestamp, not a string. */
function sortValue(run: TrainingRunSummary, column: SortColumn): string | number | null {
  if (column === 'startedAt') return Date.parse(run.startedAt);
  return run[column];
}

/**
 * Order `runs` by one column.
 *
 * Nulls sort last in **both** directions, matching how browse orders models with
 * no confidence: "no final loss recorded" is missing data, and floating it to the
 * top of an ascending sort would read as the best result.
 */
function sortRuns(
  runs: TrainingRunSummary[],
  column: SortColumn,
  direction: SortDirection,
): TrainingRunSummary[] {
  const sign = direction === 'asc' ? 1 : -1;
  return [...runs].sort((left, right) => {
    const leftValue = sortValue(left, column);
    const rightValue = sortValue(right, column);
    if (leftValue === null && rightValue === null) return 0;
    if (leftValue === null) return 1;
    if (rightValue === null) return -1;
    if (leftValue === rightValue) return left.id - right.id; // stable tiebreak
    return leftValue < rightValue ? -sign : sign;
  });
}

export function TrainingRunsPage() {
  const [runs, setRuns] = useState<TrainingRunSummary[] | null>(null);
  // Newest first — the server's own order, so the first paint doesn't reshuffle.
  const [sortColumn, setSortColumn] = useState<SortColumn>('startedAt');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  useEffect(() => {
    let active = true;
    listTrainingRuns().then((result) => {
      if (active) setRuns(result);
    });
    return () => {
      active = false;
    };
  }, []);

  const sorted = useMemo(
    () => (runs === null ? null : sortRuns(runs, sortColumn, sortDirection)),
    [runs, sortColumn, sortDirection],
  );

  function onSort(column: SortColumn) {
    if (column === sortColumn) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      // A new column starts descending for the numeric/date ones (biggest and
      // newest first is the usual question) and ascending for the text ones.
      setSortColumn(column);
      setSortDirection(column === 'arch' || column === 'status' ? 'asc' : 'desc');
    }
  }

  return (
    <AppLayout>
      <h1>Training runs</h1>
      <p className="page-lead">
        Every training run. Sort by any column — final loss ascending is the "which run went
        best" view. Open one to see its configuration, the labels it trained on, and whether
        the loss went down.
      </p>

      {sorted === null ? (
        <p className="page-lead">Loading…</p>
      ) : sorted.length === 0 ? (
        <p className="page-lead">
          No training runs yet. Start one with <code>make train</code>.
        </p>
      ) : (
        <div className="table-wrap">
          <table className="runs-table">
            <thead>
              <tr>
                {COLUMNS.map(({ key, label }) => (
                  <th
                    key={key}
                    aria-sort={
                      sortColumn === key
                        ? sortDirection === 'asc'
                          ? 'ascending'
                          : 'descending'
                        : 'none'
                    }
                  >
                    <button
                      type="button"
                      className={`column-sort${sortColumn === key ? ' is-sorted' : ''}`}
                      onClick={() => onSort(key)}
                    >
                      {label}
                      <span aria-hidden="true" className="sort-caret">
                        {sortColumn === key ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
                      </span>
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((run) => (
                <tr key={run.id}>
                  <td>
                    <Link to={`/training/${run.id}`}>#{run.id}</Link>
                  </td>
                  <td>
                    <span className={`status-badge is-${run.status}`}>{run.status}</span>
                  </td>
                  <td>{run.arch ?? '—'}</td>
                  <td>{run.labelCount ?? '—'}</td>
                  <td>{run.finalLoss === null ? '—' : run.finalLoss.toFixed(4)}</td>
                  <td className="runs-time">{new Date(run.startedAt).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </AppLayout>
  );
}
