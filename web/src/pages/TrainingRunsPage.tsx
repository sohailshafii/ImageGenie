import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { listTrainingRuns } from '../api/training';
import type { TrainingRunSummary } from '../api/types';
import { AppLayout } from '../components/AppLayout';

// Training dashboard list (FR-6): every training run, newest first, with its
// status and headline (final training loss). Each row links to the run's detail
// page (config, data snapshot, cost curve). Read-only — runs are produced by
// ml/train.py writing directly to the DB, not by the app.
export function TrainingRunsPage() {
  const [runs, setRuns] = useState<TrainingRunSummary[] | null>(null);

  useEffect(() => {
    let active = true;
    listTrainingRuns().then((result) => {
      if (active) setRuns(result);
    });
    return () => {
      active = false;
    };
  }, []);

  return (
    <AppLayout>
      <h1>Training runs</h1>
      <p className="page-lead">
        Every training run, newest first. Open one to see its configuration, the labels it
        trained on, and whether the loss went down.
      </p>

      {runs === null ? (
        <p className="page-lead">Loading…</p>
      ) : runs.length === 0 ? (
        <p className="page-lead">
          No training runs yet. Start one with <code>make train</code>.
        </p>
      ) : (
        <div className="table-wrap">
          <table className="runs-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th>Arch</th>
                <th>Labels</th>
                <th>Final loss</th>
                <th>Started</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
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
