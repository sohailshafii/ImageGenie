import { Link } from 'react-router-dom';
import { CLASS_NAMES, type ClassName, type ModelSummary } from '../api/types';

// A single model in the browse grid: a rendered-preview placeholder, its label +
// confidence, and (for admins) inline confirm/correct. Thumbnails are a class-tinted
// tile with an emoji until real multi-view renders are served by the backend.
const CLASS_EMOJI: Record<ClassName, string> = {
  animal: '🐾',
  food: '🍎',
  car: '🚗',
  chair: '🪑',
  weapon: '⚔️',
  electronics: '🔌',
  figure: '🧍',
  lamp: '💡',
  aircraft: '✈️',
  building: '🏢',
  table: '🪵',
  plant: '🪴',
};

export function ModelCard({
  model,
  canEdit,
  saving,
  onSetLabel,
}: {
  model: ModelSummary;
  canEdit: boolean;
  saving: boolean;
  onSetLabel: (uid: string, className: ClassName) => void;
}) {
  const isManual = model.source === 'manual';
  // No label yet — the model predates weak labeling, or the backfill hasn't run.
  const isUnlabeled = model.className === null;

  return (
    <article className="model-card">
      <Link to={`/models/${model.uid}`} className="model-thumb" aria-label={`Open ${model.title}`}>
        <span aria-hidden="true">{model.className ? CLASS_EMOJI[model.className] : '❓'}</span>
      </Link>

      <div className="model-body">
        <Link to={`/models/${model.uid}`} className="model-title" title={model.title}>
          {model.title}
        </Link>

        <div className="model-label-row">
          {canEdit ? (
            <select
              className="model-class-select"
              value={model.className ?? ''}
              disabled={saving}
              aria-label={`Class for ${model.title}`}
              onChange={(e) => onSetLabel(model.uid, e.target.value as ClassName)}
            >
              {isUnlabeled && (
                <option value="" disabled>
                  — pick a class —
                </option>
              )}
              {CLASS_NAMES.map((className) => (
                <option key={className} value={className}>
                  {className}
                </option>
              ))}
            </select>
          ) : (
            <span className="model-class">{model.className ?? 'unlabeled'}</span>
          )}

          {canEdit && !isManual && !isUnlabeled && (
            <button
              type="button"
              className="btn-secondary btn-confirm"
              disabled={saving}
              onClick={() => onSetLabel(model.uid, model.className as ClassName)}
            >
              Confirm
            </button>
          )}
        </div>

        <div className="model-meta">
          <span className={`source-badge is-${model.source ?? 'none'}`}>
            {model.source ?? 'unlabeled'}
          </span>
          {model.confidence !== null && (
            <span className="model-confidence">{Math.round(model.confidence * 100)}%</span>
          )}
        </div>
      </div>
    </article>
  );
}
