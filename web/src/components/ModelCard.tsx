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

  return (
    <article className="model-card">
      <div className="model-thumb" aria-hidden="true">
        <span>{CLASS_EMOJI[model.className]}</span>
      </div>

      <div className="model-body">
        <p className="model-title" title={model.title}>
          {model.title}
        </p>

        <div className="model-label-row">
          {canEdit ? (
            <select
              className="model-class-select"
              value={model.className}
              disabled={saving}
              aria-label={`Class for ${model.title}`}
              onChange={(e) => onSetLabel(model.uid, e.target.value as ClassName)}
            >
              {CLASS_NAMES.map((className) => (
                <option key={className} value={className}>
                  {className}
                </option>
              ))}
            </select>
          ) : (
            <span className="model-class">{model.className}</span>
          )}

          {canEdit && !isManual && (
            <button
              type="button"
              className="btn-secondary btn-confirm"
              disabled={saving}
              onClick={() => onSetLabel(model.uid, model.className)}
            >
              Confirm
            </button>
          )}
        </div>

        <div className="model-meta">
          <span className={`source-badge is-${model.source}`}>{model.source}</span>
          {!isManual && <span className="model-confidence">{Math.round(model.confidence * 100)}%</span>}
        </div>
      </div>
    </article>
  );
}
