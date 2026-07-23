import { useState } from 'react';
import { Link } from 'react-router-dom';
import { CLASS_NAMES, type ClassName, type ModelSummary } from '../api/types';
import { ConfirmButton } from './ConfirmButton';

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
  onDelete,
  deleting,
  cardRef,
  onFocusCard,
  tabIndex,
}: {
  model: ModelSummary;
  canEdit: boolean;
  saving: boolean;
  onSetLabel: (uid: string, className: ClassName) => void;
  /** Admin-only soft delete. Omitted for non-admins, so the control never renders. */
  onDelete?: (uid: string) => void;
  deleting?: boolean;
  /** Lets the grid move focus between cards for keyboard sweeps. */
  cardRef?: (element: HTMLElement | null) => void;
  onFocusCard?: () => void;
  /**
   * Roving tabindex: only the active card is 0, the rest -1. That way Tab enters
   * the grid once and lands where the user left off, instead of stepping through
   * all 24 cards.
   */
  tabIndex?: number;
}) {
  const isManual = model.source === 'manual';
  // No label yet — the model predates weak labeling, or the backfill hasn't run.
  const isUnlabeled = model.className === null;
  // Falls back to the class-emoji tile when the render isn't there.
  const [previewFailed, setPreviewFailed] = useState(false);

  return (
    <article
      className="model-card"
      ref={cardRef}
      // Focusable so a keyboard sweep can land on it; the grid handles the keys.
      tabIndex={tabIndex ?? -1}
      onFocus={onFocusCard}
    >
      {canEdit && onDelete && (
        <ConfirmButton
          className="card-delete"
          busy={deleting}
          onConfirm={() => onDelete(model.uid)}
          idleLabel={<span aria-hidden="true">✕</span>}
          armedLabel="Delete?"
          title={`Delete ${model.title}`}
        />
      )}

      <Link to={`/models/${model.uid}`} className="model-thumb" aria-label={`Open ${model.title}`}>
        {model.thumbnail && !previewFailed ? (
          <img
            src={model.thumbnail}
            alt=""
            className="model-thumb-image"
            loading="lazy" // a page of cards shouldn't fetch every render at once
            // The server emits the URL without checking the blob exists, so a
            // 404 here means "not rendered yet", not an error worth surfacing.
            onError={() => setPreviewFailed(true)}
          />
        ) : (
          <span aria-hidden="true">{model.className ? CLASS_EMOJI[model.className] : '❓'}</span>
        )}
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
