import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { deleteModel, getModel, getModelArtifacts, setLabel } from '../api/catalog';
import { CLASS_NAMES, type ClassName, type ModelSummary } from '../api/types';
import { useAuth } from '../auth/AuthContext';
import { AppLayout } from '../components/AppLayout';
import { ConfirmButton } from '../components/ConfirmButton';
import { ModelViewer } from '../components/ModelViewer';

// Detail view (web.md): a single model in the interactive three.js viewer, its
// candidate label + confidence with confirm/correct (admin), and the store
// metadata (title/tags) that aids the labeling decision.
export function DetailPage() {
  const { uid = '' } = useParams();
  const navigate = useNavigate();
  const { user } = useAuth();
  const canEdit = user?.role === 'admin';

  const [model, setModel] = useState<ModelSummary | null>(null);
  const [meshUrl, setMeshUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'not-found'>('loading');
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    let active = true;
    setStatus('loading');
    getModel(uid)
      .then((result) => {
        if (!active) return;
        setModel(result);
        setStatus('ready');
      })
      .catch(() => {
        if (active) setStatus('not-found');
      });
    return () => {
      active = false;
    };
  }, [uid]);

  // Fetched separately from the summary: the artifacts call checks each blob
  // exists, so it's slower, and the label panel shouldn't wait on the mesh to
  // become editable. A model with no mesh yet is normal, not an error.
  useEffect(() => {
    let active = true;
    setMeshUrl(null);
    getModelArtifacts(uid)
      .then((artifacts) => {
        if (active) setMeshUrl(artifacts.mesh);
      })
      .catch(() => {
        if (active) setMeshUrl(null);
      });
    return () => {
      active = false;
    };
  }, [uid]);

  async function onSetLabel(className: ClassName) {
    setSaving(true);
    try {
      setModel(await setLabel(uid, className));
    } finally {
      setSaving(false);
    }
  }

  async function onDelete() {
    setDeleting(true);
    try {
      await deleteModel(uid);
      // The model is gone from browse now; there's nothing left to show here.
      navigate('/');
    } catch {
      setDeleting(false); // stay on the page so the admin can retry
    }
  }

  return (
    <AppLayout>
      <p className="form-note" style={{ marginTop: 0 }}>
        <Link to="/">← Back to browse</Link>
      </p>

      {status === 'loading' && <p className="page-lead">Loading model…</p>}
      {status === 'not-found' && <p className="page-lead">That model wasn’t found.</p>}

      {status === 'ready' && model && (
        <div className="detail-layout">
          <ModelViewer src={meshUrl} />

          <aside className="detail-panel">
            <h1>{model.title}</h1>

            <div className="detail-field">
              <span className="detail-label">Label</span>
              {canEdit ? (
                <div className="model-label-row">
                  <select
                    className="model-class-select"
                    value={model.className ?? ''}
                    disabled={saving}
                    aria-label="Class"
                    onChange={(e) => onSetLabel(e.target.value as ClassName)}
                  >
                    {model.className === null && (
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
                  {model.source === 'weak' && (
                    <button
                      type="button"
                      className="btn-secondary btn-confirm"
                      disabled={saving}
                      onClick={() => onSetLabel(model.className as ClassName)}
                    >
                      Confirm
                    </button>
                  )}
                </div>
              ) : (
                <span className="model-class">{model.className ?? 'unlabeled'}</span>
              )}
            </div>

            <div className="detail-field">
              <span className="detail-label">Source</span>
              <span>
                <span className={`source-badge is-${model.source ?? 'none'}`}>
                  {model.source ?? 'unlabeled'}
                </span>
                {model.confidence !== null && (
                  <span className="model-confidence"> · {Math.round(model.confidence * 100)}%</span>
                )}
              </span>
            </div>

            <div className="detail-field">
              <span className="detail-label">Tags</span>
              <span className="detail-tags">
                {model.tags.map((tag) => (
                  <span key={tag} className="tag-chip">
                    {tag}
                  </span>
                ))}
              </span>
            </div>

            <div className="detail-field">
              <span className="detail-label">Model id</span>
              <span className="dlq-uid">{model.uid}</span>
            </div>

            {canEdit && (
              <div className="detail-danger">
                <ConfirmButton
                  className="btn-danger"
                  busy={deleting}
                  onConfirm={onDelete}
                  idleLabel="Delete model"
                  armedLabel="Click again to delete"
                  title="Soft-delete: hides the model but keeps its data, restorable later"
                />
              </div>
            )}
          </aside>
        </div>
      )}
    </AppLayout>
  );
}
