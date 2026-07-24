#!/usr/bin/env bash
#
# Adopt Alembic on the existing Cloud SQL database by DROP-and-REBUILD, then
# rebuild the model/artifact tables from object storage and seed labels + an admin
# (server.md#deploying-the-api-to-cloud-run, server.md#migrations).
#
# The database was built by the workers' create_all and has no alembic_version, so
# `alembic upgrade head` would abort on CREATE TABLE model. Rather than hand-apply a
# delta, this drops the schema and rebuilds it cleanly — safe ONLY because every
# artifact key embeds its uid, so `reconcile_from_storage` reconstructs the rows
# from the buckets. That is why this verifies the buckets are populated BEFORE it
# drops anything.
#
#   scripts/adopt_schema.sh                 # prompts before the destructive step
#   scripts/adopt_schema.sh --yes           # non-interactive (CI / repeat runs)
#
# Admin bootstrap reads IMAGEGENIE_ADMIN_EMAIL and IMAGEGENIE_ADMIN_PASSWORD; if
# either is unset it prompts (unless --yes, which then skips the admin step with a
# warning).

set -euo pipefail

# ── Config (override via env) ───────────────────────────────────────────────
PROJECT="${IMAGEGENIE_GCP_PROJECT:-imagegenie-pipeline}"
REGION="${IMAGEGENIE_GCP_REGION:-us-central1}"
INSTANCE="${IMAGEGENIE_SQL_INSTANCE:-imagegenie-pg}"
DB_SECRET="${IMAGEGENIE_DB_SECRET:-imagegenie-database-url}"
RAW_BUCKET="${IMAGEGENIE_RAW_BUCKET:-${PROJECT}-raw}"
PROCESSED_BUCKET="${IMAGEGENIE_PROCESSED_BUCKET:-${PROJECT}-processed}"
PROXY_PORT="${IMAGEGENIE_PROXY_PORT:-5433}"
DB_NAME="${IMAGEGENIE_DB_NAME:-imagegenie}"
DB_USER="${IMAGEGENIE_DB_USER:-imagegenie}"

ASSUME_YES=0
[[ "${1:-}" == "--yes" ]] && ASSUME_YES=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${REPO_ROOT}/.venv/bin/python"
ALEMBIC="${REPO_ROOT}/.venv/bin/alembic"

CONNECTION_NAME="${PROJECT}:${REGION}:${INSTANCE}"
PROXY_PID=""

log()  { printf '\033[1;34m▶ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

cleanup() { [[ -n "$PROXY_PID" ]] && kill "$PROXY_PID" 2>/dev/null || true; }
trap cleanup EXIT

# ── Preflight ───────────────────────────────────────────────────────────────
command -v gcloud >/dev/null || die "gcloud not found"
command -v psql   >/dev/null || die "psql not found (brew install libpq)"
[[ -x "$VENV_PY" ]] || die "no venv at $VENV_PY — run 'make setup'"

PROXY_BIN="$(command -v cloud-sql-proxy || true)"
if [[ -z "$PROXY_BIN" ]]; then
  PROXY_BIN="$(gcloud info --format='value(installation.sdk_root)')/bin/cloud-sql-proxy"
fi
[[ -x "$PROXY_BIN" ]] || die "cloud-sql-proxy not found (gcloud components install cloud-sql-proxy)"

log "Reading the database password from Secret Manager ($DB_SECRET)"
DB_URL_SECRET="$(gcloud secrets versions access latest --secret="$DB_SECRET" --project="$PROJECT")"
# The secret is the unix-socket form; pull the password out of user:PASSWORD@.
DB_PASSWORD="$(printf '%s' "$DB_URL_SECRET" | sed -E 's#^[^:]+://[^:]+:([^@]+)@.*$#\1#')"
[[ -n "$DB_PASSWORD" && "$DB_PASSWORD" != "$DB_URL_SECRET" ]] || die "could not parse the DB password from the secret"

# ── Safety: the buckets must hold the artifacts a rebuild reconstructs from ──
# Cheap existence check: `head -1` stops the listing early (so this doesn't
# enumerate 150k objects), and the `|| true` wrappers absorb the resulting SIGPIPE
# so pipefail + set -e don't abort on a *populated* bucket.
bucket_has_objects() {
  local first
  first="$( (gcloud storage ls "$1" 2>/dev/null || true) | head -1 || true)"
  [[ -n "$first" ]]
}

log "Verifying the buckets are populated before dropping anything"
bucket_has_objects "gs://${RAW_BUCKET}/raw/" \
  || die "raw bucket gs://${RAW_BUCKET} looks empty — refusing to drop the schema (a rebuild would have nothing to restore from)"
bucket_has_objects "gs://${PROCESSED_BUCKET}/processed/" \
  || die "processed bucket gs://${PROCESSED_BUCKET} looks empty — refusing to drop the schema"
printf '  raw and processed buckets both contain objects.\n'

# ── Confirm ─────────────────────────────────────────────────────────────────
cat <<EOF

  About to, on Cloud SQL instance '${INSTANCE}' (project ${PROJECT}):
    1. DROP SCHEMA public CASCADE   ← destroys every current table + row
    2. alembic upgrade head          ← rebuild the schema from migrations
    3. reconcile from gs://${RAW_BUCKET} + gs://${PROCESSED_BUCKET}
    4. backfill metadata + weak labels
    5. create/refresh the admin account

  The rows are reconstructed from the buckets (verified above). content_hash and
  any manual labels not in weak_labels.csv are NOT recoverable.

EOF
if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "  Type 'drop and rebuild' to proceed: " reply
  [[ "$reply" == "drop and rebuild" ]] || die "aborted"
fi

# ── Cloud SQL proxy (TCP) ───────────────────────────────────────────────────
log "Starting the Cloud SQL proxy on 127.0.0.1:${PROXY_PORT}"
"$PROXY_BIN" "$CONNECTION_NAME" --port "$PROXY_PORT" >/dev/null 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 30); do
  PGPASSWORD="$DB_PASSWORD" psql -h 127.0.0.1 -p "$PROXY_PORT" -U "$DB_USER" -d "$DB_NAME" -c '\q' 2>/dev/null && break
  sleep 1
done
PGPASSWORD="$DB_PASSWORD" psql -h 127.0.0.1 -p "$PROXY_PORT" -U "$DB_USER" -d "$DB_NAME" -c '\q' 2>/dev/null \
  || die "could not connect through the proxy"

# The TCP connection string the app tools use for the rest of this run.
export IMAGEGENIE_DATABASE_URL="postgresql+psycopg://${DB_USER}:${DB_PASSWORD}@127.0.0.1:${PROXY_PORT}/${DB_NAME}"
export IMAGEGENIE_STORAGE_BACKEND="gcs"
export IMAGEGENIE_RAW_BUCKET="$RAW_BUCKET"
export IMAGEGENIE_PROCESSED_BUCKET="$PROCESSED_BUCKET"
export IMAGEGENIE_PUBSUB_PROJECT="$PROJECT"

# ── 1. Drop + recreate the schema ───────────────────────────────────────────
log "Dropping and recreating the public schema"
PGPASSWORD="$DB_PASSWORD" psql -h 127.0.0.1 -p "$PROXY_PORT" -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 <<'SQL'
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
SQL

# ── 2. Migrations ───────────────────────────────────────────────────────────
log "Running alembic upgrade head"
( cd "$REPO_ROOT/server" && "$ALEMBIC" upgrade head )

# ── 3. Rebuild model + artifact from storage ────────────────────────────────
log "Rebuilding model/artifact from object storage"
( cd "$REPO_ROOT/server" && "$VENV_PY" -m app.reconcile_from_storage )

# ── 4. Backfills ────────────────────────────────────────────────────────────
log "Backfilling Objaverse metadata (titles/tags)"
( cd "$REPO_ROOT/server" && "$VENV_PY" -m app.backfill_metadata )
if [[ -f "$REPO_ROOT/data/exploration/weak_labels.csv" ]]; then
  log "Backfilling weak labels"
  ( cd "$REPO_ROOT/server" && "$VENV_PY" -m app.backfill_labels \
      --labels "$REPO_ROOT/data/exploration/weak_labels.csv" \
      --eval "$REPO_ROOT/data/exploration/weak_label_eval.json" )
else
  warn "no data/exploration/weak_labels.csv — skipping the weak-label backfill (models will be unlabeled)"
fi

# ── 5. Admin bootstrap ──────────────────────────────────────────────────────
if [[ "$ASSUME_YES" -eq 1 && ( -z "${IMAGEGENIE_ADMIN_EMAIL:-}" || -z "${IMAGEGENIE_ADMIN_PASSWORD:-}" ) ]]; then
  warn "--yes with no IMAGEGENIE_ADMIN_EMAIL/PASSWORD — skipping the admin step; run 'python -m app.create_admin --email ...' yourself"
else
  admin_email="${IMAGEGENIE_ADMIN_EMAIL:-}"
  [[ -z "$admin_email" ]] && read -r -p "  Admin email: " admin_email
  log "Creating/refreshing admin ${admin_email}"
  ( cd "$REPO_ROOT/server" && "$VENV_PY" -m app.create_admin --email "$admin_email" )
fi

log "Done. The database matches Alembic head and is rebuilt from storage."
