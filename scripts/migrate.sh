#!/usr/bin/env bash
#
# Apply pending Alembic migrations to the Cloud SQL database — the NON-destructive
# companion to adopt_schema.sh (server.md#deploying-the-api-to-cloud-run).
#
# adopt_schema.sh DROPS the schema and rebuilds it from storage; that is only for
# the first cutover to Alembic. Every *later* deploy that carries a new migration
# uses this instead — it just runs `alembic upgrade head`, preserving all data.
#
#   scripts/migrate.sh        # show current revision, upgrade to head, show it again
#
# Run it BEFORE rolling the new image, so the new code never sees a DB missing a
# column it expects. `alembic upgrade head` is idempotent — a no-op if already current.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
[[ -f "$REPO_ROOT/.env" ]] && { set -a; . "$REPO_ROOT/.env"; set +a; }

# ── Config (override via env) ───────────────────────────────────────────────
PROJECT="${IMAGEGENIE_GCP_PROJECT:-imagegenie-pipeline}"
REGION="${IMAGEGENIE_GCP_REGION:-us-central1}"
INSTANCE="${IMAGEGENIE_SQL_INSTANCE:-imagegenie-pg}"
DB_SECRET="${IMAGEGENIE_DB_SECRET:-imagegenie-database-url}"
PROXY_PORT="${IMAGEGENIE_PROXY_PORT:-5433}"
DB_NAME="${IMAGEGENIE_DB_NAME:-imagegenie}"
DB_USER="${IMAGEGENIE_DB_USER:-imagegenie}"

ALEMBIC="${REPO_ROOT}/.venv/bin/alembic"
CONNECTION_NAME="${PROJECT}:${REGION}:${INSTANCE}"
PROXY_PID=""

log() { printf '\033[1;34m▶ %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

cleanup() { [[ -n "$PROXY_PID" ]] && kill "$PROXY_PID" 2>/dev/null || true; }
trap cleanup EXIT

# ── Preflight ───────────────────────────────────────────────────────────────
command -v gcloud >/dev/null || die "gcloud not found"
command -v psql   >/dev/null || die "psql not found (brew install libpq)"
[[ -x "$ALEMBIC" ]] || die "no venv alembic at $ALEMBIC — run 'make setup'"

PROXY_BIN="$(command -v cloud-sql-proxy || true)"
if [[ -z "$PROXY_BIN" ]]; then
  PROXY_BIN="$(gcloud info --format='value(installation.sdk_root)')/bin/cloud-sql-proxy"
fi
[[ -x "$PROXY_BIN" ]] || die "cloud-sql-proxy not found (brew install cloud-sql-proxy)"

log "Reading the database password from Secret Manager ($DB_SECRET)"
DB_URL_SECRET="$(gcloud secrets versions access latest --secret="$DB_SECRET" --project="$PROJECT")"
DB_PASSWORD="$(printf '%s' "$DB_URL_SECRET" | sed -E 's#^[^:]+://[^:]+:([^@]+)@.*$#\1#')"
[[ -n "$DB_PASSWORD" && "$DB_PASSWORD" != "$DB_URL_SECRET" ]] || die "could not parse the DB password from the secret"

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

export IMAGEGENIE_DATABASE_URL="postgresql+psycopg://${DB_USER}:${DB_PASSWORD}@127.0.0.1:${PROXY_PORT}/${DB_NAME}"

# ── Upgrade ─────────────────────────────────────────────────────────────────
log "Current database revision:"
( cd "$REPO_ROOT/server" && "$ALEMBIC" current 2>&1 | sed 's/^/  /' )
log "Applying migrations up to head"
( cd "$REPO_ROOT/server" && "$ALEMBIC" upgrade head )
log "Now at:"
( cd "$REPO_ROOT/server" && "$ALEMBIC" current 2>&1 | sed 's/^/  /' )
log "Done. The database matches Alembic head."
