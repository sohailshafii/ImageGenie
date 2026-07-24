#!/usr/bin/env bash
#
# Verify the deployed API is healthy and — the point of this script — that URL
# signing actually works, which is the one deploy gotcha only real Cloud Run can
# confirm (server.md#deploying-the-api-to-cloud-run). It hits the health endpoint
# and scans recent logs for the "falling back to streaming" warning that signals a
# missing IAM binding.
#
#   scripts/check_deploy.sh              # health + signing + error scan
#   scripts/check_deploy.sh --since 30m  # widen the log window (default 15m)

set -euo pipefail

# Load ./.env (gitignored) so project/region/service overrides are picked up.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
[[ -f "$REPO_ROOT/.env" ]] && { set -a; . "$REPO_ROOT/.env"; set +a; }

PROJECT="${IMAGEGENIE_GCP_PROJECT:-imagegenie-pipeline}"
REGION="${IMAGEGENIE_GCP_REGION:-us-central1}"
SERVICE="${IMAGEGENIE_API_SERVICE:-imagegenie-api}"
SINCE="15m"
[[ "${1:-}" == "--since" && -n "${2:-}" ]] && SINCE="$2"

pass() { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m! %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m✗ %s\033[0m\n' "$*"; }
info() { printf '\033[1;34m▶ %s\033[0m\n' "$*"; }

command -v gcloud >/dev/null || { fail "gcloud not found"; exit 1; }

status=0

# ── Service + URL ───────────────────────────────────────────────────────────
info "Looking up $SERVICE"
URL="$(gcloud run services describe "$SERVICE" --region="$REGION" --project="$PROJECT" \
        --format='value(status.url)' 2>/dev/null || true)"
[[ -n "$URL" ]] || { fail "service $SERVICE not found in $REGION"; exit 1; }
pass "service URL: $URL"

READY="$(gcloud run services describe "$SERVICE" --region="$REGION" --project="$PROJECT" \
          --format='value(status.conditions.filter("type=Ready").extract("status"))' 2>/dev/null || true)"
if [[ "$READY" == *True* ]]; then pass "revision Ready"; else warn "revision not Ready ($READY)"; status=1; fi

# ── Health ──────────────────────────────────────────────────────────────────
info "GET /api/healthz"
code="$(curl -s -o /dev/null -w '%{http_code}' "$URL/api/healthz" || true)"
if [[ "$code" == "200" ]]; then pass "healthz 200"; else fail "healthz returned $code"; status=1; fi

info "GET / (SPA shell)"
ctype="$(curl -s -o /dev/null -w '%{content_type}' "$URL/" || true)"
if [[ "$ctype" == text/html* ]]; then pass "root serves HTML ($ctype)"; else warn "root content-type: $ctype"; status=1; fi

# ── Logs: the signing fallback + any errors ─────────────────────────────────
base_filter="resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE}"

info "Scanning the last $SINCE of logs for the signing fallback"
sign_hits="$(gcloud logging read \
  "${base_filter} AND textPayload:\"falling back to streaming\"" \
  --project="$PROJECT" --freshness="$SINCE" --limit=5 --format='value(textPayload)' 2>/dev/null || true)"
if [[ -z "$sign_hits" ]]; then
  pass "no 'falling back to streaming' warnings — URL signing is working"
else
  fail "signing is falling back to streaming — the serviceAccountTokenCreator binding or IMAGEGENIE_SIGNER_SA_EMAIL is wrong:"
  printf '    %s\n' "$sign_hits"
  status=1
fi

info "Scanning for errors (severity>=ERROR)"
errs="$(gcloud logging read \
  "${base_filter} AND severity>=ERROR" \
  --project="$PROJECT" --freshness="$SINCE" --limit=10 \
  --format='value(severity, textPayload)' 2>/dev/null || true)"
if [[ -z "$errs" ]]; then
  pass "no ERROR-level logs in the last $SINCE"
else
  warn "recent errors (investigate):"
  printf '    %s\n' "$errs"
  status=1
fi

echo
if [[ "$status" -eq 0 ]]; then pass "all checks passed"; else fail "some checks need attention (see above)"; fi
exit "$status"
