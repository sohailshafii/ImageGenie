#!/usr/bin/env bash
# PostToolUse hook: when `gh pr create` opens a pull request, run a code review
# in a fresh headless Claude session and post the findings to the PR as inline
# comments.
#
# The review deliberately starts with NO context from the session that wrote the
# code. A reviewer that never saw the reasoning is the one that catches what the
# author talked themselves into.
#
# Runs async, so nothing it prints reaches the terminal — everything is appended
# to .claude/pr-review.log (gitignored) instead, and a non-zero exit wakes Claude
# via asyncRewake rather than failing silently.
set -euo pipefail

# The spawned session inherits this same settings.json. Without this guard, a
# `gh pr create` inside it would open a third session, and so on.
if [ -n "${IMAGEGENIE_PR_REVIEW:-}" ]; then
  exit 0
fi

hook_input="$(cat)"
bash_command="$(printf '%s' "$hook_input" | jq -r '.tool_input.command // empty')"

# Cheap pre-filter. The real gate is whether a PR URL comes back below, so this
# only has to be loose enough not to miss a real `gh pr create`.
printf '%s' "$bash_command" | grep -qE 'gh[[:space:]].*pr[[:space:]]+create' || exit 0

# `gh pr create` prints the new pull request's URL on stdout. A Bash tool
# response is sometimes a bare string and sometimes an object, so read both.
tool_output="$(printf '%s' "$hook_input" | jq -r '
  if   (.tool_response | type) == "string" then .tool_response
  elif (.tool_response | type) == "object" then (.tool_response.stdout // .tool_response.output // "")
  else "" end')"

pull_request_url="$(printf '%s' "$tool_output" \
  | grep -oE 'https://github\.com/[^[:space:]]+/pull/[0-9]+' | head -1 || true)"

# No URL means no PR was actually opened — the command failed, or it only
# mentioned the words. Either way there is nothing to review.
[ -n "$pull_request_url" ] || exit 0

pull_request_number="${pull_request_url##*/}"

project_dir="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"
review_log="$project_dir/.claude/pr-review.log"

# Bounded on purpose: read the repo, read the PR, post comments. No file writes,
# no general Bash, and nothing that can merge. `gh api` is here because inline
# review comments go through the REST API rather than `gh pr comment`.
review_tools="Read,Grep,Glob"
review_tools="$review_tools,Bash(gh pr view:*),Bash(gh pr diff:*),Bash(gh pr comment:*)"
review_tools="$review_tools,Bash(gh api:*)"
review_tools="$review_tools,Bash(git diff:*),Bash(git log:*),Bash(git show:*)"

# Exercises the wiring without spending a review:
#   IMAGEGENIE_PR_REVIEW_DRY_RUN=1 .claude/hooks/review-new-pr.sh <<<"$hook_json"
if [ -n "${IMAGEGENIE_PR_REVIEW_DRY_RUN:-}" ]; then
  printf 'would review PR #%s (%s)\n' "$pull_request_number" "$pull_request_url"
  printf 'allowedTools: %s\n' "$review_tools"
  exit 0
fi

printf '\n=== %s  PR #%s  %s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$pull_request_number" "$pull_request_url" \
  >>"$review_log"

review_status=0
IMAGEGENIE_PR_REVIEW=1 claude -p "/code-review $pull_request_number --comment" \
  --allowedTools "$review_tools" >>"$review_log" 2>&1 || review_status=$?

printf '=== finished with status %s\n' "$review_status" >>"$review_log"
exit "$review_status"
