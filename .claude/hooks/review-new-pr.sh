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
# to .claude/pr-review.log (gitignored), and a failure both writes one line to
# stderr and exits non-zero so asyncRewake surfaces it instead of it vanishing.
#
# NOTE: deliberately no `set -e`. The matcher is an unfiltered `Bash`, so this
# script runs on EVERY Bash tool call; an unguarded failure here would break all
# of them. The detection phase below must never exit non-zero — only the review
# itself may.
set -uo pipefail

# Detection-phase bail-out: nothing to review, and nothing is wrong.
pass() { exit 0; }

# Hard dependencies. Absent either, silently do nothing rather than fail Bash.
command -v jq >/dev/null 2>&1 || pass
command -v gh >/dev/null 2>&1 || pass

# The spawned session inherits this same settings.json. Without this guard, a
# `gh pr create` inside it would open a third session, and so on.
[ -z "${IMAGEGENIE_PR_REVIEW:-}" ] || pass

hook_input="$(cat)" || pass
bash_command="$(printf '%s' "$hook_input" | jq -r '.tool_input.command // empty' 2>/dev/null)" || pass

# Cheap pre-filter. Loose on purpose — the real gates are the repo and freshness
# checks below, which is what stops a command that merely *mentions* the words
# (a test, a heredoc, this file) from triggering a review.
printf '%s' "$bash_command" | grep -qE 'gh[[:space:]].*pr[[:space:]]+create' || pass

project_dir="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[ -n "$project_dir" ] && [ -d "$project_dir" ] || pass
cd "$project_dir" || pass

# `gh pr create` prints the new pull request's URL. A Bash tool response is
# sometimes a bare string and sometimes an object, so read both shapes.
tool_output="$(printf '%s' "$hook_input" | jq -r '
  if   (.tool_response | type) == "string" then .tool_response
  elif (.tool_response | type) == "object" then
    ((.tool_response.stdout // "") + "\n" + (.tool_response.output // ""))
  else "" end' 2>/dev/null)" || pass

pull_request_url="$(printf '%s' "$tool_output" \
  | grep -oE 'https://github\.com/[^[:space:]]+/pull/[0-9]+' | head -1)"

# `--web`, `> /dev/null` and `url=$(gh pr create ...)` all open a real PR while
# printing no URL here. Fall back to whatever PR this branch now has; the
# freshness gate below is what keeps that from re-reviewing an old one.
if [ -z "$pull_request_url" ]; then
  pull_request_url="$(gh pr view --json url --jq '.url' 2>/dev/null)"
fi
[ -n "$pull_request_url" ] || pass

# Keep owner/repo — `${url##*/}` alone would turn another repo's PR #7 into a
# review of *this* repo's #7.
pull_request_number="${pull_request_url##*/}"
pull_request_repo="$(printf '%s' "$pull_request_url" | sed -nE 's#^https://github\.com/([^/]+/[^/]+)/pull/[0-9]+$#\1#p')"
current_repo="$(gh repo view --json nameWithOwner --jq '.nameWithOwner' 2>/dev/null)"
[ -n "$pull_request_repo" ] && [ "$pull_request_repo" = "$current_repo" ] || pass

# The one gate that matters: review only a PR that was opened seconds ago, and
# is actually open. Without it, any command echoing a PR-shaped URL starts a
# real review — including this hook's own tests, and `gh pr create`'s
# "already exists: <url>" failure on stderr, which names a real older PR.
pr_state_and_age="$(gh pr view "$pull_request_number" --json state,createdAt \
  --jq '"\(.state) \(.createdAt)"' 2>/dev/null)" || pass
pull_request_state="${pr_state_and_age%% *}"
pull_request_created="${pr_state_and_age##* }"
[ "$pull_request_state" = "OPEN" ] || pass

created_epoch="$(date -j -u -f '%Y-%m-%dT%H:%M:%SZ' "$pull_request_created" '+%s' 2>/dev/null)"
[ -n "$created_epoch" ] || pass
# Widened only by the tests, so the accept path can be proven and not just the
# rejections — a script that refused everything would look identical otherwise.
[ "$(( $(date -u '+%s') - created_epoch ))" -le "${IMAGEGENIE_PR_REVIEW_MAX_AGE:-300}" ] || pass

review_log="$project_dir/.claude/pr-review.log"

# Read the repo, read the PR, post comments. `gh api` is here because inline
# review comments go through the REST API — `gh pr comment` only posts one
# top-level blob.
review_tools="Read,Grep,Glob"
review_tools="$review_tools,Bash(gh pr view:*),Bash(gh pr diff:*),Bash(gh pr list:*),Bash(gh pr comment:*)"
review_tools="$review_tools,Bash(gh api:*),Bash(gh repo view:*)"
review_tools="$review_tools,Bash(git diff:*),Bash(git log:*),Bash(git show:*),Bash(git status:*),Bash(git rev-parse:*)"

# --allowedTools is ADDITIVE to settings.json, and ~/.claude/settings.json grants
# Write/Edit/NotebookEdit unconditionally — so an allowlist alone does NOT stop
# this unattended session editing the tree. Only a deny rule does.
#
# The gh api denials block merging and ref deletion in their common spellings.
# Being a command-prefix match this narrows the hole rather than closing it:
# `gh api <path> --method PUT` puts the flag after the path and still matches.
# Accepted deliberately — dropping `gh api` would cost inline comments.
review_denials="Write,Edit,NotebookEdit,Agent,Skill"
review_denials="$review_denials,Bash(gh api --method PUT:*),Bash(gh api -X PUT:*)"
review_denials="$review_denials,Bash(gh api --method DELETE:*),Bash(gh api -X DELETE:*)"
review_denials="$review_denials,Bash(gh api --method PATCH:*),Bash(gh api -X PATCH:*)"
review_denials="$review_denials,Bash(gh pr merge:*),Bash(gh pr close:*),Bash(git push:*)"

# Exercises the wiring without spending a review:
#   IMAGEGENIE_PR_REVIEW_DRY_RUN=1 .claude/hooks/review-new-pr.sh <<<"$hook_json"
if [ -n "${IMAGEGENIE_PR_REVIEW_DRY_RUN:-}" ]; then
  printf 'would review %s #%s (created %s)\n' \
    "$pull_request_repo" "$pull_request_number" "$pull_request_created"
  printf 'allow: %s\ndeny:  %s\n' "$review_tools" "$review_denials"
  exit 0
fi

printf '\n=== %s  %s #%s  %s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$pull_request_repo" \
  "$pull_request_number" "$pull_request_url" >>"$review_log"

review_status=0
IMAGEGENIE_PR_REVIEW=1 claude -p "/code-review $pull_request_number --comment" \
  --allowedTools "$review_tools" \
  --disallowedTools "$review_denials" >>"$review_log" 2>&1 || review_status=$?

printf '=== finished with status %s\n' "$review_status" >>"$review_log"

# asyncRewake only reports that something failed; without this the waking
# session cannot tell a rate-limit crash from a clean review.
if [ "$review_status" -ne 0 ]; then
  printf 'PR review of %s #%s failed (exit %s) — see %s\n' \
    "$pull_request_repo" "$pull_request_number" "$review_status" "$review_log" >&2
fi
exit "$review_status"
