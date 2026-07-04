#!/usr/bin/env bash
# PostToolUse hook: after a code file is edited, remind Claude to keep the
# matching domain doc (server/server.md | web/web.md | ml/ml.md) in sync.
# Reads the tool-call JSON on stdin, prints a hookSpecificOutput reminder on stdout.
set -euo pipefail

input="$(cat)"
file="$(printf '%s' "$input" | jq -r '.tool_input.file_path // .tool_response.filePath // empty')"

# Nothing to do if we can't tell what file was touched.
[ -n "$file" ] || exit 0

base="$(basename "$file")"

# Skip non-code: markdown (incl. the docs themselves), config, and the hook dir.
case "$file" in
  *.md|*.markdown) exit 0 ;;
  */.claude/*) exit 0 ;;
esac
case "$base" in
  *.json|*.lock|*.txt|*.toml|*.yaml|*.yml|*.cfg|*.ini) exit 0 ;;
esac

# Map the path to the most relevant domain doc. Lowercase for matching.
lc="$(printf '%s' "$file" | tr '[:upper:]' '[:lower:]')"
doc=""
case "$lc" in
  */web/*|*frontend*|*ui*|*.tsx|*.jsx|*.vue|*.svelte|*.css|*.html|*viewer*|*dashboard*)
    doc="web/web.md" ;;
  *train*|*eval*|*/ml/*|*model*|*.ipynb|*classifier*|*dataset*|*pointnet*|*resnet*)
    doc="ml/ml.md" ;;
  */server/*|*worker*|*api*|*pipeline*|*queue*|*ingest*|*storage*|*/db/*|*database*)
    doc="server/server.md" ;;
esac

if [ -n "$doc" ]; then
  msg="Code changed in \`$file\`. If this changes design/behavior, update **$doc** to match (design docs are the source of truth)."
else
  msg="Code changed in \`$file\`. If this changes design/behavior, update the relevant domain doc — server/server.md, web/web.md, or ml/ml.md — to match."
fi

jq -n --arg m "$msg" \
  '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $m}}'
