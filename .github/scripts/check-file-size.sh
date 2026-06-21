#!/usr/bin/env bash
# 700-line cap per source file (AGENTS.md hard rule). Counts non-empty,
# non-comment lines in tracked Rust source files; flags any that exceed the cap.
# Runs from repo root. Compatible with macOS bash 3.x (no mapfile).

set -euo pipefail

CAP=700
violations=0

while IFS= read -r f; do
  [ -f "$f" ] || continue
  loc=$(grep -cvE '^\s*(//.*)?$' "$f" || true)
  if [ "$loc" -gt "$CAP" ]; then
    echo "FAIL: $f has $loc LOC (cap: $CAP)"
    violations=$((violations + 1))
  fi
done < <(
  git ls-files '*.rs' 2>/dev/null \
    | grep -v -E '^(target|generated|out)/'
)

if [ "$violations" -gt 0 ]; then
  echo
  echo "Refactor or split files exceeding the $CAP LOC cap. See AGENTS.md."
  exit 1
fi

echo "OK: all tracked source files within the $CAP LOC cap."
