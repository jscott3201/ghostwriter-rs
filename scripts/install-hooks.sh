#!/usr/bin/env bash
# One-time local setup: point git at the tracked .githooks/ directory.
# Run once per clone:  bash scripts/install-hooks.sh
#
# .githooks/ is version-controlled (unlike .git/hooks/), so the team shares the
# same gates. Mirrors the CI split:
#   pre-commit -> cargo fmt --check + file-size cap + no-secret scan  (fast)
#   pre-push   -> cargo clippy -D warnings (fast; the full test/build matrix
#                 runs at the development->main release gate)
#
# Escape hatches: `git commit/push --no-verify` (once) or
# `export GW_SKIP_HOOKS=1` (whole shell session).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

git config core.hooksPath .githooks
chmod +x .githooks/pre-commit .githooks/pre-push 2>/dev/null || true

echo "core.hooksPath -> .githooks"
echo "  pre-commit: cargo fmt --check + file-size cap + no-secret scan"
echo "  pre-push:   cargo clippy -D warnings (fast; full matrix at release gate)"
echo "Skip once: --no-verify   |   skip session: export GW_SKIP_HOOKS=1"
