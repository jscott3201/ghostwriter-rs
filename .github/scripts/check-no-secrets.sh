#!/usr/bin/env bash
# Baseline no-secret scan. Greps tracked files for common secret-shaped
# patterns. Catches the noisy floor; the `_*/` working dirs and `*-id.json`
# identities are gitignored (never tracked) so they are out of scope by
# construction.

set -euo pipefail

violations=0

scan() {
  local pattern="$1"
  local label="$2"
  if git grep -nE "$pattern" >/dev/null 2>&1; then
    echo "FAIL: matched $label pattern in tracked files:"
    git grep -nE "$pattern" || true
    violations=$((violations + 1))
  fi
}

scan 'AKIA[0-9A-Z]{16}' 'AWS access key id'
scan '-----BEGIN (RSA|EC|OPENSSH|PGP) PRIVATE KEY-----' 'private key block'
scan 'xox[abpr]-[A-Za-z0-9-]{10,}' 'Slack token'
scan 'gh[pousr]_[A-Za-z0-9]{36,}' 'GitHub token'
scan 'sk-[A-Za-z0-9_-]{20,}' 'API token (sk- prefix)'
scan 'sk-or-v1-[A-Za-z0-9]{20,}' 'OpenRouter API key'

if [ "$violations" -gt 0 ]; then
  echo
  echo "Remove secrets from the working tree, rotate compromised credentials, and rewrite history."
  exit 1
fi

echo "OK: baseline no-secret scan clean."
