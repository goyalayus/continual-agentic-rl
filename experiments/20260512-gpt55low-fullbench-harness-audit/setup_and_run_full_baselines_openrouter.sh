#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

EXPERIMENT_DIR="experiments/20260512-gpt55low-fullbench-harness-audit"
WATCHER_UNIT="tau2-openrouter-key-watch.service"

SETUP_OPENROUTER_NO_WATCHER_WAKE=1 "${EXPERIMENT_DIR}/setup_openrouter_env.sh"

if command -v systemctl >/dev/null 2>&1 \
  && systemctl --user is-active --quiet "${WATCHER_UNIT}" 2>/dev/null; then
  echo "Stopping ${WATCHER_UNIT} before direct baseline launch."
  systemctl --user stop "${WATCHER_UNIT}" >/dev/null 2>&1 || true
fi

exec "${EXPERIMENT_DIR}/run_full_baselines_openrouter.sh"
