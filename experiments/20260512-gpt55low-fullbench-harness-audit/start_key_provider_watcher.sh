#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

REPO_ROOT="$(pwd)"
EXPERIMENT_DIR="experiments/20260512-gpt55low-fullbench-harness-audit"
WATCHER_UNIT="tau2-openrouter-key-watch.service"
LOG_PATH="${REPO_ROOT}/${EXPERIMENT_DIR}/key_provider_watcher.stdout.log"
WATCHER_SCRIPT="${REPO_ROOT}/${EXPERIMENT_DIR}/wait_for_key_and_provider_and_run.sh"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "systemd-run is not available; run ${EXPERIMENT_DIR}/wait_for_key_and_provider_and_run.sh directly." >&2
  exit 2
fi

if ! systemctl --user is-system-running >/dev/null 2>&1; then
  echo "user systemd is not running; run ${EXPERIMENT_DIR}/wait_for_key_and_provider_and_run.sh directly." >&2
  exit 2
fi

if systemctl --user is-active --quiet "${WATCHER_UNIT}" 2>/dev/null; then
  echo "${WATCHER_UNIT} is already running."
  systemctl --user status "${WATCHER_UNIT}" --no-pager --lines=8 || true
  exit 0
fi

systemctl --user reset-failed "${WATCHER_UNIT}" >/dev/null 2>&1 || true

systemd-run --user \
  --unit="${WATCHER_UNIT%.service}" \
  --collect \
  --property=WorkingDirectory="${REPO_ROOT}" \
  /usr/bin/env \
  INTERVAL_SECONDS="${INTERVAL_SECONDS}" \
  MAX_ATTEMPTS="${MAX_ATTEMPTS}" \
  /usr/bin/bash -lc "exec >> '${LOG_PATH}' 2>&1; exec '${WATCHER_SCRIPT}'"

sleep 1
systemctl --user status "${WATCHER_UNIT}" --no-pager --lines=12 || true
echo
echo "Watcher log: ${LOG_PATH}"
