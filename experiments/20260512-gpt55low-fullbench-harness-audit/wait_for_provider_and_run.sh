#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

EXPERIMENT_DIR="experiments/20260512-gpt55low-fullbench-harness-audit"
source "${EXPERIMENT_DIR}/load_openrouter_env.sh"
load_openrouter_env "${EXPERIMENT_DIR}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Missing OPENROUTER_API_KEY. Export it or put it in ${EXPERIMENT_DIR}/.env.local before using this watcher." >&2
  exit 2
fi

INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
attempt=1

while true; do
  echo "[provider-watch] attempt=${attempt} $(date -Is)"
  if uv run python "${EXPERIMENT_DIR}/provider_preflight.py"; then
    echo "[provider-watch] provider ready; starting full baseline run $(date -Is)"
    exec "${EXPERIMENT_DIR}/run_full_baselines_openrouter.sh"
  fi

  if [[ "${MAX_ATTEMPTS}" != "0" && "${attempt}" -ge "${MAX_ATTEMPTS}" ]]; then
    echo "[provider-watch] max attempts reached; exiting without starting run" >&2
    exit 1
  fi

  attempt=$((attempt + 1))
  echo "[provider-watch] not ready; sleeping ${INTERVAL_SECONDS}s"
  sleep "${INTERVAL_SECONDS}"
done
