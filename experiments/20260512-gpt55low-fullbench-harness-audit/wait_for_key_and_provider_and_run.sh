#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

EXPERIMENT_DIR="experiments/20260512-gpt55low-fullbench-harness-audit"
source "${EXPERIMENT_DIR}/load_openrouter_env.sh"

INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
attempt=1

baseline_complete() {
  uv run python "${EXPERIMENT_DIR}/check_completeness.py" >/dev/null 2>&1
}

benchmark_process_running() {
  ps -eo cmd= | grep -E \
    "run_full_baselines_openrouter.sh|baseline_custom_openrouter_gpt55low_|baseline_default_tau_bm25_openrouter_gpt55low_4trials_seed4101" \
    | grep -v -E "grep|wait_for_key_and_provider_and_run.sh" \
    >/dev/null
}

while true; do
  echo "[key-provider-watch] attempt=${attempt} $(date -Is)"
  load_openrouter_env "${EXPERIMENT_DIR}"

  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "[key-provider-watch] no OPENROUTER_API_KEY yet; waiting for ${EXPERIMENT_DIR}/.env.local or a key inherited at watcher start"
  elif baseline_complete; then
    echo "[key-provider-watch] baseline is already complete; exiting"
    exit 0
  elif benchmark_process_running; then
    echo "[key-provider-watch] benchmark process already running; not starting a duplicate"
  elif uv run python "${EXPERIMENT_DIR}/provider_preflight.py"; then
    echo "[key-provider-watch] key and provider ready; starting full baseline run $(date -Is)"
    exec "${EXPERIMENT_DIR}/run_full_baselines_openrouter.sh"
  else
    echo "[key-provider-watch] key found, but provider preflight is not ready"
  fi

  if [[ "${MAX_ATTEMPTS}" != "0" && "${attempt}" -ge "${MAX_ATTEMPTS}" ]]; then
    echo "[key-provider-watch] max attempts reached; exiting without starting run" >&2
    exit 1
  fi

  attempt=$((attempt + 1))
  echo "[key-provider-watch] sleeping ${INTERVAL_SECONDS}s"
  sleep "${INTERVAL_SECONDS}"
done
