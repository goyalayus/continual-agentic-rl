#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

REPO_ROOT="$(pwd)"
EXPERIMENT_DIR="experiments/20260512-gpt55low-fullbench-harness-audit"
WATCHER_UNIT="tau2-openrouter-key-watch.service"
WATCHER_SCRIPT="${REPO_ROOT}/${EXPERIMENT_DIR}/wait_for_key_and_provider_and_run.sh"
LOG_PATH="${REPO_ROOT}/${EXPERIMENT_DIR}/key_provider_watcher.stdout.log"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
UNIT_PATH="${USER_UNIT_DIR}/${WATCHER_UNIT}"

benchmark_process_running() {
  ps -eo cmd= | grep -E \
    "run_full_baselines_openrouter.sh|baseline_custom_openrouter_gpt55low_|baseline_default_tau_bm25_openrouter_gpt55low_4trials_seed4101" \
    | grep -v -E "grep|wait_for_key_and_provider_and_run.sh" \
    >/dev/null
}

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl is not available; run ${EXPERIMENT_DIR}/wait_for_key_and_provider_and_run.sh directly." >&2
  exit 2
fi

if ! systemctl --user is-system-running >/dev/null 2>&1; then
  echo "user systemd is not running; run ${EXPERIMENT_DIR}/wait_for_key_and_provider_and_run.sh directly." >&2
  exit 2
fi

if benchmark_process_running; then
  echo "A baseline benchmark process appears to be running; leaving ${WATCHER_UNIT} unchanged." >&2
  exit 2
fi

mkdir -p "${USER_UNIT_DIR}"
tmp_file="$(mktemp "${UNIT_PATH}.tmp.XXXXXX")"
trap 'rm -f "${tmp_file}"' EXIT

cat > "${tmp_file}" <<UNIT
[Unit]
Description=Tau2 OpenRouter key/provider watcher

[Service]
Type=simple
WorkingDirectory=${REPO_ROOT}
Environment=INTERVAL_SECONDS=${INTERVAL_SECONDS}
Environment=MAX_ATTEMPTS=${MAX_ATTEMPTS}
ExecStart=/usr/bin/bash -lc "exec >> '${LOG_PATH}' 2>&1; exec '${WATCHER_SCRIPT}'"

[Install]
WantedBy=default.target
UNIT

mv "${tmp_file}" "${UNIT_PATH}"
trap - EXIT

if systemctl --user is-active --quiet "${WATCHER_UNIT}" 2>/dev/null; then
  systemctl --user stop "${WATCHER_UNIT}"
fi
systemctl --user reset-failed "${WATCHER_UNIT}" >/dev/null 2>&1 || true
systemctl --user daemon-reload
systemctl --user enable "${WATCHER_UNIT}" >/dev/null
systemctl --user start "${WATCHER_UNIT}"

systemctl --user status "${WATCHER_UNIT}" --no-pager --lines=12 || true
echo
echo "Installed user unit: ${UNIT_PATH}"
echo "Watcher log: ${LOG_PATH}"
