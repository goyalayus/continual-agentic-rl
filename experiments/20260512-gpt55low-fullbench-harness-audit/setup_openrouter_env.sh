#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

EXPERIMENT_DIR="experiments/20260512-gpt55low-fullbench-harness-audit"
DEFAULT_ENV_FILE="${EXPERIMENT_DIR}/.env.local"
ENV_FILE="${OPENROUTER_ENV_FILE:-${DEFAULT_ENV_FILE}}"
WATCHER_UNIT="tau2-openrouter-key-watch.service"

wake_key_provider_watcher() {
  if [[ "${SETUP_OPENROUTER_NO_WATCHER_WAKE:-}" == "1" ]]; then
    return
  fi
  if [[ "${ENV_FILE}" != "${DEFAULT_ENV_FILE}" ]]; then
    return
  fi
  if ! command -v systemctl >/dev/null 2>&1; then
    return
  fi
  if systemctl --user is-active --quiet "${WATCHER_UNIT}" 2>/dev/null; then
    if systemctl --user restart "${WATCHER_UNIT}" >/dev/null 2>&1; then
      echo "Restarted ${WATCHER_UNIT}; it will preflight and launch the baseline if ready."
    else
      echo "Warning: ${WATCHER_UNIT} is active, but could not be restarted." >&2
    fi
    return
  fi
  if ! systemctl --user is-enabled --quiet "${WATCHER_UNIT}" 2>/dev/null; then
    return
  fi
  if systemctl --user start "${WATCHER_UNIT}" >/dev/null 2>&1; then
    echo "Started ${WATCHER_UNIT}; it will preflight and launch the baseline if ready."
  else
    echo "Warning: ${WATCHER_UNIT} is enabled, but could not be started." >&2
  fi
}

read -rsp "OpenRouter key: " OPENROUTER_API_KEY
echo

if [[ -z "${OPENROUTER_API_KEY}" ]]; then
  echo "No key entered; leaving ${ENV_FILE} unchanged." >&2
  exit 2
fi

if [[ "${OPENROUTER_API_KEY}" != sk-or-* ]]; then
  echo "Warning: entered key does not look like an OpenRouter key; provider preflight will verify it." >&2
fi

umask 077
mkdir -p "$(dirname "${ENV_FILE}")"
tmp_file="$(mktemp "${ENV_FILE}.tmp.XXXXXX")"
trap 'rm -f "${tmp_file}"' EXIT

printf 'OPENROUTER_API_KEY=%s\n' "${OPENROUTER_API_KEY}" > "${tmp_file}"
mv "${tmp_file}" "${ENV_FILE}"
chmod 600 "${ENV_FILE}"
trap - EXIT

echo "Wrote ${ENV_FILE} with mode 600."
wake_key_provider_watcher
