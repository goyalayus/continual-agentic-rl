#!/usr/bin/env bash

# Load a local OpenRouter key without requiring it on the command line.
# This file is sourced by the run scripts; it intentionally reads only one key.

warn_openrouter_env_permissions() {
  local env_file="$1"
  local mode=""

  if [[ ! -f "${env_file}" ]]; then
    return 0
  fi

  mode="$(stat -c '%a' "${env_file}" 2>/dev/null || true)"
  if [[ ! "${mode}" =~ ^[0-7]+$ ]]; then
    return 0
  fi

  if (( (8#${mode} & 8#077) != 0 )); then
    echo "Warning: ${env_file} is readable or writable by group/other; run chmod 600 ${env_file}." >&2
  fi
}

load_openrouter_env() {
  local experiment_dir="$1"
  local env_file="${OPENROUTER_ENV_FILE:-${experiment_dir}/.env.local}"
  local key_line=""

  if [[ -n "${OPENROUTER_API_KEY:-}" || ! -r "${env_file}" ]]; then
    return 0
  fi

  warn_openrouter_env_permissions "${env_file}"

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    case "${line}" in
      OPENROUTER_API_KEY=*)
        key_line="${line#OPENROUTER_API_KEY=}"
        ;;
      export\ OPENROUTER_API_KEY=*)
        key_line="${line#export OPENROUTER_API_KEY=}"
        ;;
    esac
  done < "${env_file}"

  if [[ -z "${key_line}" ]]; then
    return 0
  fi

  key_line="${key_line%\"}"
  key_line="${key_line#\"}"
  key_line="${key_line%\'}"
  key_line="${key_line#\'}"

  export OPENROUTER_API_KEY="${key_line}"
}
