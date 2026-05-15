#!/usr/bin/env bash

# Load local benchmark provider settings without sourcing arbitrary shell code.
# This file is sourced by run scripts; it intentionally reads only known keys.

warn_benchmark_env_permissions() {
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

strip_env_quotes() {
  local value="$1"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  printf '%s' "${value}"
}

load_benchmark_env() {
  local experiment_dir="$1"
  local env_file="${BENCHMARK_ENV_FILE:-${experiment_dir}/.env.local}"

  if [[ -r "${env_file}" ]]; then
    warn_benchmark_env_permissions "${env_file}"
    while IFS= read -r line || [[ -n "${line}" ]]; do
      line="${line%$'\r'}"
      case "${line}" in
        AZURE_OPENAI_API_KEY=*|export\ AZURE_OPENAI_API_KEY=*)
          export AZURE_OPENAI_API_KEY="$(strip_env_quotes "${line#*=}")"
          ;;
        AZURE_OPENAI_ENDPOINT=*|export\ AZURE_OPENAI_ENDPOINT=*)
          export AZURE_OPENAI_ENDPOINT="$(strip_env_quotes "${line#*=}")"
          ;;
        AZURE_OPENAI_API_VERSION=*|export\ AZURE_OPENAI_API_VERSION=*)
          export AZURE_OPENAI_API_VERSION="$(strip_env_quotes "${line#*=}")"
          ;;
        AZURE_API_KEY=*|export\ AZURE_API_KEY=*)
          export AZURE_API_KEY="$(strip_env_quotes "${line#*=}")"
          ;;
        AZURE_API_BASE=*|export\ AZURE_API_BASE=*)
          export AZURE_API_BASE="$(strip_env_quotes "${line#*=}")"
          ;;
        AZURE_API_VERSION=*|export\ AZURE_API_VERSION=*)
          export AZURE_API_VERSION="$(strip_env_quotes "${line#*=}")"
          ;;
        OPENROUTER_API_KEY=*|export\ OPENROUTER_API_KEY=*)
          export OPENROUTER_API_KEY="$(strip_env_quotes "${line#*=}")"
          ;;
      esac
    done < "${env_file}"
  fi

  if [[ -n "${AZURE_OPENAI_API_KEY:-}" ]]; then
    export AZURE_API_KEY="${AZURE_OPENAI_API_KEY}"
  fi
  if [[ -n "${AZURE_OPENAI_ENDPOINT:-}" && -z "${AZURE_API_BASE:-}" ]]; then
    export AZURE_API_BASE="${AZURE_OPENAI_ENDPOINT}"
  fi
  if [[ -n "${AZURE_OPENAI_API_VERSION:-}" && -z "${AZURE_API_VERSION:-}" ]]; then
    export AZURE_API_VERSION="${AZURE_OPENAI_API_VERSION}"
  fi
}
