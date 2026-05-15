#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

EXPERIMENT_DIR="experiments/20260512-gpt55low-fullbench-harness-audit"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

export TAU2_DEFAULT_LLM_NL_ASSERTIONS="${TAU2_DEFAULT_LLM_NL_ASSERTIONS:-azure/gpt-5.5}"
export TAU2_DEFAULT_LLM_ENV_INTERFACE="${TAU2_DEFAULT_LLM_ENV_INTERFACE:-azure/gpt-5.5}"
export TAU2_DEFAULT_LLM_EVAL_USER_SIMULATOR="${TAU2_DEFAULT_LLM_EVAL_USER_SIMULATOR:-azure/gpt-5.5}"

exec "${PYTHON_BIN}" "${EXPERIMENT_DIR}/run_full_baselines_one_process.py" "$@"
