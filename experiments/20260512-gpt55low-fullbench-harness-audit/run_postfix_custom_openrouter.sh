#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

EXPERIMENT_DIR="experiments/20260512-gpt55low-fullbench-harness-audit"
source "${EXPERIMENT_DIR}/load_benchmark_env.sh"
load_benchmark_env "${EXPERIMENT_DIR}"

if [[ -z "${AZURE_API_KEY:-}" || -z "${AZURE_API_BASE:-}" || -z "${AZURE_API_VERSION:-}" ]]; then
  echo "Missing Azure GPT-5.5 env. Put AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_VERSION in ${EXPERIMENT_DIR}/.env.local, then rerun this script." >&2
  exit 2
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Missing OPENROUTER_API_KEY for Qwen query embeddings. Put it in ${EXPERIMENT_DIR}/.env.local, then rerun this script." >&2
  exit 2
fi

MODEL="azure/gpt-5.5"
REASONING_EFFORT="low"
MAX_TOKENS=768
TEMPERATURE=1.0
MAX_STEPS=100
MAX_ERRORS=10
TIMEOUT_SECONDS=900
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CUSTOM_PARALLELISM="${CUSTOM_PARALLELISM:-388}"
POSTFIX_PREFIX="postfix_custom_azure_gpt55low_"

preflight_provider() {
  "${PYTHON_BIN}" "${EXPERIMENT_DIR}/provider_preflight.py" \
    --chat-model "${MODEL}" \
    --max-tokens "${MAX_TOKENS}" \
    --reasoning-effort "${REASONING_EFFORT}"
}

run_custom_repeats() {
  local log_path="${EXPERIMENT_DIR}/${POSTFIX_PREFIX}all_repeats.stdout.log"

  echo "[postfix-custom-start] ${POSTFIX_PREFIX}all_repeats $(date -Is)" | tee -a "${log_path}"
  "${PYTHON_BIN}" experiments/20260509-gpt54mini-harness/run_azure_batch.py \
    --model "${MODEL}" \
    --reasoning-effort "${REASONING_EFFORT}" \
    --subagent-delegation batch \
    --parallelism "${CUSTOM_PARALLELISM}" \
    --repeat "${POSTFIX_PREFIX}r1_s849558:849558" \
    --repeat "${POSTFIX_PREFIX}r2_s551167:551167" \
    --repeat "${POSTFIX_PREFIX}r3_s811445:811445" \
    --repeat "${POSTFIX_PREFIX}r4_s613921:613921" \
    --max-steps "${MAX_STEPS}" \
    --max-errors "${MAX_ERRORS}" \
    --max-tokens "${MAX_TOKENS}" \
    --timeout-seconds "${TIMEOUT_SECONDS}" \
    --temperature "${TEMPERATURE}" \
    --auto-resume 2>&1 | tee -a "${log_path}"
  echo "[postfix-custom-done] ${POSTFIX_PREFIX}all_repeats $(date -Is)" | tee -a "${log_path}"
}

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/record_launch_state.py" \
  --label postfix \
  --output "${EXPERIMENT_DIR}/postfix_launch_state.json"

preflight_provider

overall_status=0
run_custom_repeats || overall_status=1
echo "[postfix-job-exit] postfix-custom-all status=${overall_status} $(date -Is)"

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/analyze_comparison.py" \
  --custom-source-prefix "${POSTFIX_PREFIX}" \
  --default-source-prefix "baseline_default_tau_bm25_azure_gpt55low_" \
  --output-json "${EXPERIMENT_DIR}/postfix_comparison_summary.json" \
  --output-csv "${EXPERIMENT_DIR}/postfix_comparison_summary.csv"

"${PYTHON_BIN}" "${EXPERIMENT_DIR}/check_completeness.py" \
  --summary "${EXPERIMENT_DIR}/postfix_comparison_summary.json" \
  --custom-only \
  --required-custom-prefix "${POSTFIX_PREFIX}"

if [[ "${overall_status}" != "0" ]]; then
  echo "One or more postfix custom jobs failed; artifacts were preserved." >&2
  exit "${overall_status}"
fi

echo "[postfix-all-done] $(date -Is)"
