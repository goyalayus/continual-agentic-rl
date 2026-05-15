#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

EXPERIMENT_DIR="experiments/20260512-gpt55low-fullbench-harness-audit"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
source "${EXPERIMENT_DIR}/load_benchmark_env.sh"
load_benchmark_env "${EXPERIMENT_DIR}"

echo "[doctor] offline scaffold checks"
"${PYTHON_BIN}" -m py_compile "${EXPERIMENT_DIR}"/*.py
"${PYTHON_BIN}" "${EXPERIMENT_DIR}/selftest.py"
while IFS= read -r script; do
  bash -n "${script}"
done < <(find "${EXPERIMENT_DIR}" -maxdepth 1 -type f -name '*.sh' | sort)
if rg --hidden -n '[[:blank:]]+$' \
  "${EXPERIMENT_DIR}"/*.py \
  "${EXPERIMENT_DIR}"/*.sh \
  "${EXPERIMENT_DIR}"/*.md \
  "${EXPERIMENT_DIR}/.gitignore"; then
  echo "[doctor] trailing whitespace found in scaffold files" >&2
  exit 1
fi
git diff --check -- "${EXPERIMENT_DIR}"

echo
echo "[doctor] current benchmark status"
"${PYTHON_BIN}" "${EXPERIMENT_DIR}/status.py" || true

echo
echo "[doctor] final objective audit"
"${PYTHON_BIN}" "${EXPERIMENT_DIR}/completion_audit.py" || true

echo
echo "[doctor] provider preflight"
if [[ -z "${AZURE_API_KEY:-}" || -z "${AZURE_API_BASE:-}" || -z "${AZURE_API_VERSION:-}" ]]; then
  echo "Azure GPT-5.5 env is incomplete; skipping provider preflight"
elif [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set; skipping provider preflight"
else
  "${PYTHON_BIN}" "${EXPERIMENT_DIR}/provider_preflight.py" || true
fi

echo
echo "[doctor] done"
