#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

EXPERIMENT_DIR="experiments/20260512-gpt55low-fullbench-harness-audit"
RUN_ID="${RUN_ID:-ec2_full_benchmark_20260513}"
LOG_PATH="${EXPERIMENT_DIR}/${RUN_ID}.log"
RESOURCE_PATH="${EXPERIMENT_DIR}/${RUN_ID}_resource.tsv"
STATUS_PATH="${EXPERIMENT_DIR}/${RUN_ID}.status"
PID_PATH="${EXPERIMENT_DIR}/${RUN_ID}.pid"

descendants_of() {
  local frontier="$1"
  local all="$1"
  while [[ -n "${frontier// }" ]]; do
    local next=""
    for pid in $frontier; do
      if [[ -d "/proc/$pid" ]]; then
        local children
        children="$(pgrep -P "$pid" 2>/dev/null || true)"
        if [[ -n "$children" ]]; then
          next+=" ${children//$'\n'/ }"
          all+=" ${children//$'\n'/ }"
        fi
      fi
    done
    frontier="$next"
  done
  printf '%s\n' "$all"
}

sample_tree() {
  local root_pid="$1"
  local started_epoch="$2"
  local now elapsed pids rss_kib pss_kib threads proc_count
  now="$(date +%s)"
  elapsed=$((now - started_epoch))
  pids="$(descendants_of "$root_pid" | tr ' ' '\n' | awk 'NF && !seen[$1]++')"
  rss_kib=0
  pss_kib=0
  threads=0
  proc_count=0
  for pid in $pids; do
    [[ -d "/proc/$pid" ]] || continue
    proc_count=$((proc_count + 1))
    local rss one_pss one_threads
    rss="$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)"
    one_pss="$(awk '/^Pss:/ {print $2}' "/proc/$pid/smaps_rollup" 2>/dev/null || true)"
    one_threads="$(awk '/^Threads:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)"
    rss_kib=$((rss_kib + ${rss:-0}))
    pss_kib=$((pss_kib + ${one_pss:-0}))
    threads=$((threads + ${one_threads:-0}))
  done
  printf '%s\t%s\t%.1f\t%.1f\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$elapsed" \
    "$(awk -v kb="$rss_kib" 'BEGIN {print kb / 1024}')" \
    "$(awk -v kb="$pss_kib" 'BEGIN {print kb / 1024}')" \
    "$threads" \
    "$proc_count" \
    "$(awk '/MemAvailable:/ {printf "%.1f", $2 / 1024}' /proc/meminfo)" \
    "$(awk '/^SwapFree:/ {printf "%.1f", $2 / 1024}' /proc/meminfo)"
}

mkdir -p "$EXPERIMENT_DIR"
rm -f "$STATUS_PATH" "$PID_PATH"
printf 'timestamp_utc\telapsed_seconds\trss_mib\tpss_mib\tthreads\tprocesses\tmem_available_mib\tswap_free_mib\n' > "$RESOURCE_PATH"
{
  echo "run_id=${RUN_ID}"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "ulimit_nofile_before=$(ulimit -n)"
} > "$LOG_PATH"

started_epoch="$(date +%s)"
(
  set +e
  ulimit -n 65535 2>/dev/null || true
  echo "ulimit_nofile_after=$(ulimit -n)" >> "$LOG_PATH"
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}" \
    "${EXPERIMENT_DIR}/run_full_baselines_openrouter.sh" "$@" >> "$LOG_PATH" 2>&1
  code=$?
  {
    echo "exit_code=${code}"
    echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$STATUS_PATH"
  exit "$code"
) &
runner_pid=$!
echo "$runner_pid" > "$PID_PATH"

while kill -0 "$runner_pid" 2>/dev/null; do
  sample_tree "$runner_pid" "$started_epoch" >> "$RESOURCE_PATH" || true
  sleep "${SAMPLE_SECONDS:-10}"
done
sample_tree "$runner_pid" "$started_epoch" >> "$RESOURCE_PATH" || true

wait "$runner_pid"
