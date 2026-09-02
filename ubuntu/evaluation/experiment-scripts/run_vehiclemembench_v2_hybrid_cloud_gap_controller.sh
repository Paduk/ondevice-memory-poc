#!/usr/bin/env bash
set -uo pipefail

readonly PROJECT_ROOT="/home/hj153lee/PalmClaw"
readonly PYTHON_BIN="${PROJECT_ROOT}/.conda/ubuntu-agent/bin/python"
readonly SCRIPT_ROOT="${PROJECT_ROOT}/ubuntu/evaluation/experiment-scripts"
readonly OUTPUT_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid-cloud-memory-gap"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/tmp"
cd "${PROJECT_ROOT}" || exit 1

declare -a pids=()
declare -a labels=()
for arm in turnwise_summary turnwise_combined; do
  for scenario in 1 2 3 4 5; do
    label="${arm}-s${scenario}"
    log="${OUTPUT_ROOT}/logs/${label}.log"
    TMPDIR="${OUTPUT_ROOT}/tmp" PYTHONUNBUFFERED=1 PYTHONPATH=ubuntu/src \
      "${PYTHON_BIN}" \
      "${SCRIPT_ROOT}/run_vehiclemembench_v2_hybrid_cloud_memory.py" \
      --scenario "${scenario}" \
      --arm "${arm}" \
      --model gpt-5.6-luna \
      >"${log}" 2>&1 &
    pids+=("$!")
    labels+=("${label}")
  done
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed ${labels[$index]}"
  else
    echo "failed ${labels[$index]}" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 1
fi

TMPDIR="${OUTPUT_ROOT}/tmp" PYTHONUNBUFFERED=1 PYTHONPATH=ubuntu/src \
  "${PYTHON_BIN}" \
  "${SCRIPT_ROOT}/run_vehiclemembench_v2_hybrid_cloud_gap_quiz.py" \
  --model gpt-5.6-luna \
  --repeats 2 \
  --workers 10 \
  >"${OUTPUT_ROOT}/logs/quiz-agent.log" 2>&1 || exit 1

PYTHONPATH=ubuntu/src "${PYTHON_BIN}" \
  "${SCRIPT_ROOT}/summarize_vehiclemembench_v2_hybrid_cloud_memory_gap.py" \
  >"${OUTPUT_ROOT}/logs/summary.log" 2>&1 || exit 1

echo "completed Hybrid Gold vs Turn-wise Summary/Combined evaluation"
