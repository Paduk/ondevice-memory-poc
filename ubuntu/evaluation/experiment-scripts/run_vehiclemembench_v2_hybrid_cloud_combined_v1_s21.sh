#!/usr/bin/env bash
set -uo pipefail

readonly PROJECT_ROOT="/home/hj153lee/PalmClaw"
readonly PYTHON_BIN="${PROJECT_ROOT}/.conda/ubuntu-agent/bin/python"
readonly SCRIPT_ROOT="${PROJECT_ROOT}/ubuntu/evaluation/experiment-scripts"
readonly OUTPUT_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid-cloud-combined-v1-s21-20260824"
readonly PYTHON_PATH="${PROJECT_ROOT}:${PROJECT_ROOT}/ubuntu/src"
readonly TOKEN_CACHE="/mnt/nvme2/hj153lee/PalmClaw/tmp/data-gym-cache"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/tmp"
cd "${PROJECT_ROOT}" || exit 1

TMPDIR="${OUTPUT_ROOT}/tmp" TIKTOKEN_CACHE_DIR="${TOKEN_CACHE}" \
  PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHON_PATH}" \
  "${PYTHON_BIN}" \
  "${SCRIPT_ROOT}/run_vehiclemembench_v2_hybrid_cloud_memory.py" \
  --scenario 21 \
  --arm turnwise_combined \
  --output-root "${OUTPUT_ROOT}" \
  --model gpt-5.6-luna \
  --instruction-mode legacy \
  --no-redact-pii \
  --compaction-adds 64 \
  --compaction-tokens 1000 \
  --compaction-target-ratio 0.70 \
  --max-compaction-attempts 2 \
  >"${OUTPUT_ROOT}/logs/turnwise_combined-s21.log" 2>&1 || exit 1

TMPDIR="${OUTPUT_ROOT}/tmp" TIKTOKEN_CACHE_DIR="${TOKEN_CACHE}" \
  PYTHONUNBUFFERED=1 PYTHONPATH="${PYTHON_PATH}" \
  "${PYTHON_BIN}" \
  "${SCRIPT_ROOT}/run_vehiclemembench_v2_hybrid_cloud_gap_quiz.py" \
  --output-root "${OUTPUT_ROOT}" \
  --model gpt-5.6-luna \
  --arms turnwise_combined \
  --scenarios 21 \
  --repeats 1 \
  --workers 12 \
  >"${OUTPUT_ROOT}/logs/quiz-agent.log" 2>&1 || exit 1

touch "${OUTPUT_ROOT}/COMPLETED"
echo "completed S21 V1-style Turn-wise Combined memory and 40 Quiz tasks"
