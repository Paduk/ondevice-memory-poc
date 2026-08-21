#!/usr/bin/env bash
set -uo pipefail

readonly SCENARIO="${1:?scenario number is required}"
readonly REPETITION="${2:?repetition number is required}"
readonly BENCHMARK_ROOT="/home/hj153lee/VehicleMemBench"
readonly PALMCLAW_BIN="/home/hj153lee/PalmClaw/.conda/ubuntu-agent/bin/palmclaw"
readonly EXPECTED_COMMIT="5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b"
readonly EVAL_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench"
readonly SOURCE_CACHE="${EVAL_ROOT}/turnwise-temporal-vs-combined-fresh-r2-s${SCENARIO}-20260815/combined/cache"
readonly EXPERIMENT_ROOT="${EVAL_ROOT}/turnwise-combined-r2-memory-sol-quiz-r${REPETITION}-s${SCENARIO}-20260818"

if (( SCENARIO < 1 || SCENARIO > 10 )); then
  echo "Scenario must be between 1 and 10" >&2
  exit 2
fi
if (( REPETITION < 1 || REPETITION > 2 )); then
  echo "Repetition must be 1 or 2" >&2
  exit 2
fi
if [[ ! -d "${SOURCE_CACHE}" ]] || ! find "${SOURCE_CACHE}" -name memory.db -print -quit | rg -q .; then
  echo "Completed Combined R2 cache is missing for Scenario ${SCENARIO}" >&2
  exit 1
fi
if [[ -e "${EXPERIMENT_ROOT}" ]]; then
  echo "SOL Quiz experiment root already exists: ${EXPERIMENT_ROOT}" >&2
  exit 1
fi

mkdir -p "${EXPERIMENT_ROOT}/cache" "${EXPERIMENT_ROOT}/results" "${EXPERIMENT_ROOT}/tmp"
cp -a --reflink=auto "${SOURCE_CACHE}/." "${EXPERIMENT_ROOT}/cache/"
exec > >(tee -a "${EXPERIMENT_ROOT}/experiment.log") 2>&1
echo "$$" > "${EXPERIMENT_ROOT}/experiment.pid"
echo "[$(date --iso-8601=seconds)] Combined R2 frozen-memory SOL Quiz R${REPETITION} Scenario ${SCENARIO} started"

TMPDIR="${EXPERIMENT_ROOT}/tmp" \
PALMCLAW_AGENT_MAX_OUTPUT_TOKENS=2048 \
PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS=2048 \
PALMCLAW_MODEL_TIMEOUT_SECONDS=300 \
PALMCLAW_PATCH_MEMORY_LEASE_SECONDS=360 \
PYTHONUNBUFFERED=1 \
"${PALMCLAW_BIN}" eval vehicle \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --mode live \
  --profiles cloud_turnwise_recursive_summary_patch_temporal_compact \
  --scenario "${SCENARIO}" \
  --task-limit 10 \
  --max-tool-rounds 10 \
  --model gpt-5.6-sol \
  --memory-model gpt-5.6-luna \
  --embedding-model text-embedding-3-small \
  --memory-batch-tokens 8000 \
  --recursive-summary-compaction-adds 64 \
  --recursive-summary-compaction-tokens 1000 \
  --recursive-summary-compaction-target-ratio 0.70 \
  --memory-cache-dir "${EXPERIMENT_ROOT}/cache" \
  --output-dir "${EXPERIMENT_ROOT}/results"
status=$?

if [[ ${status} -eq 0 ]]; then
  touch "${EXPERIMENT_ROOT}/COMPLETED"
  echo "[$(date --iso-8601=seconds)] R${REPETITION} Scenario ${SCENARIO} completed"
else
  printf '%s\n' "${status}" > "${EXPERIMENT_ROOT}/FAILED"
  echo "[$(date --iso-8601=seconds)] R${REPETITION} Scenario ${SCENARIO} failed status=${status}" >&2
fi

find "${EXPERIMENT_ROOT}/tmp" -depth -delete 2>/dev/null || true
exit "${status}"
