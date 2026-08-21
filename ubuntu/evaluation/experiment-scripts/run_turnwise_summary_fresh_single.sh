#!/usr/bin/env bash
set -uo pipefail

readonly SCENARIO="${1:?scenario number is required}"
readonly REPETITION="${2:-1}"
readonly BENCHMARK_ROOT="/home/hj153lee/VehicleMemBench"
readonly PALMCLAW_BIN="/home/hj153lee/PalmClaw/.conda/ubuntu-agent/bin/palmclaw"
readonly EXPECTED_COMMIT="5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b"
readonly EXPERIMENT_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench/turnwise-summary-fresh-r${REPETITION}-s${SCENARIO}-20260815"

if (( SCENARIO < 21 || SCENARIO > 50 )); then
  echo "Scenario must be between 21 and 50" >&2
  exit 2
fi
if (( REPETITION < 1 )); then
  echo "Repetition must be positive" >&2
  exit 2
fi
if [[ -e "${EXPERIMENT_ROOT}" ]]; then
  echo "Fresh experiment root already exists: ${EXPERIMENT_ROOT}" >&2
  exit 1
fi

mkdir -p "${EXPERIMENT_ROOT}/cache" "${EXPERIMENT_ROOT}/results" "${EXPERIMENT_ROOT}/tmp"
exec > >(tee -a "${EXPERIMENT_ROOT}/experiment.log") 2>&1
echo "$$" > "${EXPERIMENT_ROOT}/experiment.pid"
echo "[$(date --iso-8601=seconds)] Turn-wise Summary Scenario ${SCENARIO} R${REPETITION} started"

TMPDIR="${EXPERIMENT_ROOT}/tmp" \
PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS=2048 \
PALMCLAW_MODEL_TIMEOUT_SECONDS=120 \
PYTHONUNBUFFERED=1 \
"${PALMCLAW_BIN}" eval vehicle \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --mode live \
  --profiles cloud_turnwise_recursive_summary \
  --scenario "${SCENARIO}" \
  --task-limit 10 \
  --max-tool-rounds 10 \
  --model gpt-5.6-terra \
  --memory-model gpt-5.6-luna \
  --embedding-model text-embedding-3-small \
  --memory-batch-tokens 8000 \
  --memory-cache-dir "${EXPERIMENT_ROOT}/cache" \
  --output-dir "${EXPERIMENT_ROOT}/results"
status=$?

if [[ ${status} -eq 0 ]]; then
  touch "${EXPERIMENT_ROOT}/COMPLETED"
  echo "[$(date --iso-8601=seconds)] Turn-wise Summary Scenario ${SCENARIO} R${REPETITION} completed"
else
  printf '%s\n' "${status}" > "${EXPERIMENT_ROOT}/FAILED"
  echo "[$(date --iso-8601=seconds)] Turn-wise Summary Scenario ${SCENARIO} R${REPETITION} failed status=${status}" >&2
fi

find "${EXPERIMENT_ROOT}/tmp" -depth -delete 2>/dev/null || true
exit "${status}"
