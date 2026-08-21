#!/usr/bin/env bash
set -uo pipefail

readonly SCENARIO="${1:?scenario number is required}"
readonly BENCHMARK_ROOT="/home/hj153lee/VehicleMemBench"
readonly PALMCLAW_BIN="/home/hj153lee/PalmClaw/.conda/ubuntu-agent/bin/palmclaw"
readonly EXPECTED_COMMIT="5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b"
readonly ARTIFACT_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench"
readonly EXPERIMENT_ROOT="${ARTIFACT_ROOT}/turnwise-summary-r1-frozen-replay-s${SCENARIO}-20260815"

if (( SCENARIO >= 1 && SCENARIO <= 5 )); then
  readonly CACHE_ROOT="${ARTIFACT_ROOT}/turnwise-summary-s1-s5-20260813/cache"
elif (( SCENARIO == 6 )); then
  readonly CACHE_ROOT="${ARTIFACT_ROOT}/turnwise-s6-full-summary-20260813/cache"
elif (( SCENARIO >= 7 && SCENARIO <= 10 )); then
  readonly CACHE_ROOT="${ARTIFACT_ROOT}/turnwise-summary-s7-s10-20260813/cache"
else
  echo "Scenario must be between 1 and 10" >&2
  exit 2
fi

if [[ -e "${EXPERIMENT_ROOT}" ]]; then
  echo "Replay experiment root already exists: ${EXPERIMENT_ROOT}" >&2
  exit 1
fi

if ! find "${CACHE_ROOT}" -type f -name memory.db -path "*/scenario-$(printf '%02d' "${SCENARIO}")/*" -print -quit | rg -q .; then
  echo "R1 memory cache is missing for Scenario ${SCENARIO}" >&2
  exit 1
fi

mkdir -p "${EXPERIMENT_ROOT}/results" "${EXPERIMENT_ROOT}/tmp"
exec > >(tee -a "${EXPERIMENT_ROOT}/experiment.log") 2>&1
echo "$$" > "${EXPERIMENT_ROOT}/experiment.pid"
echo "[$(date --iso-8601=seconds)] Turn-wise Recursive Summary R1 frozen replay Scenario ${SCENARIO} started"

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
  --memory-cache-dir "${CACHE_ROOT}" \
  --output-dir "${EXPERIMENT_ROOT}/results"
status=$?

if [[ ${status} -eq 0 ]]; then
  touch "${EXPERIMENT_ROOT}/COMPLETED"
  echo "[$(date --iso-8601=seconds)] Scenario ${SCENARIO} replay completed"
else
  printf '%s\n' "${status}" > "${EXPERIMENT_ROOT}/FAILED"
  echo "[$(date --iso-8601=seconds)] Scenario ${SCENARIO} replay failed status=${status}" >&2
fi

find "${EXPERIMENT_ROOT}/tmp" -depth -delete 2>/dev/null || true
exit "${status}"
