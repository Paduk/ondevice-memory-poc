#!/usr/bin/env bash
set -euo pipefail

readonly PROFILE="${1:?profile is required}"
readonly SLUG="${2:?slug is required}"
readonly PROJECT_ROOT="/home/hj153lee/PalmClaw/ubuntu"
readonly ARTIFACT_ROOT="${PALMCLAW_ARTIFACT_ROOT:-/mnt/data/hj153lee/PalmClaw}"
readonly BENCHMARK_ROOT="/home/hj153lee/VehicleMemBench"
readonly PALMCLAW_BIN="/home/hj153lee/PalmClaw/.conda/ubuntu-agent/bin/palmclaw"
readonly EXPERIMENT_ROOT="${ARTIFACT_ROOT}/evaluation/vehiclemembench/turnwise-s6-full-${SLUG}-20260813"
readonly CACHE_ROOT="${EXPERIMENT_ROOT}/cache"
readonly RESULTS_ROOT="${EXPERIMENT_ROOT}/results"
readonly TEMP_ROOT="${EXPERIMENT_ROOT}/tmp"
readonly EXPECTED_COMMIT="5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b"

if [[ -e "${EXPERIMENT_ROOT}" || -e "${CACHE_ROOT}" ]]; then
  echo "Fresh experiment or cache root already exists for ${SLUG}" >&2
  exit 1
fi

mkdir -p "${CACHE_ROOT}" "${RESULTS_ROOT}" "${TEMP_ROOT}"
exec > >(tee -a "${EXPERIMENT_ROOT}/experiment.log") 2>&1

export PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS=2048
export PALMCLAW_MODEL_TIMEOUT_SECONDS=120
export PYTHONUNBUFFERED=1
export TMPDIR="${TEMP_ROOT}"

echo "$$" > "${EXPERIMENT_ROOT}/experiment.pid"
echo "[$(date --iso-8601=seconds)] ${PROFILE} Scenario 6 full run started"

set +e
"${PALMCLAW_BIN}" eval vehicle \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --mode live \
  --profiles "${PROFILE}" \
  --scenario 6 \
  --task-limit 10 \
  --max-tool-rounds 10 \
  --model gpt-5.6-terra \
  --memory-model gpt-5.6-luna \
  --embedding-model text-embedding-3-small \
  --memory-batch-tokens 8000 \
  --memory-cache-dir "${CACHE_ROOT}" \
  --output-dir "${RESULTS_ROOT}"
status=$?
set -e

if [[ ${status} -eq 0 ]]; then
  touch "${EXPERIMENT_ROOT}/COMPLETED"
  echo "[$(date --iso-8601=seconds)] ${PROFILE} Scenario 6 full run completed"
else
  printf '%s\n' "${status}" > "${EXPERIMENT_ROOT}/FAILED"
  echo "[$(date --iso-8601=seconds)] ${PROFILE} failed status=${status}" >&2
fi

find "${TEMP_ROOT}" -depth -delete 2>/dev/null || true
exit "${status}"
