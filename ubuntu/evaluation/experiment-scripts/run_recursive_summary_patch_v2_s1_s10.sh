#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="/home/hj153lee/PalmClaw/ubuntu"
readonly ARTIFACT_ROOT="${PALMCLAW_ARTIFACT_ROOT:-/mnt/data/hj153lee/PalmClaw}"
readonly BENCHMARK_ROOT="/home/hj153lee/VehicleMemBench"
readonly PALMCLAW_BIN="/home/hj153lee/PalmClaw/.conda/ubuntu-agent/bin/palmclaw"
readonly EXPERIMENT_ROOT="${ARTIFACT_ROOT}/evaluation/vehiclemembench/recursive-summary-patch-v2-s1-s10-20260812"
readonly CACHE_ROOT="${EXPERIMENT_ROOT}/cache"
readonly RESULTS_ROOT="${EXPERIMENT_ROOT}/results"
readonly EXPECTED_COMMIT="5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi
if [[ ! -x "${PALMCLAW_BIN}" ]]; then
  echo "PalmClaw executable is missing: ${PALMCLAW_BIN}" >&2
  exit 1
fi
if [[ -e "${EXPERIMENT_ROOT}" ]]; then
  echo "Fresh experiment root already exists: ${EXPERIMENT_ROOT}" >&2
  exit 1
fi

mkdir -p "${CACHE_ROOT}" "${RESULTS_ROOT}"
exec > >(tee -a "${EXPERIMENT_ROOT}/experiment.log") 2>&1

export PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS=2048
export PALMCLAW_MODEL_TIMEOUT_SECONDS=120
export PYTHONUNBUFFERED=1

echo "$$" > "${EXPERIMENT_ROOT}/experiment.pid"
echo "[$(date --iso-8601=seconds)] Recursive Summary Patch v2 S1-10 started"

set +e
"${PALMCLAW_BIN}" eval vehicle \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --mode live \
  --profiles cloud_recursive_summary_patch \
  --scenario 1 \
  --scenario-limit 10 \
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
  echo "[$(date --iso-8601=seconds)] experiment completed"
else
  printf '%s\n' "${status}" > "${EXPERIMENT_ROOT}/FAILED"
  echo "[$(date --iso-8601=seconds)] experiment failed status=${status}" >&2
fi
exit "${status}"
