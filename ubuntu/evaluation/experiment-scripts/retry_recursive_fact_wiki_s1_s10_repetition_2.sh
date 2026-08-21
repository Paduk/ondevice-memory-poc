#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="/home/hj153lee/PalmClaw/ubuntu"
readonly ARTIFACT_ROOT="${PALMCLAW_ARTIFACT_ROOT:-/mnt/data/hj153lee/PalmClaw}"
readonly BENCHMARK_ROOT="/home/hj153lee/VehicleMemBench"
readonly PALMCLAW_BIN="/home/hj153lee/PalmClaw/.conda/ubuntu-agent/bin/palmclaw"
readonly EXPERIMENT_ROOT="${ARTIFACT_ROOT}/evaluation/vehiclemembench/fresh-recursive-fact-wiki-s1-s10-2x-20260810"
readonly RETRY_ROOT="${EXPERIMENT_ROOT}/repetition-2-retry-1"
readonly EXPECTED_COMMIT="5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi
if [[ -e "${RETRY_ROOT}" ]]; then
  echo "Fresh retry root already exists: ${RETRY_ROOT}" >&2
  exit 1
fi

mkdir -p "${RETRY_ROOT}/cache" "${RETRY_ROOT}/results"
exec > >(tee -a "${EXPERIMENT_ROOT}/experiment.log") 2>&1

export PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS=2048
export PALMCLAW_MODEL_TIMEOUT_SECONDS=120
export PYTHONUNBUFFERED=1

run_profile() {
  local profile="$1"
  local label="$2"

  echo "[$(date --iso-8601=seconds)] retry starting profile=${profile}"
  "${PALMCLAW_BIN}" eval vehicle \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --expected-commit "${EXPECTED_COMMIT}" \
    --mode live \
    --profiles "${profile}" \
    --scenario 1 \
    --scenario-limit 10 \
    --task-limit 10 \
    --max-tool-rounds 10 \
    --model gpt-5.6-terra \
    --memory-model gpt-5.6-luna \
    --embedding-model text-embedding-3-small \
    --memory-batch-tokens 8000 \
    --memory-cache-dir "${RETRY_ROOT}/cache/${label}" \
    --output-dir "${RETRY_ROOT}/results/${label}"
  echo "[$(date --iso-8601=seconds)] retry completed profile=${profile}"
}

run_profile "cloud_recursive_summary" "recursive-summary"
run_profile "cloud_post_normalized_fact_wiki" "fact-wiki"
touch "${EXPERIMENT_ROOT}/COMPLETED"
echo "[$(date --iso-8601=seconds)] experiment completed with clean repetition-2 retry"
