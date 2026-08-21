#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="/home/hj153lee/PalmClaw/ubuntu"
readonly ARTIFACT_ROOT="${PALMCLAW_ARTIFACT_ROOT:-/mnt/data/hj153lee/PalmClaw}"
readonly BENCHMARK_ROOT="/home/hj153lee/VehicleMemBench"
readonly PALMCLAW_BIN="/home/hj153lee/PalmClaw/.conda/ubuntu-agent/bin/palmclaw"
readonly EXPERIMENT_ROOT="${ARTIFACT_ROOT}/evaluation/vehiclemembench/fresh-recursive-fact-wiki-s1-s10-2x-20260810"
readonly EXPECTED_COMMIT="5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi
if [[ ! -x "${PALMCLAW_BIN}" ]]; then
  echo "PalmClaw executable is missing: ${PALMCLAW_BIN}" >&2
  exit 1
fi
mkdir -p "${EXPERIMENT_ROOT}/cache" "${EXPERIMENT_ROOT}/results"
exec > >(tee -a "${EXPERIMENT_ROOT}/experiment.log") 2>&1

export PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS=2048
export PALMCLAW_MODEL_TIMEOUT_SECONDS=120
export PYTHONUNBUFFERED=1

run_profile() {
  local repetition="$1"
  local profile="$2"
  local label="$3"
  local cache_dir="${EXPERIMENT_ROOT}/cache/repetition-${repetition}/${label}"
  local output_dir="${EXPERIMENT_ROOT}/results/repetition-${repetition}/${label}"
  local manifest=""
  local run_id=""
  local run_status=""
  local -a resume_args=()

  if [[ -d "${output_dir}" ]]; then
    manifest="$(find "${output_dir}" -mindepth 2 -maxdepth 2 -type f -name manifest.json -print -quit)"
  fi
  if [[ -n "${manifest}" ]]; then
    run_id="$(jq -r '.run_id' "${manifest}")"
    run_status="$(jq -r '.status' "${manifest}")"
    if [[ "${run_status}" == "completed" ]]; then
      echo "[$(date --iso-8601=seconds)] skipping completed repetition=${repetition} profile=${profile} run_id=${run_id}"
      return 0
    fi
    echo "[$(date --iso-8601=seconds)] resuming repetition=${repetition} profile=${profile} run_id=${run_id} status=${run_status}"
    resume_args=(--resume-run "${run_id}")
  elif [[ -e "${cache_dir}" ]]; then
    echo "Cache exists without a resumable run: ${cache_dir}" >&2
    return 1
  fi

  mkdir -p "${output_dir}"
  echo "[$(date --iso-8601=seconds)] starting repetition=${repetition} profile=${profile}"
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
    --memory-cache-dir "${cache_dir}" \
    --output-dir "${output_dir}" \
    "${resume_args[@]}"
  echo "[$(date --iso-8601=seconds)] completed repetition=${repetition} profile=${profile}"
}

echo "[$(date --iso-8601=seconds)] experiment started"
for repetition in 1 2; do
  run_profile "${repetition}" "cloud_recursive_summary" "recursive-summary"
  run_profile "${repetition}" "cloud_post_normalized_fact_wiki" "fact-wiki"
done
echo "[$(date --iso-8601=seconds)] experiment completed"
touch "${EXPERIMENT_ROOT}/COMPLETED"
