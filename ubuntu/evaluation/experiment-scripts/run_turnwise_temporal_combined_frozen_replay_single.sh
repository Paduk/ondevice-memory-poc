#!/usr/bin/env bash
set -uo pipefail

readonly SCENARIO="${1:?scenario number is required}"
readonly PROJECT_ROOT="/home/hj153lee/PalmClaw/ubuntu"
readonly BENCHMARK_ROOT="/home/hj153lee/VehicleMemBench"
readonly PALMCLAW_BIN="/home/hj153lee/PalmClaw/.conda/ubuntu-agent/bin/palmclaw"
readonly EXPECTED_COMMIT="5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b"
readonly EVAL_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench"
readonly EXPERIMENT_ROOT="${EVAL_ROOT}/turnwise-temporal-vs-combined-frozen-replay-s${SCENARIO}-20260815"

case "${SCENARIO}" in
  1|2|4|5)
    temporal_cache="${EVAL_ROOT}/turnwise-temporal-patch-s${SCENARIO}-v1-20260814/cache"
    ;;
  3)
    temporal_cache="${EVAL_ROOT}/turnwise-temporal-vs-combined-fresh-r2-s3-20260815/temporal/cache"
    ;;
  6|7|8|9|10)
    temporal_cache="${EVAL_ROOT}/turnwise-temporal-patch-s6-s10-v1-20260814/cache"
    ;;
  *)
    echo "Scenario must be between 1 and 10" >&2
    exit 2
    ;;
esac

if [[ "${SCENARIO}" == "3" ]]; then
  combined_cache="${EVAL_ROOT}/turnwise-temporal-vs-combined-fresh-r2-s3-20260815/combined/cache"
elif [[ "${SCENARIO}" == "2" || "${SCENARIO}" == "6" ]]; then
  combined_cache="${EVAL_ROOT}/turnwise-temporal-compact-s${SCENARIO}-v1-20260814/cache"
else
  combined_cache="${EVAL_ROOT}/turnwise-temporal-compact-s${SCENARIO}-v1-20260815/cache"
fi

if [[ -e "${EXPERIMENT_ROOT}" ]]; then
  echo "Frozen replay root already exists: ${EXPERIMENT_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${temporal_cache}" || ! -d "${combined_cache}" ]]; then
  echo "Required ready cache is missing for Scenario ${SCENARIO}" >&2
  exit 1
fi

mkdir -p "${EXPERIMENT_ROOT}"
exec > >(tee -a "${EXPERIMENT_ROOT}/experiment.log") 2>&1
echo "$$" > "${EXPERIMENT_ROOT}/experiment.pid"

run_arm() {
  local arm="$1"
  local profile cache_root
  local -a compact_args=()
  if [[ "${arm}" == "temporal" ]]; then
    profile="cloud_turnwise_recursive_summary_patch_temporal"
    cache_root="${temporal_cache}"
  else
    profile="cloud_turnwise_recursive_summary_patch_temporal_compact"
    cache_root="${combined_cache}"
    compact_args=(
      --recursive-summary-compaction-adds 64
      --recursive-summary-compaction-tokens 1000
      --recursive-summary-compaction-target-ratio 0.70
    )
  fi

  local arm_root="${EXPERIMENT_ROOT}/${arm}"
  mkdir -p "${arm_root}/results" "${arm_root}/tmp"
  echo "[$(date --iso-8601=seconds)] frozen ${arm} Scenario ${SCENARIO} started"
  TMPDIR="${arm_root}/tmp" \
  PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS=2048 \
  PALMCLAW_MODEL_TIMEOUT_SECONDS=120 \
  PYTHONUNBUFFERED=1 \
  "${PALMCLAW_BIN}" eval vehicle \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --expected-commit "${EXPECTED_COMMIT}" \
    --mode live \
    --profiles "${profile}" \
    --scenario "${SCENARIO}" \
    --task-limit 10 \
    --max-tool-rounds 10 \
    --model gpt-5.6-terra \
    --memory-model gpt-5.6-luna \
    --embedding-model text-embedding-3-small \
    --memory-batch-tokens 8000 \
    "${compact_args[@]}" \
    --memory-cache-dir "${cache_root}" \
    --output-dir "${arm_root}/results"
  local status=$?
  if [[ ${status} -eq 0 ]]; then
    touch "${arm_root}/COMPLETED"
  else
    printf '%s\n' "${status}" > "${arm_root}/FAILED"
  fi
  find "${arm_root}/tmp" -depth -delete 2>/dev/null || true
  return "${status}"
}

overall_status=0
run_arm temporal || overall_status=1
run_arm combined || overall_status=1
if [[ ${overall_status} -eq 0 ]]; then
  touch "${EXPERIMENT_ROOT}/COMPLETED"
else
  printf '%s\n' "${overall_status}" > "${EXPERIMENT_ROOT}/FAILED"
fi
exit "${overall_status}"
