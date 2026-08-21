#!/usr/bin/env bash
set -uo pipefail

readonly SCENARIO="${1:?scenario number is required}"
readonly PROJECT_ROOT="/home/hj153lee/PalmClaw/ubuntu"
readonly BENCHMARK_ROOT="/home/hj153lee/VehicleMemBench"
readonly PALMCLAW_BIN="/home/hj153lee/PalmClaw/.conda/ubuntu-agent/bin/palmclaw"
readonly EXPECTED_COMMIT="5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b"
readonly EXPERIMENT_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench/turnwise-temporal-vs-combined-fresh-r2-s${SCENARIO}-20260815"

if [[ -e "${EXPERIMENT_ROOT}" ]]; then
  echo "Fresh experiment root already exists: ${EXPERIMENT_ROOT}" >&2
  exit 1
fi

mkdir -p "${EXPERIMENT_ROOT}"
exec > >(tee -a "${EXPERIMENT_ROOT}/experiment.log") 2>&1
echo "$$" > "${EXPERIMENT_ROOT}/experiment.pid"

run_arm() {
  local arm="$1"
  local profile
  local -a compact_args=()
  if [[ "${arm}" == "temporal" ]]; then
    profile="cloud_turnwise_recursive_summary_patch_temporal"
  else
    profile="cloud_turnwise_recursive_summary_patch_temporal_compact"
    compact_args=(
      --recursive-summary-compaction-adds 64
      --recursive-summary-compaction-tokens 1000
      --recursive-summary-compaction-target-ratio 0.70
    )
  fi

  local arm_root="${EXPERIMENT_ROOT}/${arm}"
  mkdir -p "${arm_root}/cache" "${arm_root}/results" "${arm_root}/tmp"
  echo "[$(date --iso-8601=seconds)] ${arm} Scenario ${SCENARIO} started"
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
    --memory-cache-dir "${arm_root}/cache" \
    --output-dir "${arm_root}/results"
  local status=$?
  if [[ ${status} -eq 0 ]]; then
    touch "${arm_root}/COMPLETED"
    echo "[$(date --iso-8601=seconds)] ${arm} Scenario ${SCENARIO} completed"
  else
    printf '%s\n' "${status}" > "${arm_root}/FAILED"
    echo "[$(date --iso-8601=seconds)] ${arm} Scenario ${SCENARIO} failed status=${status}" >&2
  fi
  find "${arm_root}/tmp" -depth -delete 2>/dev/null || true
  return "${status}"
}

# Alternate order to reduce a systematic time/order bias between methods.
if (( SCENARIO % 2 == 1 )); then
  arms=(temporal combined)
else
  arms=(combined temporal)
fi

overall_status=0
for arm in "${arms[@]}"; do
  run_arm "${arm}" || overall_status=1
done

if [[ ${overall_status} -eq 0 ]]; then
  touch "${EXPERIMENT_ROOT}/COMPLETED"
else
  printf '%s\n' "${overall_status}" > "${EXPERIMENT_ROOT}/FAILED"
fi
exit "${overall_status}"
