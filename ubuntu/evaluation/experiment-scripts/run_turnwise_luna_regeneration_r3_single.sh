#!/usr/bin/env bash
set -uo pipefail

readonly METHOD="${1:?summary or combined is required}"
readonly SCENARIO="${2:?scenario number is required}"
readonly BENCHMARK_ROOT="/home/hj153lee/VehicleMemBench"
readonly PALMCLAW_BIN="/home/hj153lee/PalmClaw/.conda/ubuntu-agent/bin/palmclaw"
readonly EXPECTED_COMMIT="5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b"
readonly EVAL_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench"
readonly EXPERIMENT_ROOT="${EVAL_ROOT}/turnwise-${METHOD}-fresh-r3-s${SCENARIO}-20260818"

if [[ "${METHOD}" != "summary" && "${METHOD}" != "combined" ]]; then
  echo "Method must be summary or combined" >&2
  exit 2
fi
if (( SCENARIO < 1 || SCENARIO > 50 )); then
  echo "Scenario must be between 1 and 50" >&2
  exit 2
fi
if [[ -e "${EXPERIMENT_ROOT}" ]]; then
  if [[ -f "${EXPERIMENT_ROOT}/COMPLETED" ]]; then
    echo "R3 experiment already completed: ${EXPERIMENT_ROOT}"
    exit 0
  fi
  if [[ -f "${EXPERIMENT_ROOT}/FAILED" ]]; then
    echo "R3 experiment already failed: ${EXPERIMENT_ROOT}" >&2
    exit 1
  fi
  if [[ -f "${EXPERIMENT_ROOT}/experiment.pid" ]]; then
    existing_pid="$(<"${EXPERIMENT_ROOT}/experiment.pid")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
      echo "Waiting for existing R3 experiment: ${EXPERIMENT_ROOT}"
      while kill -0 "${existing_pid}" 2>/dev/null; do
        sleep 10
      done
      if [[ -f "${EXPERIMENT_ROOT}/COMPLETED" ]]; then
        exit 0
      fi
      echo "Existing R3 experiment ended without COMPLETED: ${EXPERIMENT_ROOT}" >&2
      exit 1
    fi
  fi
  echo "Stale R3 experiment root exists: ${EXPERIMENT_ROOT}" >&2
  exit 1
fi

if [[ "${METHOD}" == "summary" ]]; then
  profile="cloud_turnwise_recursive_summary"
  compact_args=()
else
  profile="cloud_turnwise_recursive_summary_patch_temporal_compact"
  compact_args=(
    --recursive-summary-compaction-adds 64
    --recursive-summary-compaction-tokens 1000
    --recursive-summary-compaction-target-ratio 0.70
  )
fi

mkdir -p "${EXPERIMENT_ROOT}/cache" "${EXPERIMENT_ROOT}/results" "${EXPERIMENT_ROOT}/tmp"
exec > >(tee -a "${EXPERIMENT_ROOT}/experiment.log") 2>&1
echo "$$" > "${EXPERIMENT_ROOT}/experiment.pid"
echo "[$(date --iso-8601=seconds)] ${METHOD} Luna R3 S${SCENARIO} started"

TMPDIR="${EXPERIMENT_ROOT}/tmp" \
PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS=2048 \
PALMCLAW_MODEL_TIMEOUT_SECONDS=120 \
PYTHONUNBUFFERED=1 \
"${PALMCLAW_BIN}" eval vehicle \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --mode live \
  --profiles "${profile}" \
  --scenario "${SCENARIO}" \
  --task-limit 1 \
  --max-tool-rounds 10 \
  --model gpt-5.6-terra \
  --memory-model gpt-5.6-luna \
  --embedding-model text-embedding-3-small \
  --memory-batch-tokens 8000 \
  "${compact_args[@]}" \
  --memory-cache-dir "${EXPERIMENT_ROOT}/cache" \
  --output-dir "${EXPERIMENT_ROOT}/results"
status=$?

if [[ ${status} -eq 0 ]]; then
  touch "${EXPERIMENT_ROOT}/COMPLETED"
  echo "[$(date --iso-8601=seconds)] ${METHOD} Luna R3 S${SCENARIO} completed"
else
  printf '%s\n' "${status}" > "${EXPERIMENT_ROOT}/FAILED"
  echo "[$(date --iso-8601=seconds)] ${METHOD} Luna R3 S${SCENARIO} failed status=${status}" >&2
fi

find "${EXPERIMENT_ROOT}/tmp" -depth -delete 2>/dev/null || true
exit "${status}"
