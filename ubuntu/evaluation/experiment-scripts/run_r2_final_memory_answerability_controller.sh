#!/usr/bin/env bash
set -euo pipefail

readonly METHOD="${1:?summary or combined is required}"
readonly START_SCENARIO="${2:?start scenario is required}"
readonly END_SCENARIO="${3:?end scenario is required}"
readonly OUTPUT_DIR="${4:?output directory is required}"
readonly SOURCE_REPETITION="${5:-2}"
readonly SCENARIO_CSV="${6:-}"
readonly PROJECT_ROOT="/home/hj153lee/PalmClaw"
readonly PYTHON="${PROJECT_ROOT}/.conda/ubuntu-agent/bin/python"
readonly WORKER="${PROJECT_ROOT}/ubuntu/evaluation/experiment-scripts/audit_r2_combined_final_memory_answerability.py"
readonly ARTIFACT_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench"
readonly BENCHMARK_ROOT="/home/hj153lee/VehicleMemBench"
readonly MAX_PARALLEL=10

if [[ "${METHOD}" != "summary" && "${METHOD}" != "combined" ]]; then
  echo "Method must be summary or combined" >&2
  exit 2
fi
if (( START_SCENARIO < 1 || END_SCENARIO > 50 || START_SCENARIO > END_SCENARIO )); then
  echo "Invalid scenario range" >&2
  exit 2
fi
if [[ "${SOURCE_REPETITION}" != "1" && "${SOURCE_REPETITION}" != "2" ]]; then
  echo "Source repetition must be 1 or 2" >&2
  exit 2
fi

if [[ -n "${SCENARIO_CSV}" ]]; then
  IFS=',' read -r -a scenarios <<<"${SCENARIO_CSV}"
else
  mapfile -t scenarios < <(seq "${START_SCENARIO}" "${END_SCENARIO}")
fi

mkdir -p "${OUTPUT_DIR}/logs"
exec > >(tee -a "${OUTPUT_DIR}/controller.log") 2>&1
echo "[$(date --iso-8601=seconds)] ${METHOD} R${SOURCE_REPETITION} audit started"

running=0
worker_pids=()
for scenario in "${scenarios[@]}"; do
  if (( scenario < START_SCENARIO || scenario > END_SCENARIO )); then
    echo "Scenario outside configured range: ${scenario}" >&2
    exit 2
  fi
  destination="${OUTPUT_DIR}/scenario-$(printf '%02d' "${scenario}").json"
  if [[ -s "${destination}" ]]; then
    echo "skip existing ${destination}"
    continue
  fi
  "${PYTHON}" "${WORKER}" \
    --scenario "${scenario}" \
    --method "${METHOD}" \
    --source-repetition "${SOURCE_REPETITION}" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    >"${OUTPUT_DIR}/logs/scenario-$(printf '%02d' "${scenario}").log" 2>&1 &
  worker_pids+=("$!")
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then
    for worker_pid in "${worker_pids[@]}"; do
      wait "${worker_pid}"
    done
    worker_pids=()
    running=0
  fi
done
for worker_pid in "${worker_pids[@]}"; do
  wait "${worker_pid}"
done
touch "${OUTPUT_DIR}/COMPLETED"
echo "[$(date --iso-8601=seconds)] ${METHOD} completed"
