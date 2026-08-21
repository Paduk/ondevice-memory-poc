#!/usr/bin/env bash
set -euo pipefail

readonly METHOD="${1:?summary or combined is required}"
readonly SCENARIO_CSV="${2:?comma-separated scenarios are required}"
readonly PROJECT_ROOT="/home/hj153lee/PalmClaw"
readonly WORKER="${PROJECT_ROOT}/ubuntu/evaluation/experiment-scripts/run_turnwise_luna_regeneration_r3_single.sh"
readonly CONTROLLER_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench/turnwise-${METHOD}-luna-r3-regeneration-controller-20260818"
readonly MAX_PARALLEL=10

if [[ "${METHOD}" != "summary" && "${METHOD}" != "combined" ]]; then
  echo "Method must be summary or combined" >&2
  exit 2
fi
IFS=',' read -r -a scenarios <<<"${SCENARIO_CSV}"

mkdir -p "${CONTROLLER_ROOT}"
exec > >(tee -a "${CONTROLLER_ROOT}/controller.log") 2>&1
echo "[$(date --iso-8601=seconds)] ${METHOD} Luna R3 controller started"

worker_pids=()
for scenario in "${scenarios[@]}"; do
  bash "${WORKER}" "${METHOD}" "${scenario}" &
  worker_pids+=("$!")
  if (( ${#worker_pids[@]} >= MAX_PARALLEL )); then
    for worker_pid in "${worker_pids[@]}"; do
      wait "${worker_pid}"
    done
    worker_pids=()
  fi
done
for worker_pid in "${worker_pids[@]}"; do
  wait "${worker_pid}"
done

touch "${CONTROLLER_ROOT}/COMPLETED"
echo "[$(date --iso-8601=seconds)] ${METHOD} Luna R3 controller completed"
