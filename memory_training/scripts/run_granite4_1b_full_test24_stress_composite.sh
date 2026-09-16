#!/usr/bin/env bash
set -euo pipefail

gpu_a="${1:-3}"
gpu_b="${2:-5}"
run_tag="${3:-20260915-v1}"

if [[ "${gpu_a}" == "${gpu_b}" ]]; then
  echo "Two different GPU indices are required." >&2
  exit 2
fi

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
runner="${repo_root}/memory_training/scripts/run_full_test24_update_stress_eval.sh"
log_root="${workspace}/benchmarks/granite4-1b-test24-update-stress-${run_tag}/logs"

patch_checkpoint="${workspace}/runs/granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"
k2_checkpoint="${workspace}/runs/granite4-1b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1/checkpoints/epoch-03"
k5_checkpoint="${workspace}/runs/granite4-1b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1/checkpoints/epoch-04"
k10_checkpoint="${workspace}/runs/granite4-1b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1/checkpoints/epoch-04"

for checkpoint in \
  "${patch_checkpoint}" "${k2_checkpoint}" "${k5_checkpoint}" "${k10_checkpoint}"; do
  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "Checkpoint adapter not found: ${checkpoint}/adapter" >&2
    exit 2
  fi
done

mkdir -p "${log_root}"

run_method() {
  local gpu=$1
  local method=$2
  local checkpoint=$3
  STRESS_ONLY=1 \
  COMPOSITE_ONLY=1 \
  SCENARIO_BATCH_SIZE=2 \
  QUIZ_BATCH_SIZE=8 \
    bash "${runner}" \
      "${gpu}" granite4-1b "${method}" "${checkpoint}" "${run_tag}"
}

run_gpu_a() {
  run_method "${gpu_a}" patch "${patch_checkpoint}"
  run_method "${gpu_a}" delta_v3_compact_k5 "${k5_checkpoint}"
}

run_gpu_b() {
  run_method "${gpu_b}" delta_v3_compact_k2 "${k2_checkpoint}"
  run_method "${gpu_b}" delta_v3_compact_k10 "${k10_checkpoint}"
}

run_gpu_a >"${log_root}/gpu${gpu_a}-patch-k5.log" 2>&1 &
pid_a=$!
run_gpu_b >"${log_root}/gpu${gpu_b}-k2-k10.log" 2>&1 &
pid_b=$!

status=0
if ! wait "${pid_a}"; then
  status=1
fi
if ! wait "${pid_b}"; then
  status=1
fi
if [[ "${status}" -ne 0 ]]; then
  echo "One or more Granite 1B Test24 stress Composite workers failed." >&2
  exit "${status}"
fi

echo "[$(date -Is)] Granite 1B Test24 U40/U60/U80 Composite complete."
