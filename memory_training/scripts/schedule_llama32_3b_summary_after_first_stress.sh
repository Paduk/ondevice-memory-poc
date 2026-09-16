#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
pipeline="${repo_root}/memory_training/scripts/run_llama32_1b_patch_delta_k5_pipeline.sh"
stress_runner="${repo_root}/memory_training/scripts/run_llama32_3b_stress_test_once.sh"
stress_root="${workspace}/benchmarks/llama32-3b-stress-test-once-20260908-v1"

candidate_sessions=(
  llama32_3b_stress_k2_wait_gpu4_20260908
  llama32_3b_stress_k10_wait_gpu5_20260908
)
candidate_gpus=(4 5)
candidate_methods=(delta_v3_compact_k2 delta_v3_compact_k10)

stress_complete() {
  local method=$1 load
  for load in base 20 40 60 80; do
    [[ -s "${stress_root}/cache/${load}/${method}/on/summary.json" ]] || return 1
    [[ -s "${stress_root}/composite/${load}/${method}/summary.json" ]] || return 1
    [[ $(jq -r '.complete // false' "${stress_root}/composite/${load}/${method}/summary.json") == true ]] || return 1
  done
}

while true; do
  for index in 0 1; do
    session=${candidate_sessions[$index]}
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      gpu=${candidate_gpus[$index]}
      completed_method=${candidate_methods[$index]}
      if ! stress_complete "${completed_method}"; then
        echo "[$(date -Is)] ${session} ended before its Stress Test completed" >&2
        exit 1
      fi
      echo "[$(date -Is)] GPU ${gpu} released by ${completed_method}; starting Llama 3.2 3B Summary pipeline"
      "${pipeline}" "${gpu}" summary 4 45 r1 llama3.2-3b
      echo "[$(date -Is)] Llama 3.2 3B Summary Test complete; starting Stress Test"
      exec "${stress_runner}" "${gpu}" summary
    fi
  done
  sleep 30
done
