#!/usr/bin/env bash
set -euo pipefail

gpu_a="${1:-3}"
gpu_b="${2:-5}"
run_tag="${3:-20260915-v1}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
runner="${repo_root}/memory_training/scripts/run_full_test24_update_stress_eval.sh"
campaign_root="${workspace}/benchmarks/test24-stress-composite-campaign-${run_tag}"
state_root="${campaign_root}/state"
log_root="${campaign_root}/logs"
initial_session=granite1b-test24-stress-composite

mkdir -p "${state_root}" "${log_root}"
rm -f "${state_root}/campaign-complete.state"

method_complete() {
  local model=$1 method=$2 load summary
  for load in 40 60 80; do
    summary="${workspace}/benchmarks/${model}-${method}-test24-update-stress-${run_tag}/composite/update${load}/summary.json"
    if [[ ! -s "${summary}" ]] || \
       [[ "$(jq -r '.complete // false' "${summary}" 2>/dev/null)" != true ]]; then
      return 1
    fi
  done
}

run_job() {
  local gpu=$1 model=$2 method=$3 checkpoint=$4 batch=$5 quiz_batch=$6
  local include_base="${7:-0}"
  local job_id="${model}-${method}" attempt current_batch current_quiz_batch state
  state="${state_root}/${job_id}.state"
  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "Checkpoint adapter not found: ${checkpoint}/adapter" >&2
    printf 'FAILED\t%s\t%s\tmissing_checkpoint\t%s\n' \
      "${gpu}" "${job_id}" "$(date -Is)" >"${state}"
    return 1
  fi
  if method_complete "${model}" "${method}"; then
    printf 'COMPLETED\t%s\t%s\treused\t%s\n' "${gpu}" "${job_id}" "$(date -Is)" >"${state}"
    return 0
  fi

  current_batch=${batch}
  current_quiz_batch=${quiz_batch}
  for attempt in 1 2 3 4; do
    printf 'RUNNING\t%s\t%s\tattempt=%s\tbatch=%s\tquiz_batch=%s\t%s\n' \
      "${gpu}" "${job_id}" "${attempt}" "${current_batch}" \
      "${current_quiz_batch}" "$(date -Is)" >"${state}"
    echo "[$(date -Is)] start GPU=${gpu} job=${job_id} attempt=${attempt} batch=${current_batch} quiz_batch=${current_quiz_batch}"
    if STRESS_ONLY=1 \
       COMPOSITE_ONLY=1 \
       INCLUDE_BASE="${include_base}" \
       SCENARIO_BATCH_SIZE="${current_batch}" \
       QUIZ_BATCH_SIZE="${current_quiz_batch}" \
         bash "${runner}" \
           "${gpu}" "${model}" "${method}" "${checkpoint}" "${run_tag}"; then
      printf 'COMPLETED\t%s\t%s\tattempt=%s\t%s\n' \
        "${gpu}" "${job_id}" "${attempt}" "$(date -Is)" >"${state}"
      return 0
    fi
    echo "[$(date -Is)] failed GPU=${gpu} job=${job_id} attempt=${attempt}" >&2
    current_batch=$(((current_batch + 1) / 2))
    current_quiz_batch=$(((current_quiz_batch + 1) / 2))
    sleep 30
  done
  printf 'FAILED\t%s\t%s\tattempts=4\t%s\n' \
    "${gpu}" "${job_id}" "$(date -Is)" >"${state}"
  return 1
}

wait_for_initial_branch() {
  local model=$1 first_method=$2 second_method=$3
  while tmux has-session -t "${initial_session}" 2>/dev/null; do
    if method_complete "${model}" "${first_method}" && \
       method_complete "${model}" "${second_method}"; then
      return 0
    fi
    sleep 60
  done
}

worker_patch() {
  local failed=0
  wait_for_initial_branch granite4-1b patch delta_v3_compact_k5
  run_job "${gpu_a}" granite4-1b patch \
    "${workspace}/runs/granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03" 8 32 || failed=1
  run_job "${gpu_a}" granite4-350m patch \
    "${workspace}/runs/granite4-350m-patch-multitask-noop5-full6val-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-04" 12 48 || failed=1
  run_job "${gpu_a}" qwen3.5-0.8b patch \
    "${workspace}/runs/qwen35-0.8b-patch-multitask-noop5-grouped-v2-v1-10-e4-b8-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03" 10 40 || failed=1
  run_job "${gpu_a}" qwen3.5-2b patch \
    "${workspace}/runs/qwen35-2b-patch-multitask-noop5-grouped-v2-v1-10-e5-b4-r1/checkpoints/epoch-03" 8 32 || failed=1
  run_job "${gpu_a}" llama3.2-1b patch \
    "${workspace}/runs/llama3.2-1b-patch-multitask-noop5-e4-b8-trainseed45-evalfixed-noop5-r2/checkpoints/epoch-03" 8 32 || failed=1
  run_job "${gpu_a}" llama3.2-3b patch \
    "${workspace}/runs/llama3.2-3b-patch-multitask-noop5-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-04" 6 24 || failed=1
  return "${failed}"
}

worker_k5() {
  local failed=0
  wait_for_initial_branch granite4-1b patch delta_v3_compact_k5
  run_job "${gpu_a}" granite4-1b delta_v3_compact_k5 \
    "${workspace}/runs/granite4-1b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1/checkpoints/epoch-04" 8 32 || failed=1
  run_job "${gpu_a}" granite4-350m delta_v3_compact_k5 \
    "${workspace}/runs/granite4-350m-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1/checkpoints/epoch-03" 12 48 || failed=1
  run_job "${gpu_a}" qwen3.5-0.8b delta_v3_compact_k5 \
    "${workspace}/runs/qwen35-0.8b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03" 10 40 || failed=1
  run_job "${gpu_a}" qwen3.5-2b delta_v3_compact_k5 \
    "${workspace}/runs/qwen35-2b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b2-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-04" 8 32 || failed=1
  run_job "${gpu_a}" llama3.2-1b delta_v3_compact_k5 \
    "${workspace}/runs/llama3.2-1b-delta_v3_compact_k5-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r2/checkpoints/epoch-04" 8 32 || failed=1
  run_job "${gpu_a}" llama3.2-3b delta_v3_compact_k5 \
    "${workspace}/runs/llama3.2-3b-delta_v3_compact_k5-multitask-noop5-e4-b1-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-04" 6 24 || failed=1
  return "${failed}"
}

worker_k2() {
  local failed=0
  wait_for_initial_branch granite4-1b delta_v3_compact_k2 delta_v3_compact_k10
  run_job "${gpu_b}" granite4-1b delta_v3_compact_k2 \
    "${workspace}/runs/granite4-1b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1/checkpoints/epoch-03" 8 32 || failed=1
  run_job "${gpu_b}" granite4-350m delta_v3_compact_k2 \
    "${workspace}/runs/granite4-350m-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1/checkpoints/epoch-03" 12 48 || failed=1
  run_job "${gpu_b}" qwen3.5-0.8b delta_v3_compact_k2 \
    "${workspace}/runs/qwen35-0.8b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03" 10 40 || failed=1
  run_job "${gpu_b}" qwen3.5-2b delta_v3_compact_k2 \
    "${workspace}/runs/qwen35-2b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b2-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03" 8 32 || failed=1
  run_job "${gpu_b}" llama3.2-1b delta_v3_compact_k2 \
    "${workspace}/runs/llama3.2-1b-delta_v3_compact_k2-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03" 8 32 || failed=1
  run_job "${gpu_b}" llama3.2-3b delta_v3_compact_k2 \
    "${workspace}/runs/llama3.2-3b-delta_v3_compact_k2-multitask-noop5-e4-b1-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03" 6 24 || failed=1
  return "${failed}"
}

worker_k10() {
  local failed=0
  wait_for_initial_branch granite4-1b delta_v3_compact_k2 delta_v3_compact_k10
  run_job "${gpu_b}" granite4-1b delta_v3_compact_k10 \
    "${workspace}/runs/granite4-1b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1/checkpoints/epoch-04" 8 32 || failed=1
  run_job "${gpu_b}" granite4-350m delta_v3_compact_k10 \
    "${workspace}/runs/granite4-350m-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1/checkpoints/epoch-04" 12 48 || failed=1
  run_job "${gpu_b}" qwen3.5-0.8b delta_v3_compact_k10 \
    "${workspace}/runs/qwen35-0.8b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03" 10 40 || failed=1
  run_job "${gpu_b}" qwen3.5-2b delta_v3_compact_k10 \
    "${workspace}/runs/qwen35-2b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b2-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-04" 8 32 || failed=1
  run_job "${gpu_b}" llama3.2-1b delta_v3_compact_k10 \
    "${workspace}/runs/llama3.2-1b-delta_v3_compact_k10-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03" 8 32 1 || failed=1
  run_job "${gpu_b}" llama3.2-3b delta_v3_compact_k10 \
    "${workspace}/runs/llama3.2-3b-delta_v3_compact_k10-multitask-noop5-e4-b1-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-04" 6 24 || failed=1
  return "${failed}"
}

worker_patch >"${log_root}/gpu${gpu_a}-patch.log" 2>&1 &
pid_patch=$!
worker_k5 >"${log_root}/gpu${gpu_a}-k5.log" 2>&1 &
pid_k5=$!
worker_k2 >"${log_root}/gpu${gpu_b}-k2.log" 2>&1 &
pid_k2=$!
worker_k10 >"${log_root}/gpu${gpu_b}-k10.log" 2>&1 &
pid_k10=$!

status=0
wait "${pid_patch}" || status=1
wait "${pid_k5}" || status=1
wait "${pid_k2}" || status=1
wait "${pid_k10}" || status=1
printf '%s\t%s\n' "${status}" "$(date -Is)" >"${state_root}/campaign-complete.state"
exit "${status}"
