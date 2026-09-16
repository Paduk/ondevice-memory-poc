#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-5}"
worker_pid="${2:-899539}"
run_tag="${3:-20260915-v1}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
runner="${repo_root}/memory_training/scripts/run_full_test24_update_stress_eval.sh"
campaign_root="${workspace}/benchmarks/test24-stress-composite-campaign-${run_tag}"
state="${campaign_root}/state/llama3.2-1b-delta_v3_compact_k10.state"
checkpoint="${workspace}/runs/llama3.2-1b-delta_v3_compact_k10-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03"
output_root="${workspace}/benchmarks/llama3.2-1b-delta_v3_compact_k10-test24-update-stress-${run_tag}/composite"

if [[ ! -d "${checkpoint}/adapter" ]]; then
  echo "Checkpoint adapter not found: ${checkpoint}/adapter" >&2
  exit 2
fi

all_complete() {
  local label summary
  for label in base update40 update60 update80; do
    summary="${output_root}/${label}/summary.json"
    if [[ ! -s "${summary}" ]] || \
       [[ "$(jq -r '.complete // false' "${summary}" 2>/dev/null)" != true ]] || \
       [[ "$(jq -r '(.completed_scenarios // []) | length' "${summary}" 2>/dev/null)" != 24 ]]; then
      return 1
    fi
  done
}

batch=8
quiz_batch=32
for attempt in 1 2 3 4; do
  printf 'RUNNING\t%s\tllama3.2-1b-delta_v3_compact_k10\trepair_attempt=%s\tcheckpoint=epoch-03\tbatch=%s\tquiz_batch=%s\t%s\n' \
    "${gpu}" "${attempt}" "${batch}" "${quiz_batch}" "$(date -Is)" >"${state}"
  echo "[$(date -Is)] Llama 3.2 1B k10 epoch-03 repair start attempt=${attempt} batch=${batch} quiz_batch=${quiz_batch}"

  if STRESS_ONLY=1 \
     INCLUDE_BASE=1 \
     COMPOSITE_ONLY=1 \
     SCENARIO_BATCH_SIZE="${batch}" \
     QUIZ_BATCH_SIZE="${quiz_batch}" \
       bash "${runner}" \
         "${gpu}" llama3.2-1b delta_v3_compact_k10 \
         "${checkpoint}" "${run_tag}" && all_complete; then
    printf 'COMPLETED\t%s\tllama3.2-1b-delta_v3_compact_k10\trepaired_epoch-03\tbase_u40_u60_u80\t%s\n' \
      "${gpu}" "$(date -Is)" >"${state}"
    echo "[$(date -Is)] Llama 3.2 1B k10 epoch-03 repair completed; resuming worker ${worker_pid}."
    kill -CONT "${worker_pid}"
    exit 0
  fi

  echo "[$(date -Is)] Llama 3.2 1B k10 epoch-03 repair failed attempt=${attempt}" >&2
  batch=$(((batch + 1) / 2))
  quiz_batch=$(((quiz_batch + 1) / 2))
  sleep 30
done

printf 'FAILED\t%s\tllama3.2-1b-delta_v3_compact_k10\trepair_attempts=4\tepoch-03\t%s\n' \
  "${gpu}" "$(date -Is)" >"${state}"
echo "Repair exhausted all attempts; worker ${worker_pid} remains stopped for inspection." >&2
exit 1
