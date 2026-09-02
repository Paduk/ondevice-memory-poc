#!/usr/bin/env bash
set -uo pipefail

readonly REPO_ROOT="/home/hj153lee/PalmClaw"
readonly UBUNTU_ROOT="$REPO_ROOT/ubuntu"
readonly PYTHON_BIN="$REPO_ROOT/.conda/ubuntu-agent/bin/python"
readonly GENERATION_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid/hybrid-scale-s51-s100-state-evolution-r1"
readonly EVALUATION_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid-state-evolution-s51-s100-evaluation"

scenarios=($(seq 51 100))
completed_count="$(find "$GENERATION_ROOT/status" -type f -name 'scenario-*.status' -exec grep -l '^COMPLETED' {} + 2>/dev/null | wc -l)"
if [[ "$completed_count" -ne 50 ]]; then
  printf '[%s] Parallel evaluation aborted: only %s/50 completed\n' \
    "$(date -u +%FT%TZ)" "$completed_count" >&2
  exit 1
fi

mkdir -p "$EVALUATION_ROOT"
cd "$UBUNTU_ROOT"

run_update_audit() {
  printf '[%s] UPDATE/NO_OP Judge started\n' "$(date -u +%FT%TZ)"
  "$PYTHON_BIN" evaluation/experiment-scripts/run_vehiclemembench_v2_update_audit_judges.py \
    --evaluation-root /mnt/data/hj153lee/PalmClaw/evaluation \
    --output-root "$EVALUATION_ROOT/update-audit" \
    --methods hybrid \
    --scenarios "${scenarios[@]}" \
    --luna-model gpt-5.6-luna \
    --terra-model gpt-5.6-terra \
    --sol-model gpt-5.6-sol \
    --workers 8 \
    --max-attempts 2 \
    --timeout-seconds 600 \
    >"$EVALUATION_ROOT/update-audit.log" 2>&1
}

run_answerability() {
  printf '[%s] Answerability Judge started\n' "$(date -u +%FT%TZ)"
  "$PYTHON_BIN" evaluation/experiment-scripts/run_vehiclemembench_v2_answerability_audit.py \
    --evaluation-root /mnt/data/hj153lee/PalmClaw/evaluation \
    --output-root "$EVALUATION_ROOT" \
    --methods hybrid \
    --scenarios "${scenarios[@]}" \
    --model gpt-5.6-terra \
    --workers 4 \
    --batch-size 10 \
    --max-attempts 2 \
    --timeout-seconds 600 \
    >"$EVALUATION_ROOT/answerability.log" 2>&1
}

run_agent() {
  printf '[%s] Agent evaluation started\n' "$(date -u +%FT%TZ)"
  "$PYTHON_BIN" evaluation/experiment-scripts/run_vehiclemembench_v2_three_way_agent_eval.py \
    --dataset-root /home/hj153lee/VehicleMemBench \
    --evaluation-root /mnt/data/hj153lee/PalmClaw/evaluation \
    --output-root "$EVALUATION_ROOT" \
    --methods hybrid \
    --scenarios "${scenarios[@]}" \
    --model gpt-5.6-luna \
    --repeats 2 \
    --workers 8 \
    --task-attempts 2 \
    --timeout-seconds 300 \
    >"$EVALUATION_ROOT/agent.log" 2>&1
}

run_update_audit & update_pid=$!
run_answerability & answerability_pid=$!
run_agent & agent_pid=$!

failed=0
wait "$update_pid" || failed=1
wait "$answerability_pid" || failed=1
wait "$agent_pid" || failed=1

if ((failed)); then
  printf '[%s] Parallel evaluation finished with a failed stage\n' \
    "$(date -u +%FT%TZ)" >&2
  exit 1
fi
printf '[%s] Hybrid state-evolution S51-S100 parallel evaluation completed\n' \
  "$(date -u +%FT%TZ)"
