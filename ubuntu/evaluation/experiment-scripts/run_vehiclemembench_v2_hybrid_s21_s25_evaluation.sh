#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/home/hj153lee/PalmClaw"
readonly UBUNTU_ROOT="$REPO_ROOT/ubuntu"
readonly PYTHON_BIN="$REPO_ROOT/.conda/ubuntu-agent/bin/python"
readonly GENERATION_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid/hybrid-scale-s21-s25-state-evolution-r1"
readonly EVALUATION_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid-state-evolution-s21-s25-evaluation"
readonly GENERATION_SESSION="v2-hybrid-s21-s25-controller"
readonly RECOVERY_SESSION="v2-hybrid-s21-s25-recovery"

scenarios=($(seq 21 25))

while tmux has-session -t "$GENERATION_SESSION" 2>/dev/null; do
  sleep 60
done
while tmux has-session -t "$RECOVERY_SESSION" 2>/dev/null; do
  sleep 60
done

completed_count="$(find "$GENERATION_ROOT/status" -type f -name 'scenario-*.status' -exec grep -l '^COMPLETED' {} + 2>/dev/null | wc -l)"
if [[ "$completed_count" -ne 5 ]]; then
  printf '[%s] Evaluation aborted: only %s/5 scenarios completed\n' \
    "$(date -u +%FT%TZ)" "$completed_count" >&2
  exit 1
fi

mkdir -p "$EVALUATION_ROOT"
cd "$UBUNTU_ROOT"

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
  --timeout-seconds 600

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
  --timeout-seconds 600

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
  --timeout-seconds 300

printf '[%s] Hybrid state-evolution S21-S25 evaluation completed\n' \
  "$(date -u +%FT%TZ)"
