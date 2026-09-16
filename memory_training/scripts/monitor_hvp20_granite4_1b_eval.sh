#!/usr/bin/env bash
set -u

session=hvp20-granite1b-eval
root=/mnt/data/hj153lee/PalmClaw/on-device-memory-training/evaluations/hvp01-hvp20-external-adapted-v1/granite4-1b
interval_seconds="${MONITOR_INTERVAL_SECONDS:-180}"

report() {
  local method progress result evaluator_pid selected_gpu
  date -Is
  evaluator_pid=$(pgrep -f '[m]emory_training.evaluate_hf_closed_loop.*--scenarios 901' | head -1 || true)
  selected_gpu=""
  if [[ -n "$evaluator_pid" && -r "/proc/${evaluator_pid}/environ" ]]; then
    selected_gpu=$(tr '\0' '\n' <"/proc/${evaluator_pid}/environ" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
  fi
  if [[ "$selected_gpu" =~ ^[0-6]$ ]]; then
    printf 'ACTIVE_GPU=%s PID=%s\n' "$selected_gpu" "$evaluator_pid"
    nvidia-smi -i "$selected_gpu" \
      --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null || true
  else
    echo "STATUS=waiting_for_gpu_0_to_6"
    nvidia-smi \
      --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | sed -n '1,7p' || true
  fi
  for method in summary patch delta_v3; do
    progress="${root}/${method}/progress.json"
    result="${root}/${method}/summary.json"
    if [[ -s "$result" ]]; then
      printf 'METHOD=%s ' "$method"
      jq -c '{complete, scenarios: (.completed_scenarios | length), expected: (.expected_scenarios | length), esm: .closed_loop_quiz.esm, tool_f1: .closed_loop_quiz.tool_f1, arg_exact: .closed_loop_quiz.arg_exact}' "$result"
    elif [[ -s "$progress" ]]; then
      printf 'METHOD=%s ' "$method"
      jq -c '{status: "running", scenarios: (.completed_scenarios | length), expected: 20, updated_at}' "$progress"
    else
      echo "METHOD=${method} status=pending"
    fi
  done
}

while tmux has-session -t "$session" 2>/dev/null; do
  report
  sleep "$interval_seconds"
done

report
echo "MONITOR_TARGET_EXITED"
tail -25 "${root}/scheduler.log" 2>/dev/null || true
