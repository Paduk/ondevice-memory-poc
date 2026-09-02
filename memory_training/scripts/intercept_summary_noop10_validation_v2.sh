#!/usr/bin/env bash
set -euo pipefail

run_id=qwen35-4b-summary-multitask-noop10-r1
training_session=qwen35-summary-mt10-gpu3
run_dir=/mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/$run_id
checkpoint=$run_dir/checkpoints/epoch-04
launcher=/home/hj153lee/PalmClaw/memory_training/scripts/run_checkpoint_validation_v2.sh

while [[ ! -s "$checkpoint/adapter/adapter_model.safetensors" ]] || \
      ! jq -e '.state == "VALIDATING" and .epoch == 4' "$run_dir/status.json" >/dev/null; do
  sleep 10
done

printf '%s\n' "Epoch 4 checkpoint is complete; stopping legacy Validation."
if tmux has-session -t "$training_session" 2>/dev/null; then
  tmux send-keys -t "$training_session":0.0 C-c
fi
sleep 10

while [[ ! -x "$launcher" ]]; do
  sleep 5
done

validation_session=qwen35-summary-noop10-validation-v2-gpu3
tmux kill-session -t "$validation_session" 2>/dev/null || true
tmux new-session -d -s "$validation_session" \
  "bash $launcher $run_id 3"
printf '%s\n' "Started $validation_session"
