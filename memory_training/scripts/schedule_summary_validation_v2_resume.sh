#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 RUN_ID TMUX_SESSION GPU SEED NOOP_PER_UPDATE EPOCH [step|epoch]" >&2
  exit 2
fi

run_id=$1
training_session=$2
gpu=$3
seed=$4
noop_per_update=$5
epoch=$6
resume_from=${7:-step}
run_dir="/mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/$run_id"
epoch_label=$(printf '%02d' "$epoch")
validation="$run_dir/validation-epoch-$epoch_label.json"
resume_script=/home/hj153lee/PalmClaw/memory_training/scripts/resume_summary_multitask_validation_v2.sh

while [[ ! -s "$validation" ]]; do
  sleep 30
done

# The artifact is written before best-checkpoint metadata is committed. Wait until
# training resumes, which guarantees the epoch checkpoint has been saved again.
while jq -e '.state == "VALIDATING"' "$run_dir/status.json" >/dev/null; do
  sleep 10
done

cp -n "$validation" "$run_dir/validation-epoch-$epoch_label.pre-closed-quiz.json"
if [[ "$resume_from" == "epoch" ]]; then
  checkpoint="$run_dir/checkpoints/epoch-$epoch_label"
elif [[ "$resume_from" == "step" ]]; then
  checkpoint=$(find "$run_dir/checkpoints" -maxdepth 1 -type d -name 'step-*' \
    | sort -V | tail -1)
else
  echo "Resume mode must be step or epoch, got: $resume_from" >&2
  exit 2
fi
if [[ -z "$checkpoint" || ! -d "$checkpoint" ]]; then
  echo "No step checkpoint is available before Epoch $epoch Validation" >&2
  exit 1
fi

tmux send-keys -t "$training_session":0.0 C-c
sleep 10
resume_command="bash $resume_script $run_id $gpu $seed $noop_per_update $checkpoint"
if tmux has-session -t "$training_session" 2>/dev/null; then
  tmux respawn-pane -k -t "$training_session":0.0 "$resume_command"
else
  tmux new-session -d -s "$training_session" "$resume_command"
fi
