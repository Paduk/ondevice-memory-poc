#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
patch_9b_run=qwen35-9b-patch-multitask-noop5-grouped-v2-v1-10-e3-b1-r1
patch_9b_root="${workspace}/runs/${patch_9b_run}"
v2_progress="${patch_9b_root}/test-hf-closed-loop-s86-s100-t12-t20-best-epoch-02/progress.json"
v1_cancel_dir="${patch_9b_root}/test-hf-v1-s1-s50-best-epoch-02"
v1_cancel_progress="${v1_cancel_dir}/progress.json"
patch_9b_test_tmux=qwen35-9b-patch-tests-after-train-gpu2
runner="${repo_root}/memory_training/scripts/run_qwen35_2b_patch.sh"
gpu=2
log_path="${workspace}/qwen35-2b-patch-e5-b4-trainseed46-after-9b-v2-gpu2-scheduler.log"

exec > >(tee -a "${log_path}") 2>&1
echo "[$(date -Is)] Waiting for 9B Patch V2 Test; its V1 Test remains cancelled."

read_status() {
  if [[ -f "$1" ]]; then
    "${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "UNKNOWN"))' "$1"
  else
    echo MISSING
  fi
}

while true; do
  v2_state="$(read_status "${v2_progress}")"
  if [[ "${v2_state}" == COMPLETED ]]; then
    while pgrep -f '[m]emory_training.evaluate_hf_.*qwen35-9b-patch-multitask' >/dev/null; do
      sleep 10
    done
    for _ in {1..12}; do
      if ! tmux has-session -t "${patch_9b_test_tmux}" 2>/dev/null; then
        break
      fi
      sleep 5
    done
    if tmux has-session -t "${patch_9b_test_tmux}" 2>/dev/null; then
      tmux kill-session -t "${patch_9b_test_tmux}"
    fi
    rm -f "${v1_cancel_progress}"
    rmdir "${v1_cancel_dir}" 2>/dev/null || true
    echo "[$(date -Is)] Starting 2B Patch E5 seed-46 repeat on GPU ${gpu}."
    exec bash "${runner}" "${gpu}" 5 4 4 46 r2
  fi
  if [[ "${v2_state}" == FAILED ]]; then
    echo "[$(date -Is)] 9B Patch V2 Test failed; repeat training will not start." >&2
    exit 1
  fi
  sleep 30
done
