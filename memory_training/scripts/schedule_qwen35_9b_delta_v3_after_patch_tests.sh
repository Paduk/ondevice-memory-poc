#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
patch_run_id=qwen35-9b-patch-multitask-noop5-grouped-v2-v1-10-e3-b1-r1
patch_root="${workspace}/runs/${patch_run_id}"
v2_progress="${patch_root}/test-hf-closed-loop-s86-s100-t12-t20-best-epoch-02/progress.json"
v1_cancel_dir="${patch_root}/test-hf-v1-s1-s50-best-epoch-02"
v1_cancel_progress="${v1_cancel_dir}/progress.json"
runner="${repo_root}/memory_training/scripts/run_qwen35_9b_delta_v3.sh"
gpu=2
log_path="${workspace}/qwen35-9b-delta-v3-e4-after-patch-tests-gpu2-scheduler.log"
patch_test_tmux=qwen35-9b-patch-tests-after-train-gpu2

exec > >(tee -a "${log_path}") 2>&1
echo "[$(date -Is)] Waiting for 9B Patch V2 Test to complete; V1 Test is cancelled."

read_status() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    "${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "UNKNOWN"))' "${path}"
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
    # The legacy test runner sees the temporary COMPLETED marker and skips V1.
    # Give it a moment to exit before removing the marker so no fake V1 result remains.
    for _ in {1..12}; do
      if ! tmux has-session -t "${patch_test_tmux}" 2>/dev/null; then
        break
      fi
      sleep 5
    done
    if tmux has-session -t "${patch_test_tmux}" 2>/dev/null; then
      tmux kill-session -t "${patch_test_tmux}"
    fi
    rm -f "${v1_cancel_progress}"
    rmdir "${v1_cancel_dir}" 2>/dev/null || true
    echo "[$(date -Is)] 9B Patch V2 Test completed; starting 9B Delta v3 E4 on GPU ${gpu}."
    exec bash "${runner}" "${gpu}" 4
  fi
  if [[ "${v2_state}" == FAILED ]]; then
    echo "[$(date -Is)] 9B Patch V2 Test failed; Delta v3 training will not start." >&2
    exit 1
  fi
  sleep 30
done
