#!/usr/bin/env bash
set -euo pipefail

variant="${1:?variant must be patch-r2 or temporal-patch-r1}"
gpu="${2:?GPU index is required}"

workspace="/mnt/data/hj153lee/PalmClaw/on-device-memory-training"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"
test_script="/home/hj153lee/PalmClaw/memory_training/scripts/run_qwen35_4b_patch_temporal_mix_test.sh"

case "${variant}" in
  patch-r2)
    run_id="qwen35-4b-patch-multitask-noop5-st-t1t10-r2"
    training_session="qwen35-patch-t1t10-r2-gpu5"
    ;;
  temporal-patch-r1)
    run_id="qwen35-4b-temporal-patch-multitask-noop5-st-t1t10-r1"
    training_session="qwen35-temporal-t1t10-r1-gpu4"
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

run_root="${workspace}/runs/${run_id}"
status_path="${run_root}/status.json"
checkpoint="${run_root}/checkpoints/epoch-04"

echo "Waiting for ${run_id} to complete before Test on GPU ${gpu}."
while true; do
  state="MISSING"
  if [[ -f "${status_path}" ]]; then
    state="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", "UNKNOWN"))' "${status_path}")"
  fi

  case "${state}" in
    COMPLETED)
      if [[ ! -d "${checkpoint}" ]]; then
        echo "Training completed but checkpoint is missing: ${checkpoint}" >&2
        exit 1
      fi
      while tmux has-session -t "${training_session}" 2>/dev/null; do
        sleep 10
      done
      echo "Training completed. Starting S86-S100 + T12-T20 Test."
      exec "${test_script}" "${variant}" "${gpu}" 04
      ;;
    FAILED)
      echo "Training failed; Test will not start. See ${status_path}" >&2
      exit 1
      ;;
  esac

  sleep 30
done
