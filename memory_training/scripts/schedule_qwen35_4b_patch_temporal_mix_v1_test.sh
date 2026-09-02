#!/usr/bin/env bash
set -euo pipefail

variant="${1:?variant must be patch-r2 or temporal-patch-r1}"
gpu="${2:?GPU index is required}"

workspace="/mnt/data/hj153lee/PalmClaw/on-device-memory-training"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"
runner="/home/hj153lee/PalmClaw/memory_training/scripts/run_qwen35_4b_patch_temporal_mix_v1_test.sh"

case "${variant}" in
  patch-r2)
    run_id="qwen35-4b-patch-multitask-noop5-st-t1t10-r2"
    current_session="qwen35-patch-t1t10-test-scheduler-gpu5"
    ;;
  temporal-patch-r1)
    run_id="qwen35-4b-temporal-patch-multitask-noop5-st-t1t10-r1"
    current_session="qwen35-temporal-t1t10-test-scheduler-gpu4"
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

run_root="${workspace}/runs/${run_id}"
current_progress="${run_root}/test-hf-closed-loop-s86-s100-t12-t20-epoch04/progress.json"

echo "Waiting for ${run_id} S86-S100 + T12-T20 Test before V1 S1-S50."
missing_process_checks=0
while true; do
  state="MISSING"
  if [[ -f "${current_progress}" ]]; then
    state="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "UNKNOWN"))' "${current_progress}")"
  fi

  case "${state}" in
    COMPLETED)
      break
      ;;
    FAILED)
      echo "Current Test failed; V1 Test will not start: ${current_progress}" >&2
      exit 1
      ;;
  esac

  if tmux has-session -t "${current_session}" 2>/dev/null; then
    missing_process_checks=0
  else
    missing_process_checks=$((missing_process_checks + 1))
    if (( missing_process_checks >= 5 )); then
      echo "Current Test exited without COMPLETED status; V1 Test will not start." >&2
      exit 1
    fi
  fi
  sleep 30
done

while tmux has-session -t "${current_session}" 2>/dev/null; do
  sleep 10
done
while nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits \
  2>/dev/null | grep -Eq '[0-9]'; do
  sleep 30
done

echo "Current Test completed. Starting V1 S1-S50 on GPU ${gpu}."
exec "${runner}" "${variant}" "${gpu}" 04
