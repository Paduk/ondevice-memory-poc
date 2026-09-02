#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
model="${2:?Model key is required}"
run_id="${3:?Run ID is required}"
epochs="${4:-4}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-plus-v1-style-20-v1
run_dir="${workspace}/runs/${run_id}"
log_path="${run_dir}/v1style20-post.log"

cd "${repo_root}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"
exec > >(tee -a "${log_path}") 2>&1

validate_epoch() {
  local epoch_number=$1
  local epoch
  epoch=$(printf '%02d' "${epoch_number}")
  local checkpoint="${run_dir}/checkpoints/epoch-${epoch}"
  local output="${run_dir}/validation-v2-candidate-epoch-${epoch}.json"
  local epoch_log="${run_dir}/validation-v2-candidate-epoch-${epoch}.log"
  test -d "${checkpoint}/adapter"
  if [[ -s "${output}" ]]; then
    echo "[$(date -Is)] Validation epoch ${epoch} already exists; skipping."
    return 0
  fi
  echo "[$(date -Is)] Starting full Validation epoch ${epoch} on GPU ${gpu}."
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
    -m memory_training.validate_hf_checkpoint_v2 \
    --run-id "${run_id}" \
    --checkpoint "${checkpoint}" \
    --output "${output}" \
    --workspace "${workspace}" \
    --data-root "${data_root}" \
    --validation-scenarios 81 82 83 84 85 111 \
    --full-scenarios 81 82 83 84 85 111 \
    --closed-loop-only \
    --scenario-batch-size 3 \
    --quiz-batch-size 16 \
    >"${epoch_log}" 2>&1
}

echo "[$(date -Is)] Starting all-epoch pairwise Validation for ${run_id}."
declare -A running=()
next=1
failed=0
while (( next <= epochs || ${#running[@]} > 0 )); do
  while (( next <= epochs && ${#running[@]} < 2 )); do
    validate_epoch "${next}" &
    running[$!]="${next}"
    next=$((next + 1))
  done
  reaped=0
  for pid in "${!running[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      epoch_number="${running[$pid]}"
      if ! wait "${pid}"; then
        echo "[$(date -Is)] Validation epoch ${epoch_number} failed." >&2
        failed=1
      fi
      unset 'running[$pid]'
      reaped=1
    fi
  done
  if (( failed )); then
    for pid in "${!running[@]}"; do kill "${pid}" 2>/dev/null || true; done
    exit 1
  fi
  if (( reaped == 0 )); then sleep 2; fi
done

"${python_bin}" -m memory_training.validation_candidates full \
  --run-dir "${run_dir}" \
  --output "${run_dir}/best-full-validation-checkpoint.json"

winner=$(jq -r '.winner.checkpoint' "${run_dir}/best-full-validation-checkpoint.json")
winner_epoch=$(basename "${winner}")
test_output="${run_dir}/test-hf-v1-s1-s50-best-${winner_epoch}"

echo "[$(date -Is)] Starting V1 S1-S50 Test with ${winner_epoch} on GPU ${gpu}."
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m memory_training.evaluate_hf_v1 \
  --model "${model}" \
  --method patch \
  --checkpoint "${winner}" \
  --output-dir "${test_output}" \
  --workspace "${workspace}" \
  --data-root "${data_root}" \
  --vehiclemembench-root /home/hj153lee/VehicleMemBench \
  --scenarios {1..50} \
  --max-length 2048 \
  --max-new-tokens 768 \
  --quiz-max-new-tokens 256 \
  --scenario-batch-size 16 \
  --quiz-batch-size 16 \
  --checkpoint-interval 50

"${python_bin}" - "${run_dir}" "${winner}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
winner = sys.argv[2]
selection = json.loads((run_dir / "best-full-validation-checkpoint.json").read_text())
status = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "state": "COMPLETED",
    "run_id": run_dir.name,
    "final_state": "COMPLETED",
    "best_checkpoint": winner,
    "best_score": selection["winner"]["full_esm"],
    "best_metric": "full_validation_esm_then_final_state_f1",
    "external_validation": "best-full-validation-checkpoint.json",
    "v1_test": f"test-hf-v1-s1-s50-best-{Path(winner).name}/summary.json",
}
temporary = run_dir / "status.json.tmp"
temporary.write_text(json.dumps(status, indent=2) + "\n")
os.replace(temporary, run_dir / "status.json")
PY

echo "[$(date -Is)] Validation and V1 Test completed for ${run_id}."
