#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
run_id="${2:?Run ID is required}"
first_epoch="${3:-1}"
last_epoch="${4:-4}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
run_dir="${workspace}/runs/${run_id}"
log_path="${run_dir}/validation-6full-pairwise.log"

cd "${repo_root}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

validate_epoch() {
  local epoch_number=$1
  local epoch
  epoch=$(printf '%02d' "${epoch_number}")
  local checkpoint="${run_dir}/checkpoints/epoch-${epoch}"
  local output="${run_dir}/validation-6full-pairwise-epoch-${epoch}.json"
  local epoch_log="${run_dir}/validation-6full-pairwise-epoch-${epoch}.log"

  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "[$(date -Is)] Missing checkpoint adapter: ${checkpoint}" >&2
    return 1
  fi
  if [[ -s "${output}" ]] && jq -e \
      '.protocol.closed_loop_only == true and
       .protocol.validation_scenarios == [81,82,83,84,85,111] and
       .protocol.scenario_batch_size == 3' \
      "${output}" >/dev/null 2>&1; then
    echo "[$(date -Is)] Epoch ${epoch} already completed; skipping."
    return 0
  fi

  echo "[$(date -Is)] Starting epoch ${epoch} on GPU ${gpu}."
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

exec > >(tee -a "${log_path}") 2>&1
echo "[$(date -Is)] Starting pairwise checkpoint Validation for ${run_id}."

declare -A running_epochs=()
next_epoch=${first_epoch}
failed=0

while (( next_epoch <= last_epoch || ${#running_epochs[@]} > 0 )); do
  while (( next_epoch <= last_epoch && ${#running_epochs[@]} < 2 )); do
    validate_epoch "${next_epoch}" &
    running_epochs[$!]="${next_epoch}"
    next_epoch=$((next_epoch + 1))
  done

  reaped=0
  for pid in "${!running_epochs[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      epoch="${running_epochs[$pid]}"
      if ! wait "${pid}"; then
        echo "[$(date -Is)] Epoch $(printf '%02d' "${epoch}") Validation failed." >&2
        failed=1
      fi
      unset 'running_epochs[$pid]'
      reaped=1
    fi
  done
  if (( failed )); then
    for pid in "${!running_epochs[@]}"; do
      kill "${pid}" 2>/dev/null || true
    done
    exit 1
  fi
  if (( reaped == 0 )); then
    sleep 2
  fi
done

"${python_bin}" - "${run_dir}" "${first_epoch}" "${last_epoch}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
first_epoch = int(sys.argv[2])
last_epoch = int(sys.argv[3])
rows = []
for epoch in range(first_epoch, last_epoch + 1):
    path = run_dir / f"validation-6full-pairwise-epoch-{epoch:02d}.json"
    value = json.loads(path.read_text())
    rows.append(
        {
            "epoch": epoch,
            "checkpoint": value["checkpoint"],
            "closed_loop_update_f1": value["closed_loop"]["update_f1"],
            "closed_loop_final_state_f1": value["closed_loop"]["final_state_f1"],
            "closed_loop_quiz_esm": value["closed_loop_quiz"]["esm"],
            "output": str(path),
        }
    )
winner = max(
    rows,
    key=lambda row: (
        row["closed_loop_quiz_esm"], row["closed_loop_final_state_f1"]
    ),
)
summary = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "protocol": "6-full pairwise, two checkpoints concurrently, batch-3 each",
    "results": rows,
    "winner": winner,
}
(run_dir / "validation-6full-pairwise-summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
status = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "state": "COMPLETED",
    "run_id": run_dir.name,
    "final_state": "COMPLETED",
    "best_checkpoint": winner["checkpoint"],
    "best_score": winner["closed_loop_quiz_esm"],
    "best_metric": "closed_loop_quiz.esm_then_final_state_f1",
    "external_validation": "validation-6full-pairwise-summary.json",
}
target = run_dir / "status.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(status, indent=2) + "\n")
os.replace(temporary, target)
PY

echo "[$(date -Is)] Completed pairwise checkpoint Validation for ${run_id}."
