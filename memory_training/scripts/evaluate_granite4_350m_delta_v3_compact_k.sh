#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
interval="${2:?Compaction interval is required}"
epochs="${3:-4}"
training_seed="${4:-45}"
run_tag="${5:-r1}"

case "${interval}" in 2|5|10) ;; *) echo "Invalid K=${interval}" >&2; exit 2 ;; esac

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training
eval_root="${data_parent}/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1-delta-v3-compact-k${interval}-v1"
method="delta_v3_compact_k${interval}"
run_id="granite4-350m-${method}-multitask-noop5-uniform-depth-e${epochs}-b8-trainseed${training_seed}-${run_tag}"
run_dir="${workspace}/runs/${run_id}"
pipeline_log="${run_dir}/fixed-eval-pipeline.log"

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"
exec > >(tee -a "${pipeline_log}") 2>&1

validate_epoch() {
  local epoch="$1" checkpoint output log
  checkpoint="${run_dir}/checkpoints/epoch-${epoch}"
  output="${run_dir}/eval-fixed-validation-epoch-${epoch}.json"
  log="${run_dir}/eval-fixed-validation-epoch-${epoch}.log"
  if [[ -s "${output}" ]]; then
    echo "[$(date -Is)] Epoch ${epoch} validation exists; skipping."
    return
  fi
  "${python_bin}" -m memory_training.validate_hf_checkpoint_v2 \
    --run-id "${run_id}" --checkpoint "${checkpoint}" --output "${output}" \
    --workspace "${workspace}" --data-root "${eval_root}" \
    --catalog-path "${eval_root}/catalog.sqlite" \
    --validation-scenarios 81 82 83 84 85 111 \
    --full-scenarios 81 82 83 84 85 111 --closed-loop-only \
    --scenario-batch-size 3 --quiz-batch-size 16 --no-status-update \
    >"${log}" 2>&1
}

echo "[$(date -Is)] Starting epoch 03 and 04 fixed validation for K=${interval}."
validate_epoch 03 & p3=$!
validate_epoch 04 & p4=$!
failed=0
wait "${p3}" || failed=1
wait "${p4}" || failed=1
(( failed == 0 )) || exit 1

selection="${run_dir}/eval-fixed-best-checkpoint.json"
"${python_bin}" - "${run_dir}" "${selection}" <<'PY'
import json, sys
from pathlib import Path
run_dir, output = map(Path, sys.argv[1:])
rows = []
for epoch in (3, 4):
    path = run_dir / f"eval-fixed-validation-epoch-{epoch:02d}.json"
    value = json.loads(path.read_text())
    row = {
        "epoch": epoch,
        "checkpoint": value["checkpoint"],
        "closed_loop_quiz_esm": value["closed_loop_quiz"]["esm"],
        "closed_loop_final_state_f1": value["closed_loop"]["final_state_f1"],
        "closed_loop_update_f1": value["closed_loop"]["update_f1"],
        "output": str(path),
    }
    row["composite_score"] = (
        0.60 * row["closed_loop_quiz_esm"]
        + 0.25 * row["closed_loop_final_state_f1"]
        + 0.15 * row["closed_loop_update_f1"]
    )
    rows.append(row)
winner = max(rows, key=lambda row: (
    row["composite_score"], row["closed_loop_quiz_esm"],
    row["closed_loop_final_state_f1"], row["closed_loop_update_f1"], -row["epoch"],
))
output.write_text(json.dumps({"selection_metric": "composite_60_25_15",
                              "evaluated_epochs": [3, 4],
                              "results": rows, "winner": winner}, indent=2) + "\n")
PY

winner=$(jq -r '.winner.checkpoint' "${selection}")
winner_epoch=$(basename "${winner}")
test_output="${run_dir}/eval-fixed-test-best-${winner_epoch}"
echo "[$(date -Is)] Starting fixed Test with ${winner_epoch} for K=${interval}."
"${python_bin}" -m memory_training.evaluate_hf_closed_loop \
  --model granite4-350m --method "${method}" --checkpoint "${winner}" \
  --output-dir "${test_output}" --workspace "${workspace}" \
  --data-root "${eval_root}" --catalog-path "${eval_root}/catalog.sqlite" \
  --scenarios $(seq 86 100) $(seq 112 120) \
  --max-length 2048 --max-new-tokens 768 --quiz-max-new-tokens 256 \
  --scenario-batch-size 16 --quiz-batch-size 16

"${python_bin}" - "${run_dir}" "${winner}" "${selection}" "${test_output}" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
run_dir, winner, selection, test_output = map(Path, sys.argv[1:])
status = {"updated_at": datetime.now(timezone.utc).isoformat(), "state": "COMPLETED",
          "run_id": run_dir.name, "final_state": "COMPLETED",
          "best_checkpoint": str(winner),
          "best_metric": "eval_fixed.composite.quiz_esm_60.final_state_f1_25.update_f1_15",
          "external_validation": selection.name,
          "external_test": str(test_output / "summary.json")}
temporary = run_dir / "status.json.tmp"
temporary.write_text(json.dumps(status, indent=2) + "\n")
os.replace(temporary, run_dir / "status.json")
PY
echo "[$(date -Is)] COMPLETED K=${interval} validation and Test."
