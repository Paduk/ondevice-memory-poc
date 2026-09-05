#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index required}"
method="${2:?method required: patch, delta_v3, or summary}"
run_override="${3:-}"
start_epoch="${4:-1}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
eval_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1
eval_catalog="${eval_root}/catalog.sqlite"

case "${method}" in
  patch)
    run_id=qwen35-2b-patch-multitask-noop5-grouped-v2-v1-10-e5-b4-r1
    epochs=5
    ;;
  summary)
    run_id=qwen35-2b-summary-multitask-noop5-grouped-v2-v1-10-e4-b2-r1
    epochs=4
    ;;
  delta_v3)
    run_id=qwen35-2b-delta-v3-multitask-noop5-grouped-v2-v1-10-e4-b2-trainseed45-evalfixed-noop5-r1
    epochs=4
    ;;
  *) echo "unsupported method: ${method}" >&2; exit 2 ;;
esac

if [[ -n "${run_override}" ]]; then
  run_id="${run_override}"
fi

run_dir="${workspace}/runs/${run_id}"
log_path="${run_dir}/eval-fixed-v2-pipeline.log"
validation_scenarios=(81 82 83 84 85 111)
test_scenarios=($(seq 86 100) $(seq 112 120))

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"
exec > >(tee -a "${log_path}") 2>&1

validate_epoch() {
  local n="$1" epoch checkpoint output epoch_log
  epoch=$(printf '%02d' "${n}")
  checkpoint="${run_dir}/checkpoints/epoch-${epoch}"
  output="${run_dir}/eval-fixed-v2-validation-epoch-${epoch}.json"
  epoch_log="${run_dir}/eval-fixed-v2-validation-epoch-${epoch}.log"
  if [[ -s "${output}" ]]; then
    echo "[$(date -Is)] Validation epoch ${epoch} exists; skipping."
    return
  fi
  echo "[$(date -Is)] Starting validation epoch ${epoch} on GPU ${gpu}."
  "${python_bin}" -m memory_training.validate_hf_checkpoint_v2 \
    --run-id "${run_id}" --checkpoint "${checkpoint}" --output "${output}" \
    --workspace "${workspace}" --data-root "${eval_root}" --catalog-path "${eval_catalog}" \
    --validation-scenarios "${validation_scenarios[@]}" --full-scenarios "${validation_scenarios[@]}" \
    --closed-loop-only --scenario-batch-size 3 --quiz-batch-size 16 --no-status-update \
    >"${epoch_log}" 2>&1
}

# Two checkpoints fit on an A100 and substantially shorten checkpoint selection.
declare -A running=()
next="${start_epoch}"
while (( next <= epochs || ${#running[@]} > 0 )); do
  while (( next <= epochs && ${#running[@]} < 2 )); do
    validate_epoch "${next}" & running[$!]="${next}"; next=$((next + 1))
  done
  reaped=0
  for pid in "${!running[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      n="${running[$pid]}"
      if ! wait "${pid}"; then echo "Validation epoch ${n} failed" >&2; exit 1; fi
      unset 'running[$pid]'; reaped=1
    fi
  done
  (( reaped )) || sleep 2
done

selection="${run_dir}/eval-fixed-v2-best-checkpoint.json"
"${python_bin}" - "${run_dir}" "${selection}" "${epochs}" "${start_epoch}" <<'PY'
import json, sys
from pathlib import Path
run_dir, output, epochs, start_epoch = (
    Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
)
rows = []
for epoch in range(start_epoch, epochs + 1):
    p = run_dir / f"eval-fixed-v2-validation-epoch-{epoch:02d}.json"
    x = json.loads(p.read_text())
    rows.append({"epoch": epoch, "checkpoint": x["checkpoint"],
                 "closed_loop_update_f1": x["closed_loop"]["update_f1"],
                 "closed_loop_final_state_f1": x["closed_loop"]["final_state_f1"],
                 "closed_loop_quiz_esm": x["closed_loop_quiz"]["esm"], "output": str(p)})
for row in rows:
    row["composite_score"] = (0.60 * row["closed_loop_quiz_esm"]
                              + 0.25 * row["closed_loop_final_state_f1"]
                              + 0.15 * row["closed_loop_update_f1"])
winner = max(rows, key=lambda r: (r["composite_score"], r["closed_loop_quiz_esm"],
    r["closed_loop_final_state_f1"], r["closed_loop_update_f1"], -r["epoch"]))
output.write_text(json.dumps({"selection_metric": "composite_60_25_15",
    "results": rows, "winner": winner}, indent=2) + "\n")
PY

winner=$(jq -r '.winner.checkpoint' "${selection}")
winner_epoch=$(basename "${winner}")
test_output="${run_dir}/eval-fixed-v2-test-best-${winner_epoch}"
if [[ $(jq -r '.complete // false' "${test_output}/summary.json" 2>/dev/null || true) == true ]]; then
  echo "[$(date -Is)] Test already complete; skipping."
else
  echo "[$(date -Is)] Starting test with ${winner_epoch}."
  "${python_bin}" -m memory_training.evaluate_hf_closed_loop \
    --model qwen3.5-2b --method "${method}" --checkpoint "${winner}" \
    --output-dir "${test_output}" --workspace "${workspace}" \
    --data-root "${eval_root}" --catalog-path "${eval_catalog}" \
    --scenarios "${test_scenarios[@]}" --max-length 2048 --max-new-tokens 768 \
    --quiz-max-new-tokens 256 --scenario-batch-size 8 --quiz-batch-size 16
fi
echo "[$(date -Is)] COMPLETED ${method}: validation and test."
