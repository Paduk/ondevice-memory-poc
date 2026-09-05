#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?GPU index required}
size=${2:?Model size required: 350m or 1b}
repo=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1

case "${size}" in
  350m)
    runs=(
      granite4-350m-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1
      granite4-350m-patch-multitask-noop5-full6val-grouped-v2-v1-10-e4-b8-trainseed45-r1
      granite4-350m-delta-v3-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1
    )
    model=granite4-350m
    ;;
  1b)
    runs=(
      granite4-1b-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1
      granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1
      granite4-1b-delta-v3-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1
    )
    model=granite4-1b
    ;;
  *) exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${gpu}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${repo}:${repo}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

test_run() {
  local run_id=$1 run_dir validation checkpoint epoch method output log
  run_dir="${workspace}/runs/${run_id}"
  validation=$(find "${run_dir}" -maxdepth 1 -type f -name 'eval-fixed-v2-validation-best-epoch-*.json' | head -1)
  checkpoint=$(jq -r '.checkpoint' "${validation}")
  epoch=$(basename "${checkpoint}")
  method=$(jq -r '.method' "${run_dir}/config.json")
  output="${run_dir}/eval-fixed-v2-test-best-${epoch}"
  log="${run_dir}/eval-fixed-v2-test-best-${epoch}.log"
  "${python}" -m memory_training.evaluate_hf_closed_loop \
    --model "${model}" --method "${method}" --checkpoint "${checkpoint}" \
    --output-dir "${output}" --workspace "${workspace}" \
    --data-root "${data}" --catalog-path "${data}/catalog.sqlite" \
    --scenarios 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 112 113 114 115 116 117 118 119 120 \
    --max-length 2048 --max-new-tokens 768 --quiz-max-new-tokens 256 \
    --scenario-batch-size 16 --quiz-batch-size 16 >"${log}" 2>&1
}

cd "${repo}"
test_run "${runs[0]}" & first=$!
test_run "${runs[1]}" & second=$!
failed=0
wait "${first}" || failed=1
wait "${second}" || failed=1
(( failed == 0 )) || exit 1
test_run "${runs[2]}"
