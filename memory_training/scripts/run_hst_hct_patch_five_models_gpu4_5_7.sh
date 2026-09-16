#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
hst_data=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hve01-hve20-easy-natural-balanced-v1-eval
hct_data=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hvp01-hvp20-multiparty-quiz-expansion-v1-eval
output_root="${workspace}/evaluations/hst-hct-40-patch-five-models-v1"
log_root="${output_root}/logs"

granite350_checkpoint="${workspace}/runs/granite4-350m-patch-multitask-noop5-full6val-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-04"
qwen08_checkpoint="${workspace}/runs/qwen35-0.8b-patch-multitask-noop5-grouped-v2-v1-10-e4-b8-trainseed46-evalfixed-noop5-r1/checkpoints/epoch-03"
qwen2_checkpoint="${workspace}/runs/qwen35-2b-patch-multitask-noop5-grouped-v2-v1-10-e5-b4-trainseed46-r2/checkpoints/epoch-03"
llama1_checkpoint="${workspace}/runs/llama3.2-1b-patch-multitask-noop5-e4-b8-trainseed45-evalfixed-noop5-r2/checkpoints/epoch-03"
llama3_checkpoint="${workspace}/runs/llama3.2-3b-patch-multitask-noop5-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-04"

hst_scenarios=(921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940)
hct_scenarios=(901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920)

mkdir -p "$log_root"
exec 9>"${output_root}/scheduler.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] another HST/HCT five-model scheduler is running" >&2
  exit 3
fi

gpu_is_empty() {
  local gpu=$1
  local memory_used pids
  memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$pids" && "$memory_used" =~ ^[0-9]+$ && "$memory_used" -le 100 ]]
}

for gpu in 4 5 7; do
  if ! gpu_is_empty "$gpu"; then
    echo "[$(date -Is)] required GPU ${gpu} is no longer empty" >&2
    exit 4
  fi
done

for checkpoint in \
  "$granite350_checkpoint" "$qwen08_checkpoint" "$qwen2_checkpoint" \
  "$llama1_checkpoint" "$llama3_checkpoint"; do
  [[ -d "${checkpoint}/adapter" ]] || {
    echo "[$(date -Is)] checkpoint adapter not found: ${checkpoint}" >&2
    exit 2
  }
done

for data_root in "$hst_data" "$hct_data"; do
  for required in catalog.sqlite patch.jsonl turn_quiz.jsonl final_quiz.jsonl quiz_sft.jsonl vehicle_tools.json; do
    [[ -s "${data_root}/${required}" ]] || {
      echo "[$(date -Is)] evaluation input not found: ${data_root}/${required}" >&2
      exit 2
    }
  done
done

cd "$repo_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_eval() {
  local gpu=$1
  local split=$2
  local model=$3
  local checkpoint=$4
  local scenario_batch_size=$5
  local quiz_batch_size=$6
  local data_root output_dir log_path
  local -a scenarios

  if [[ "$split" == hst ]]; then
    data_root=$hst_data
    scenarios=("${hst_scenarios[@]}")
  else
    data_root=$hct_data
    scenarios=("${hct_scenarios[@]}")
  fi

  output_dir="${output_root}/${split}/${model}/patch"
  log_path="${log_root}/gpu${gpu}-${model}-${split}.log"
  mkdir -p "$output_dir"

  if [[ -s "${output_dir}/summary.json" ]] \
      && [[ "$(jq -r '.complete' "${output_dir}/summary.json")" == true ]]; then
    echo "[$(date -Is)] reuse complete GPU=${gpu} model=${model} split=${split}" \
      | tee -a "${output_root}/scheduler.log"
    return 0
  fi

  echo "[$(date -Is)] start GPU=${gpu} model=${model} split=${split}" \
    | tee -a "${output_root}/scheduler.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m memory_training.evaluate_hf_closed_loop \
    --model "$model" \
    --method patch \
    --checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --workspace "$workspace" \
    --data-root "$data_root" \
    --catalog-path "${data_root}/catalog.sqlite" \
    --vehicle-tools-path "${data_root}/vehicle_tools.json" \
    --quiz-sft-path "${data_root}/quiz_sft.jsonl" \
    --vehiclemembench-root /home/hj153lee/VehicleMemBench \
    --scenarios "${scenarios[@]}" \
    --max-length 4096 \
    --max-new-tokens 768 \
    --quiz-max-new-tokens 256 \
    --scenario-batch-size "$scenario_batch_size" \
    --quiz-batch-size "$quiz_batch_size" \
    >>"$log_path" 2>&1
  echo "[$(date -Is)] done GPU=${gpu} model=${model} split=${split}" \
    | tee -a "${output_root}/scheduler.log"
}

worker_gpu4() {
  run_eval 4 hst llama3.2-3b "$llama3_checkpoint" 4 12
  run_eval 4 hct llama3.2-3b "$llama3_checkpoint" 4 12
}

worker_gpu5() {
  run_eval 5 hst qwen3.5-2b "$qwen2_checkpoint" 5 16
  run_eval 5 hct qwen3.5-2b "$qwen2_checkpoint" 5 16
  run_eval 5 hst granite4-350m "$granite350_checkpoint" 5 16
  run_eval 5 hct granite4-350m "$granite350_checkpoint" 5 16
}

worker_gpu7() {
  run_eval 7 hst llama3.2-1b "$llama1_checkpoint" 5 16
  run_eval 7 hct llama3.2-1b "$llama1_checkpoint" 5 16
  run_eval 7 hst qwen3.5-0.8b "$qwen08_checkpoint" 5 16
  run_eval 7 hct qwen3.5-0.8b "$qwen08_checkpoint" 5 16
}

worker_gpu4 &
pid4=$!
worker_gpu5 &
pid5=$!
worker_gpu7 &
pid7=$!

failed=0
wait "$pid4" || failed=1
wait "$pid5" || failed=1
wait "$pid7" || failed=1

if [[ "$failed" -ne 0 ]]; then
  echo "[$(date -Is)] one or more HST/HCT workers failed" \
    | tee -a "${output_root}/scheduler.log" >&2
  exit 1
fi

echo "[$(date -Is)] all five-model HST/HCT Patch evaluations completed" \
  | tee -a "${output_root}/scheduler.log"
