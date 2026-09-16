#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
hst_data=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hve01-hve20-easy-natural-balanced-v1-eval
hct_data=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hvp01-hvp20-multiparty-quiz-expansion-v1-eval
output_root="${workspace}/evaluations/hst-hct-40-summary-delta-six-models-v1"
log_root="${output_root}/logs"

hst_scenarios=(921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940)
hct_scenarios=(901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920)

g350_summary="${workspace}/runs/granite4-350m-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-04"
g350_delta="${workspace}/runs/granite4-350m-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1/checkpoints/epoch-03"
q08_summary="${workspace}/runs/qwen35-0.8b-summary-multitask-noop5-grouped-v2-v1-10-e4-b4-trainseed45-evalfixed-noop5-r2/checkpoints/epoch-03"
q08_delta="${workspace}/runs/qwen35-0.8b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03"
g1_summary="${workspace}/runs/granite4-1b-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1/checkpoints/epoch-03"
g1_delta="${workspace}/runs/granite4-1b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1/checkpoints/epoch-04"
l1_summary="${workspace}/runs/llama3.2-1b-summary-multitask-noop5-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-04"
l1_delta="${workspace}/runs/llama3.2-1b-delta_v3_compact_k5-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r2/checkpoints/epoch-04"
q2_summary="${workspace}/runs/qwen35-2b-summary-multitask-noop5-grouped-v2-v1-10-e4-b2-r1/checkpoints/epoch-04"
q2_delta="${workspace}/runs/qwen35-2b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b2-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-04"
l3_summary="${workspace}/runs/llama3.2-3b-summary-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03"
l3_delta="${workspace}/runs/llama3.2-3b-delta_v3_compact_k5-multitask-noop5-e4-b1-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-04"

mkdir -p "$log_root"
exec 9>"${output_root}/scheduler.lock"
flock -n 9 || { echo "[$(date -Is)] another Summary/Delta HST/HCT scheduler is running" >&2; exit 3; }

gpu_is_empty() {
  local gpu=$1 memory_used pids
  memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$pids" && "$memory_used" =~ ^[0-9]+$ && "$memory_used" -le 100 ]]
}

for gpu in 4 5 7; do
  gpu_is_empty "$gpu" || { echo "[$(date -Is)] required GPU ${gpu} is not empty" >&2; exit 4; }
done

for checkpoint in \
  "$g350_summary" "$g350_delta" "$q08_summary" "$q08_delta" \
  "$g1_summary" "$g1_delta" "$l1_summary" "$l1_delta" \
  "$q2_summary" "$q2_delta" "$l3_summary" "$l3_delta"; do
  [[ -d "${checkpoint}/adapter" ]] || { echo "missing adapter: ${checkpoint}" >&2; exit 2; }
done

for data_root in "$hst_data" "$hct_data"; do
  for required in catalog.sqlite summary.jsonl delta.jsonl turn_quiz.jsonl final_quiz.jsonl quiz_sft.jsonl vehicle_tools.json; do
    [[ -s "${data_root}/${required}" ]] || { echo "missing input: ${data_root}/${required}" >&2; exit 2; }
  done
done

cd "$repo_root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_eval() {
  local gpu=$1 split=$2 model=$3 method=$4 checkpoint=$5 scenario_batch=$6 quiz_batch=$7
  local data_root output_dir log_path
  local -a scenarios
  if [[ "$split" == hst ]]; then
    data_root=$hst_data; scenarios=("${hst_scenarios[@]}")
  else
    data_root=$hct_data; scenarios=("${hct_scenarios[@]}")
  fi
  output_dir="${output_root}/${split}/${model}/${method}"
  log_path="${log_root}/gpu${gpu}-${model}-${method}-${split}.log"
  mkdir -p "$output_dir"
  if [[ -s "${output_dir}/summary.json" ]] && [[ "$(jq -r '.complete' "${output_dir}/summary.json")" == true ]]; then
    echo "[$(date -Is)] reuse GPU=${gpu} ${model} ${method} ${split}" | tee -a "${output_root}/scheduler.log"
    return
  fi
  echo "[$(date -Is)] start GPU=${gpu} ${model} ${method} ${split}" | tee -a "${output_root}/scheduler.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m memory_training.evaluate_hf_closed_loop \
    --model "$model" --method "$method" --checkpoint "$checkpoint" \
    --output-dir "$output_dir" --workspace "$workspace" --data-root "$data_root" \
    --catalog-path "${data_root}/catalog.sqlite" \
    --vehicle-tools-path "${data_root}/vehicle_tools.json" \
    --quiz-sft-path "${data_root}/quiz_sft.jsonl" \
    --vehiclemembench-root /home/hj153lee/VehicleMemBench \
    --scenarios "${scenarios[@]}" --max-length 4096 --max-new-tokens 768 \
    --quiz-max-new-tokens 256 --scenario-batch-size "$scenario_batch" --quiz-batch-size "$quiz_batch" \
    >>"$log_path" 2>&1
  echo "[$(date -Is)] done GPU=${gpu} ${model} ${method} ${split}" | tee -a "${output_root}/scheduler.log"
}

run_pair() {
  local gpu=$1 model=$2 method=$3 checkpoint=$4 scenario_batch=$5 quiz_batch=$6
  run_eval "$gpu" hst "$model" "$method" "$checkpoint" "$scenario_batch" "$quiz_batch"
  run_eval "$gpu" hct "$model" "$method" "$checkpoint" "$scenario_batch" "$quiz_batch"
}

worker4() {
  run_pair 4 llama3.2-3b summary "$l3_summary" 4 12
  run_pair 4 llama3.2-3b delta_v3_compact_k5 "$l3_delta" 4 12
}

worker5() {
  run_pair 5 qwen3.5-2b summary "$q2_summary" 5 16
  run_pair 5 qwen3.5-2b delta_v3_compact_k5 "$q2_delta" 5 16
  run_pair 5 granite4-350m summary "$g350_summary" 5 16
  run_pair 5 granite4-350m delta_v3_compact_k5 "$g350_delta" 5 16
}

worker7() {
  run_pair 7 granite4-1b summary "$g1_summary" 5 16
  run_pair 7 granite4-1b delta_v3_compact_k5 "$g1_delta" 5 16
  run_pair 7 llama3.2-1b summary "$l1_summary" 5 16
  run_pair 7 llama3.2-1b delta_v3_compact_k5 "$l1_delta" 5 16
  run_pair 7 qwen3.5-0.8b summary "$q08_summary" 5 16
  run_pair 7 qwen3.5-0.8b delta_v3_compact_k5 "$q08_delta" 5 16
}

worker4 & pid4=$!
worker5 & pid5=$!
worker7 & pid7=$!
failed=0
wait "$pid4" || failed=1
wait "$pid5" || failed=1
wait "$pid7" || failed=1
if [[ "$failed" -ne 0 ]]; then
  echo "[$(date -Is)] one or more Summary/Delta workers failed" | tee -a "${output_root}/scheduler.log" >&2
  exit 1
fi
echo "[$(date -Is)] all Summary/Delta HST/HCT evaluations completed" | tee -a "${output_root}/scheduler.log"
