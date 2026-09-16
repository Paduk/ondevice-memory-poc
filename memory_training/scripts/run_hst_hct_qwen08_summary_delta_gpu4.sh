#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
output_root="${workspace}/evaluations/hst-hct-40-summary-delta-six-models-v1"
hst_data=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hve01-hve20-easy-natural-balanced-v1-eval
hct_data=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hvp01-hvp20-multiparty-quiz-expansion-v1-eval
summary_checkpoint="${workspace}/runs/qwen35-0.8b-summary-multitask-noop5-grouped-v2-v1-10-e4-b4-trainseed45-evalfixed-noop5-r2/checkpoints/epoch-03"
delta_checkpoint="${workspace}/runs/qwen35-0.8b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1/checkpoints/epoch-03"
hst_scenarios=(921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940)
hct_scenarios=(901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920)

mkdir -p "${output_root}/logs"
exec 9>"${output_root}/qwen08-gpu4.lock"
flock -n 9 || exit 3

used=$(nvidia-smi -i 4 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
pids=$(nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
[[ -z "$pids" && "$used" -le 100 ]] || { echo "GPU 4 is not empty" >&2; exit 4; }

cd "$repo_root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_one() {
  local split=$1 method=$2 checkpoint=$3 data_root output_dir log_path
  local -a scenarios
  if [[ "$split" == hst ]]; then data_root=$hst_data; scenarios=("${hst_scenarios[@]}");
  else data_root=$hct_data; scenarios=("${hct_scenarios[@]}"); fi
  output_dir="${output_root}/${split}/qwen3.5-0.8b/${method}"
  log_path="${output_root}/logs/gpu4-qwen3.5-0.8b-${method}-${split}.log"
  mkdir -p "$output_dir"
  if [[ -s "${output_dir}/summary.json" ]] && [[ "$(jq -r '.complete' "${output_dir}/summary.json")" == true ]]; then return; fi
  echo "[$(date -Is)] start GPU=4 qwen3.5-0.8b ${method} ${split} (rebalanced)" | tee -a "${output_root}/scheduler.log"
  CUDA_VISIBLE_DEVICES=4 "$python_bin" -m memory_training.evaluate_hf_closed_loop \
    --model qwen3.5-0.8b --method "$method" --checkpoint "$checkpoint" \
    --output-dir "$output_dir" --workspace "$workspace" --data-root "$data_root" \
    --catalog-path "${data_root}/catalog.sqlite" --vehicle-tools-path "${data_root}/vehicle_tools.json" \
    --quiz-sft-path "${data_root}/quiz_sft.jsonl" --vehiclemembench-root /home/hj153lee/VehicleMemBench \
    --scenarios "${scenarios[@]}" --max-length 4096 --max-new-tokens 768 \
    --quiz-max-new-tokens 256 --scenario-batch-size 5 --quiz-batch-size 16 >>"$log_path" 2>&1
  echo "[$(date -Is)] done GPU=4 qwen3.5-0.8b ${method} ${split} (rebalanced)" | tee -a "${output_root}/scheduler.log"
}

run_one hst summary "$summary_checkpoint"
run_one hct summary "$summary_checkpoint"
run_one hst delta_v3_compact_k5 "$delta_checkpoint"
run_one hct delta_v3_compact_k5 "$delta_checkpoint"
