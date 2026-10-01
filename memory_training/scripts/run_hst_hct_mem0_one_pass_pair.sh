#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU MODEL CHECKPOINT" >&2
  exit 2
fi

gpu=$1
model=$2
checkpoint=$3
repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
source_root=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory
hst_data="${source_root}/hve01-hve20-easy-natural-balanced-v1-one-pass-eval"
hct_data="${source_root}/hvp01-hvp20-multiparty-quiz-expansion-v1-one-pass-eval"
hst_catalog="${source_root}/hve01-hve20-easy-natural-balanced-v1-eval/catalog.sqlite"
hct_catalog="${source_root}/hvp01-hvp20-multiparty-quiz-expansion-v1-eval/catalog.sqlite"
output_root="${workspace}/evaluations/hst-hct-40-mem0-one-pass-six-models-v1"
log_root="${output_root}/logs"
hst_scenarios=(921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940)
hct_scenarios=(901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920)

case "$model" in
  llama3.2-3b) scenario_batch=4; quiz_batch=12 ;;
  *) scenario_batch=5; quiz_batch=16 ;;
esac

mkdir -p "$log_root"
exec 9>"${output_root}/${model}.pair.lock"
flock -n 9 || { echo "[$(date -Is)] ${model} pair already running" >&2; exit 3; }

if [[ "${ALLOW_BUSY:-0}" != 1 ]]; then
  memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$pids" && "$memory_used" =~ ^[0-9]+$ && "$memory_used" -le 100 ]] || {
    echo "[$(date -Is)] GPU ${gpu} is not empty" >&2
    exit 4
  }
fi
[[ -d "${checkpoint}/adapter" ]] || { echo "missing adapter: ${checkpoint}" >&2; exit 2; }

cd "$repo_root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export PALMCLAW_MEM0_RETRIEVAL_DEVICE=cpu
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

for split in hst hct; do
  if [[ "$split" == hst ]]; then
    data_root=$hst_data; catalog=$hst_catalog; scenarios=("${hst_scenarios[@]}")
  else
    data_root=$hct_data; catalog=$hct_catalog; scenarios=("${hct_scenarios[@]}")
  fi
  output_dir="${output_root}/${split}/${model}/mem0_one_pass"
  log_path="${log_root}/gpu${gpu}-${model}-${split}.log"
  mkdir -p "$output_dir"
  if [[ -s "${output_dir}/summary.json" ]] \
      && [[ "$(jq -r '.complete' "${output_dir}/summary.json")" == true ]]; then
    echo "[$(date -Is)] reuse GPU=${gpu} model=${model} split=${split}" | tee -a "${output_root}/scheduler.log"
    continue
  fi
  echo "[$(date -Is)] start GPU=${gpu} model=${model} split=${split} (pair worker)" | tee -a "${output_root}/scheduler.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m memory_training.evaluate_hf_closed_loop \
    --model "$model" --method mem0_one_pass --checkpoint "$checkpoint" \
    --output-dir "$output_dir" --workspace "$workspace" --data-root "$data_root" \
    --catalog-path "$catalog" --vehicle-tools-path "${data_root}/vehicle_tools.json" \
    --quiz-sft-path "${data_root}/quiz_sft.jsonl" --vehiclemembench-root /home/hj153lee/VehicleMemBench \
    --scenarios "${scenarios[@]}" --max-length 4096 --max-new-tokens 768 \
    --quiz-max-new-tokens 256 --scenario-batch-size "$scenario_batch" --quiz-batch-size "$quiz_batch" \
    >>"$log_path" 2>&1
  echo "[$(date -Is)] done GPU=${gpu} model=${model} split=${split} (pair worker)" | tee -a "${output_root}/scheduler.log"
done
