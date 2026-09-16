#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
output_root="${workspace}/evaluations/luna-reasoning-none-v2-hst-hct-v1"
log_root="${output_root}/logs"
v2_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1
hst_root=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hve01-hve20-easy-natural-balanced-v1-eval
hct_root=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hvp01-hvp20-multiparty-quiz-expansion-v1-eval
scenario_parallelism="${SCENARIO_PARALLELISM:-4}"
quiz_workers="${QUIZ_WORKERS:-4}"

v2_scenarios=(86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 112 113 114 115 116 117 118 119 120)
hst_scenarios=(921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940)
hct_scenarios=(901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920)

mkdir -p "$output_root" "$log_root"
exec 9>"${output_root}/scheduler.lock"
flock -n 9 || { echo "[$(date -Is)] another Luna reasoning-none scheduler is running" >&2; exit 3; }
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "OPENAI_API_KEY is required" >&2; exit 2; }
[[ "$scenario_parallelism" =~ ^[1-9][0-9]*$ ]] || { echo "invalid SCENARIO_PARALLELISM" >&2; exit 2; }
[[ "$quiz_workers" =~ ^[1-9][0-9]*$ ]] || { echo "invalid QUIZ_WORKERS" >&2; exit 2; }

cd "$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_scenario() {
  local split=$1 data_root=$2 scenario=$3 scenario_tag scenario_output scenario_log
  printf -v scenario_tag 's%03d' "$scenario"
  scenario_output="${output_root}/${split}/patch/${scenario_tag}"
  scenario_log="${log_root}/${split}-${scenario_tag}.log"
  mkdir -p "$scenario_output"
  if [[ -s "${scenario_output}/summary.json" ]] \
      && [[ "$(jq -r '.complete // false' "${scenario_output}/summary.json")" == true ]] \
      && [[ "$(jq -r '.reasoning_effort // ""' "${scenario_output}/summary.json")" == none ]]; then
    echo "[$(date -Is)] reuse ${split} ${scenario_tag}"
    return
  fi
  echo "[$(date -Is)] start ${split} ${scenario_tag}"
  "$python_bin" -m memory_training.evaluate_cloud_closed_loop \
    --method patch \
    --scenario "$scenario" \
    --model gpt-5.6-luna \
    --reasoning-effort none \
    --output-dir "$scenario_output" \
    --data-root "$data_root" \
    --workspace "$workspace" \
    --catalog-path "${data_root}/catalog.sqlite" \
    --vehicle-tools-path "${data_root}/vehicle_tools.json" \
    --quiz-sft-path "${data_root}/quiz_sft.jsonl" \
    --vehiclemembench-root /home/hj153lee/VehicleMemBench \
    --memory-max-output-tokens 768 \
    --quiz-max-output-tokens 256 \
    --quiz-workers "$quiz_workers" \
    >>"$scenario_log" 2>&1
  echo "[$(date -Is)] done ${split} ${scenario_tag}"
}

aggregate_split() {
  local split=$1 split_root="${output_root}/${split}/patch" temporary
  local -a summaries
  mapfile -t summaries < <(find "$split_root" -mindepth 2 -maxdepth 2 -name summary.json -type f | sort)
  temporary="${split_root}/aggregate.json.tmp"
  jq -s --arg split "$split" '
    (map(.memory.decision_counts["UPDATE->UPDATE"] // 0) | add) as $tp |
    (map(.memory.decision_counts["NO_OP->UPDATE"] // 0) | add) as $fp |
    (map((.memory.decision_counts["UPDATE->NO_OP"] // 0) +
         (.memory.decision_counts["UPDATE->INVALID"] // 0)) | add) as $fn |
    (map(.closed_loop_quiz.records[]) ) as $quiz |
    ($quiz | length) as $quiz_tasks |
    ($quiz | map(select(.quiz_type == "TURN"))) as $turn_quiz |
    ($quiz | map(select(.quiz_type == "FINAL"))) as $final_quiz |
    ($tp / (($tp + $fp) | if . == 0 then 1 else . end)) as $precision |
    ($tp / (($tp + $fn) | if . == 0 then 1 else . end)) as $recall |
    {
      schema_version: "palmclaw-cloud-three-split-aggregate-v1",
      split: $split,
      model: "gpt-5.6-luna",
      method: "patch",
      reasoning_effort: "none",
      complete: (map(.complete == true) | all),
      scenarios: length,
      memory: {
        update_precision: $precision,
        update_recall: $recall,
        update_f1: (if ($precision + $recall) == 0 then 0 else 2 * $precision * $recall / ($precision + $recall) end),
        final_state_f1: (map(.memory.final_state_f1) | add / length),
        invalid_outputs: (map(.memory.invalid_outputs) | add)
      },
      closed_loop_quiz: {
        tasks: $quiz_tasks,
        esm: ($quiz | map(.exact_state_match) | add / $quiz_tasks),
        turn_esm: ($turn_quiz | map(.exact_state_match) | add / length),
        final_esm: ($final_quiz | map(.exact_state_match) | add / length),
        tool_f1: ($quiz | map(.tool_f1) | add / $quiz_tasks),
        argument_exact: ($quiz | map(.arg_exact) | add / $quiz_tasks)
      },
      total: {
        cost_usd: (map(.total.cost_usd) | add),
        logical_calls: (map(.total.logical_calls) | add),
        usage: {
          input_tokens: (map(.total.usage.input_tokens) | add),
          cached_input_tokens: (map(.total.usage.cached_input_tokens) | add),
          output_tokens: (map(.total.usage.output_tokens) | add),
          total_tokens: (map(.total.usage.total_tokens) | add)
        }
      }
    }
    | .composite = (0.60 * .closed_loop_quiz.esm + 0.25 * .memory.final_state_f1 + 0.15 * .memory.update_f1)
  ' "${summaries[@]}" >"$temporary"
  mv "$temporary" "${split_root}/aggregate.json"
}

run_split() {
  local split=$1 data_root=$2
  shift 2
  local -a scenarios=("$@") batch pids
  local offset pid failed
  mkdir -p "${output_root}/${split}/patch"
  echo "[$(date -Is)] begin ${split}: ${#scenarios[@]} scenarios"
  for ((offset=0; offset<${#scenarios[@]}; offset+=scenario_parallelism)); do
    pids=()
    batch=("${scenarios[@]:offset:scenario_parallelism}")
    for scenario in "${batch[@]}"; do
      run_scenario "$split" "$data_root" "$scenario" &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
    [[ "$failed" -eq 0 ]] || { echo "[$(date -Is)] ${split} batch failed: ${batch[*]}" >&2; exit 1; }
  done
  aggregate_split "$split"
  echo "[$(date -Is)] completed ${split}"
}

run_split v2_test "$v2_root" "${v2_scenarios[@]}"
run_split hst "$hst_root" "${hst_scenarios[@]}"
run_split hct "$hct_root" "${hct_scenarios[@]}"
echo "[$(date -Is)] all Luna reasoning-none evaluations completed"
