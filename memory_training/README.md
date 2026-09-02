# On-device memory training

Phase 1 provides an isolated NVMe workspace and read-only preflight checks. Phase 2
adds an aligned byte-offset catalog, lazy JSONL access and deterministic epoch
sampling. Canonical JSONL files stay in their existing location and are not copied.

The default Train sampler keeps every UPDATE and selects NO_OP at `1:10`. Of the
selected NO_OP rows, 30% are hard negatives nearest to UPDATE turns (distance 1 first,
then distance 2) and 70% are deterministic epoch-random rows.

```bash
source /mnt/data/miniconda3/bin/activate
conda env create \
  -p /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft \
  -f memory_training/environment.yml
conda activate /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft

python -m memory_training.workspace
python -m memory_training.preflight --mode full --hashes
python -m memory_training.model_canary --model granite4.1-3b --mode metadata
python -m memory_training.model_canary --model qwen3.5-4b --mode metadata

python -m memory_training.dataset build
python -m memory_training.dataset inspect
python -m memory_training.sampling --epoch 0
python -m memory_training.validate_methods

# One GPU per run; run the 4x4 matrix as separate processes.
CUDA_VISIBLE_DEVICES=0 python -m memory_training.train \
  --model granite4.1-3b --method patch --run-id granite3b-patch-r1

# Resume exactly from an adapter-only checkpoint.
CUDA_VISIBLE_DEVICES=0 python -m memory_training.train \
  --model granite4.1-3b --method patch --run-id granite3b-patch-r1-resume \
  --resume /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/\
granite3b-patch-r1/checkpoints/epoch-01
```

Every 500 optimizer steps, the runner records teacher-forced loss on a fixed 512-row
Validation subset: every S81-S90 UPDATE plus an equal-budget mix of adjacent hard and
deterministic random NO_OP rows. Each epoch also records stratified one-step generation
and full S81-S90 closed-loop state metrics. The held-out S91-S100 Test is run only once
after selecting the best checkpoint. `--quiz-hook` optionally invokes an external evaluator with
`{context}` and `{output}` placeholders; the later Ollama phase supplies the production
VehicleMemBench Quiz bridge. `--mlflow-uri` mirrors numeric metrics to MLflow, while
local `metrics.jsonl` and atomic `status.json` remain authoritative if MLflow is down.

Start the local multi-run dashboard (overview matrix, comparison, and per-run detail)
on port 5060. `COMPLETED` and `FAILED` runs can be archived from the UI; active and
queued runs are protected. Archives remain recoverable under
`on-device-memory-training/trash/runs` and are not permanently deleted.

```bash
python -m memory_training.dashboard.server \
  --runs-root /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs \
  --host 0.0.0.0 --port 5060
```

## Ollama export and S91-S100 Test

Export only the checkpoint selected by Validation. `plan` is read-only and checks
free disk, model architecture support, the converter, and the Ollama server before
large files are created.

```bash
python -m memory_training.export_ollama \
  --run-dir /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/<run-id> \
  --stage plan

python -m memory_training.export_ollama \
  --run-dir /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/<run-id> \
  --stage all --cleanup-intermediate --threads 16
```

The default full stage merges the LoRA adapter, creates a **BF16 GGUF without
quantization**, registers a unique Ollama tag, records its digest, and removes the
merged HF intermediate after success. Export and test one model at a time because
the BF16 source GGUF and Ollama blob each require roughly the full 16-bit model size.

Quantization is a later ablation, not the primary Test condition. To create Q4_K_M
explicitly, pass `--gguf-outtype f16 --quantization Q4_K_M`.

Run the held-out Test exactly once with the fine-tuned memory model and one fixed,
tool-capable Quiz Agent tag. The default scenarios are S91-S100 and all calls use
greedy decoding.

```bash
python -m memory_training.ollama_test \
  --run-dir /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/<run-id> \
  --memory-model <fine-tuned-ollama-tag> \
  --quiz-model <fixed-quiz-agent-ollama-tag>
```

The runner checkpoints every 25 turns, resumes without repeating completed calls,
and rejects stale results when its model/data/config signature changes. It stores
Memory State F1 and update/no-op/schema metrics, Turn/Final Quiz ESM/State F1/Tool
F1/Arg Exact, prefill/decode tokens, and memory/Quiz latency. `--force` is reserved
for diagnostics; it must not be used to choose a better paper Test result. The
dashboard reads `ollama-test-summary.json` from each run automatically.

For the fixed five-scenario diagnostic, evaluate S91-S95 on both axes without
running Quiz inference. `closed_loop` feeds each prediction into the next turn;
`teacher_forced` resets every turn to the Gold previous memory. Their gap isolates
error accumulation from one-turn generation quality.

```bash
python -m memory_training.ollama_test \
  --run-dir /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/<run-id> \
  --ollama-url http://127.0.0.1:11435 \
  --memory-model <fine-tuned-ollama-tag> \
  --scenarios 91 92 93 94 95 \
  --memory-modes closed_loop teacher_forced \
  --memory-only
```

The report preserves closed-loop results under `memory`, writes the independent
axis under `memory_teacher_forced`, and records their F1 gaps in
`memory_mode_comparison`. Per-turn files are separated into `memory/` and
`memory_teacher_forced/` so runs can resume independently.

When `--quiz-model` is supplied with both memory modes, the same Turn/Final Quiz
set is evaluated twice. `quiz` uses closed-loop snapshots, while
`quiz_teacher_forced` uses snapshots generated from the Gold previous memory.
Their ESM/State F1/Tool F1/Arg Exact gaps are stored in `quiz_mode_comparison`,
and per-Quiz checkpoints live in `quiz/` and `quiz_teacher_forced/`. Existing
memory-only snapshots can be reused without memory inference via `--quiz-only`.

For the initial overfitting/generalization check, use S91-S92 as held-out Test and
S21 as a separately labelled Train diagnostic. S21 is never included in the Test
aggregate.

```bash
python -m memory_training.ollama_test \
  --run-dir /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/<run-id> \
  --ollama-url http://127.0.0.1:11435 \
  --memory-model <fine-tuned-ollama-tag> \
  --scenarios 91 92 \
  --train-diagnostic-scenarios 21 \
  --memory-modes closed_loop teacher_forced \
  --memory-only
```

### Isolated Qwen3.5 runtime

The system Ollama `0.6.5` on port `11434` is intentionally untouched. Qwen3.5 uses
an isolated Ollama `0.32.15` server, model store, and current llama.cpp checkout:

```text
/mnt/data/hj153lee/PalmClaw/on-device-memory-training/tools/ollama-v0.32.15
/mnt/data/hj153lee/PalmClaw/on-device-memory-training/tools/llama.cpp-qwen35
/mnt/data/hj153lee/PalmClaw/on-device-memory-training/ollama-qwen35-models
```

Manage the server on `127.0.0.1:11435` without affecting the existing service:

```bash
python -m memory_training.ollama_isolated start
python -m memory_training.ollama_isolated status
python -m memory_training.ollama_isolated stop
```

It defaults to GPU 6; override with `PALMCLAW_OLLAMA_GPU=<index>`. Initial GPU
discovery may take about one minute, so startup waits up to 120 seconds. The isolated
server already contains the official `qwen3.5:4b` Q4_K_M model for runtime smoke tests.

For a trained Qwen3.5 run, the exporter automatically selects the isolated llama.cpp,
recognizes both Qwen3.5 HF architectures, and passes `--no-mtp` to avoid the known
NextN/MTP extra-block mismatch. Register and test against the isolated server:

```bash
python -m memory_training.export_ollama \
  --run-dir /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/<qwen-run> \
  --ollama-url http://127.0.0.1:11435 --stage plan

python -m memory_training.export_ollama \
  --run-dir /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/<qwen-run> \
  --ollama-url http://127.0.0.1:11435 \
  --gguf-outtype bf16 --quantization NONE \
  --stage all --cleanup-intermediate --threads 16

python -m memory_training.ollama_test \
  --run-dir /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/<qwen-run> \
  --ollama-url http://127.0.0.1:11435 \
  --memory-model <fine-tuned-qwen-tag> --quiz-model <fixed-agent-tag>
```

Qwen3.5 enables thinking by default. Both Memory and Quiz runners explicitly send
`think=false`; otherwise thinking can consume the decode budget before the required
JSON or Tool Call is produced.

Large model downloads, checkpoints, caches, reports and MLflow artifacts are rooted at:

```text
/mnt/data/hj153lee/PalmClaw/on-device-memory-training/
```

## Tool-Calling SFT data

Build the compact Quiz SFT view before multitask training. The command resolves every
Quiz memory reference against the canonical Summary view, validates Gold Tool names and
arguments against the official 111 VehicleMemBench schemas, and atomically writes the
data plus its manifest.

```bash
python -m memory_training.quiz_sft
```

The generated `quiz_sft.jsonl` contains HF-style `messages` with assistant
`tool_calls`. The shared schemas are stored once in `vehicle_tools.json` and referenced
by SHA-256 instead of being duplicated in every row. For a single-call target, each
row selects the Gold Tool, one deterministic same-module hard negative and one
deterministic global random negative. Multi-call targets retain every Gold Tool and
the same two negative categories. This models an on-device selector without exposing
only the exact answer Tool or overflowing the 4K context.
`quiz_sft_manifest.json` records the S15-S80 Train, S81-S85 Validation, S86-S100 Test
policy; S1-S14 remain explicitly marked `excluded`.

Enable capped Memory/Quiz multitask SFT explicitly for a new run:

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m memory_training.train \
  --model qwen3.5-4b --method patch \
  --run-id qwen35-4b-patch-multitask-r1 \
  --multitask --quiz-total-passes 2
```

Each epoch retains every sampled Memory batch from S15-S80. The Quiz dataset is
deterministically shuffled into two exact full passes, divided across all epochs, and
distributed evenly among Memory batches. With 2,640 Quiz rows and three epochs this is
about 1,760 Quiz rows per epoch, rather than repeatedly cycling them to force 1:1.
The collator keeps batches task-homogeneous, masks all prompts/schemas, and applies
loss only to the Memory assistant output or assistant Tool Calls. Run metrics record
aggregate, `memory_loss`, and `quiz_loss` separately. Memory-only runs keep their
legacy S1-S80 sampler and are unaffected unless `--multitask` is passed.

At every epoch boundary, multitask runs may enable `--closed-loop-quiz` with S83 and
S84 as `--closed-loop-full-scenarios`. The runner replays every Turn, captures predicted
memory at every Quiz reference, and evaluates all 80 S83/S84 Tool-Calling Quizzes. A
separate, deterministic 50-row Gold-memory diagnostic is sampled from S81/S82/S85 only,
so the two axes do not overlap. The dashboard and epoch artifact report Gold Quiz and
closed-loop Quiz ESM/Tool F1/Arg Exact separately; closed-loop Quiz ESM is the primary
checkpoint-selection metric and final State F1 is its tie-break.

## S + Temporal Patch experiment

The mixed exporter preserves the original S1-S100 artifacts and encodes T1-T20 as
scenario IDs 101-120. Its fixed split is Train `S15-S80 + T1-T10`, Validation
`S81-S85 + T11`, and Test `S86-S100 + T12-T20`.

```bash
python -m memory_training.prepare_temporal_mixed_data

# Training (choose a free GPU)
memory_training/scripts/run_qwen35_4b_patch_temporal_mix.sh patch-r2 <gpu>
memory_training/scripts/run_qwen35_4b_patch_temporal_mix.sh temporal-patch-r1 <gpu>

# Final closed-loop Test after epoch 4
memory_training/scripts/run_qwen35_4b_patch_temporal_mix_test.sh patch-r2 <gpu>
memory_training/scripts/run_qwen35_4b_patch_temporal_mix_test.sh temporal-patch-r1 <gpu>
```

## Patch-only V1 auxiliary training

Prepare the fixed 10-scenario V1 augmentation from the trusted Cloud Patch R2 traces.
The exporter retains all turns, adds only four deterministic Final Quizzes per V1
scenario, verifies every reconstructed Patch by replay, and leaves the existing
Validation/Test partitions unchanged.

```bash
python -m memory_training.prepare_v1_patch_augmentation

# Start only after choosing a free GPU.
memory_training/scripts/run_qwen35_4b_patch_v1_10_mix.sh <gpu>
```

V1 S1-S50 are encoded as 201-250 and are training-only. The selected IDs, source
trace hashes, Quiz IDs, replay statistics, and rare full-memory reconstruction
fallbacks are recorded in the generated `manifest.json`.

Patch data:
`/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/hybrid-s1-s100-plus-temporal-t1-t20-patch-t1t10-v2`

Temporal-Patch data:
Temporal Patch 학습은 Terra plan 감사와 lossless replay를 통과한 다음 경로를 사용한다.

`/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/hybrid-s1-s100-plus-temporal-t1-t20-temporal-patch-t1t10-terra-audited-v2`

원본 deterministic `...temporal-patch-v1`은 비교와 재감사를 위해 그대로 보존한다.
