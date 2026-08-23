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
