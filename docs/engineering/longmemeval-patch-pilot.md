# LongMemEval Patch pilot

This pilot adapts an existing V2 Patch checkpoint without additional CloudLLM
labeling. It uses official LongMemEval `has_answer` turn markers, question
metadata, and final answers. The public Mem-alpha 200/300 split is reused,
restricted to knowledge-update and temporal-reasoning.

## Leakage and label policy

- Development/test questions: 91/120, with zero question-ID overlap.
- Eight development abstention questions are audit-only, not SFT targets.
- The remaining 83 questions are split into 75 train and 8 validation questions.
- For this deliberately simple pilot, every `has_answer=false` user turn is
  labeled `NO_OP`. This is question-evidence supervision, not a claim that those
  turns have no generally useful information.
- Knowledge update becomes `add` followed by `replace`; temporal facts become
  chronological `add` operations. Every operation is replay-validated.
- Each usable development question also contributes one natural-language final
  QA row: `[gold final memory + question date + question] -> official answer`.
- Eight abstention questions have no deterministic memory target and remain
  audit-only. All remaining 83 questions are used for training; no internal
  validation split or validation step is used in this trend-finding run.
- The downstream question is never shown to the on-device memory model.

## Gated workflow

First inspect the generated statistics; this does not start training:

```bash
DATA_ROOT=/mnt/data/hj153lee/PalmClaw/evaluation/longmemeval-patch-pilot-v1 \
  memory_training/scripts/run_longmemeval_patch_pilot.sh stats
```

After approving the statistics, start three fresh multitask adaptation epochs from
an existing checkpoint (new optimizer/scheduler, copied LoRA initialization).
Each epoch includes all available NO_OP rows and one pass over final-QA rows:

```bash
CUDA_VISIBLE_DEVICES=GPU_ID \
INIT_CHECKPOINT=/absolute/path/to/checkpoint \
MODEL=granite4-1b \
  memory_training/scripts/run_longmemeval_patch_pilot.sh train
```

If `INIT_CHECKPOINT` is omitted, the launcher uses the current Granite 1B Patch
V1-style best checkpoint (`epoch-03`).

Then process all 120 held-out questions turn-wise. This produces final memories
and `reader_inputs.jsonl`; use the same fixed CloudLLM reader and official
LongMemEval judge for every method.

```bash
CUDA_VISIBLE_DEVICES=GPU_ID \
MODEL=granite4-1b \
  memory_training/scripts/run_longmemeval_patch_pilot.sh memory-test
```

The held-out memory pass contains about 29,407 user turns, so a small `--limit`
smoke test should precede the full run.

Run the fixed reader to create an official-compatible hypotheses JSONL:

```bash
python -m memory_training.run_longmemeval_reader \
  --inputs "$WORKSPACE/runs/$RUN_ID/longmemeval-test/reader_inputs.jsonl" \
  --output "$WORKSPACE/runs/$RUN_ID/longmemeval-reader" \
  --model FIXED_READER_MODEL
```

Finally score `hypotheses.jsonl` with LongMemEval's official
`src/evaluation/evaluate_qa.py`. Keep both reader and judge models fixed across
all compared memory methods.
