# Easy natural-explicit HVE scenarios

This directory contains the naturalized Easy calibration set, `HVE01`–`HVE20`.
It preserves the complete Gold semantics and external source records of
`easy-explicit-v1`, but replaces the rigid durable-memory suffixes and Quiz
templates with conversational, still-explicit wording.

Each scenario retains 30 turns, two human memory owners, 6 UPDATEs, 24 NO_OPs,
4 ADDs, 2 REPLACEs, 8 Turn Quizzes, and 8 Final Quizzes. Every memory-bearing
utterance still explicitly communicates the owner, exact value, persistence,
and—when applicable—replacement of the old value. Quiz text never exposes the
answer value.

The external source clause and its record-level trace remain unchanged. The
naturalized preference clauses and Quiz wording are model-assisted rewrites,
not independent human authorship. Consequently every scenario remains
`pilot_excluded` and `eligible_for_human_test: false` until a human independently
rewrites or approves it.

Validation:

```bash
cd /home/hj153lee/PalmClaw
PYTHONPATH=/home/hj153lee/PalmClaw/ubuntu/src \
  /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python \
  -m memory_training.scripts.validate_easy_natural_hvp --count 20
```
