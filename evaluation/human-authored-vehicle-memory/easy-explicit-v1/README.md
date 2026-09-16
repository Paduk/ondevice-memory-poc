# Easy explicit human-validation pilots

This directory contains 20 scenarios in an intentionally easy evaluation stratum for checking
whether an apparent performance ceiling is reachable before interpreting the
harder HVP results.

## Fixed scenario contract

Each scenario has:

- 30 chronological dialogue turns from three speakers: two human memory owners
  and one assistant;
- 6 explicit durable vehicle-memory updates and 24 `NO_OP` turns;
- 4 `ADD` and 2 `REPLACE` operations, leaving four active memory slots;
- 8 Turn Quizzes and 8 Final Quizzes;
- one named owner, one memory fact, and one Tool call per Quiz;
- no implicit coreference, conditional inference, or multi-hop reasoning;
- no answer value exposed in the Quiz text.

The two replaced slots are the primary driver's temperature and the co-driver's
music volume. The other active slots are the primary driver's ambient color and
the co-driver's passenger-zone temperature. Values are deliberately distinct
within each scenario where owner confusion could otherwise be hidden.

## Provenance and status

Every turn is adapted from a record in Envisioned Voice Assistant Dialogues or
Audio2Tool and keeps record-level source tracing. Human-authored external text is
used verbatim after role normalization. Each vehicle UPDATE starts with a
source-derived, one-time request and adds an explicit durable-profile sentence.

These are model-assembled pilots, not yet human-authored evaluation items.
`eligible_for_human_test` therefore remains `false`. A separate human must
rewrite/approve the adapted UPDATE sentence and independently verify its label
before the corpus can support a human-authored-data claim.

## Validation

`CORPUS_VALIDATION.json` is produced by:

```bash
cd /home/hj153lee/PalmClaw
PYTHONPATH=/home/hj153lee/PalmClaw/ubuntu/src \
  /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python \
  -m memory_training.scripts.validate_easy_hvp_pilots --count 20
```

The validator checks source exactness and disjointness, the update state
machine, owner/persistence explicitness, answer non-leakage, derived V2 row
counts, Patch replay, Tool schemas, and the official VehicleWorld simulator.
