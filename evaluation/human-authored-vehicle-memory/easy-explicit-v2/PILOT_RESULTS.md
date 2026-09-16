# HVE01–HVE10 Easy-explicit v2 pilot results

Evaluation date: 2026-09-12 UTC

Easy-explicit v2 is a deliberately label-transparent ceiling/sanity set. It is
not a replacement for the natural human-authored HVP test. Relative to v1, it
keeps the same owners, durable facts, timestamps, update labels, Quiz cutoffs,
and Gold Tool calls while making the language easier:

- each UPDATE contains exactly one explicit permanent vehicle-profile fact;
- each REPLACE explicitly identifies the new value and says to discard the old;
- each non-vehicle user turn explicitly says it is a one-time request and not a
  saved vehicle preference;
- Quiz wording uses a direct and uniform owner–setting request without revealing
  the answer value.

Both systems performed the full closed loop: turn-wise Patch memory generation
followed by Quiz answering from their own predicted memory.

- Local Patch: Granite 4 1B Patch multitask, epoch 3 checkpoint, as both writer
  and reader.
- Cloud reference: GPT-5.6-luna zero-shot Patch, as both writer and reader,
  reasoning effort `low`.
- Evaluation set: 10 scenarios, 300 dialogue turns, 60 UPDATEs, 240 NO_OPs,
  and 160 Quizzes (80 Turn + 80 Final).
- Composite: `0.60 × Quiz ESM + 0.25 × Final-memory F1 + 0.15 × Update F1`.

## Aggregate results

| System | Version | Composite | Quiz ESM | Turn ESM | Final ESM | Final-memory F1 | Update F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Granite 4 1B Patch | v1 | 54.65 | 61.25 | 61.25 | 61.25 | 26.05 | 75.91 |
| Granite 4 1B Patch | **v2** | **76.48** | **95.63** | **96.25** | **95.00** | **26.43** | **83.33** |
| GPT-5.6-luna Patch | v1 | 63.63 | 89.38 | 92.50 | 86.25 | 0.00* | 66.67 |
| GPT-5.6-luna Patch | **v2** | **71.49** | **96.25** | **97.50** | **95.00** | 0.00* | **91.60** |

`*` Luna writes free-form flat memories rather than the grouped Gold-memory
line representation. The deterministic structural scorer therefore reports
zero Final-memory F1 even when the durable value is readable. Luna Composite is
representation-sensitive; Quiz ESM is the meaningful downstream reference.

## Easy-explicit v2 scenario-level Quiz ESM

| Scenario | Granite Patch | Luna Patch |
| --- | ---: | ---: |
| HVE01 | 100.00 | 100.00 |
| HVE02 | 100.00 | 100.00 |
| HVE03 | 100.00 | 100.00 |
| HVE04 | 100.00 | 100.00 |
| HVE05 | 100.00 | 81.25 |
| HVE06 | 93.75 | 100.00 |
| HVE07 | 68.75 | 100.00 |
| HVE08 | 93.75 | 87.50 |
| HVE09 | 100.00 | 100.00 |
| HVE10 | 100.00 | 93.75 |

## Writer diagnostics

| System | Version | Update precision | Update recall | False UPDATEs / 240 NO_OPs | Missed UPDATEs | Invalid outputs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Granite 4 1B Patch | v1 | 67.53 | 86.67 | 25 | 8 | 9 |
| Granite 4 1B Patch | **v2** | **71.43** | **100.00** | 24 | **0** | **0** |
| GPT-5.6-luna Patch | v1 | 50.43 | 98.33 | 58 | 1 | 1 |
| GPT-5.6-luna Patch | **v2** | **84.51** | **100.00** | **11** | **0** | **0** |

## Interpretation

The change achieved its intended calibration target: both the trained 1B model
and the zero-shot Cloud reference exceed 95% aggregate Quiz ESM, and neither
misses a Gold UPDATE. This supports using v2 as a pipeline ceiling/sanity check.
It must not be presented as natural-dialogue generalization because its UPDATE
and NO_OP wording exposes the intended memory decision explicitly.

The remaining low Final-memory F1 is primarily a strict representation/extra-
fact issue rather than a downstream answerability ceiling: Quiz ESM remains
above 95%. Scenario HVE07 is the main Granite outlier, while Luna's errors are
concentrated in HVE05 and HVE08 rather than distributed across the corpus.
