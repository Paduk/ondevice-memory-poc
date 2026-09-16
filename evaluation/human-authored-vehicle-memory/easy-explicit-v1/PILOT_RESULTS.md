# HVE01–HVE10 Easy pilot results

Evaluation date: 2026-09-12 UTC

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

| System | Composite | Quiz ESM | Turn ESM | Final ESM | Final-memory F1 | Update F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Granite 4 1B Patch | 54.65 | 61.25 | 61.25 | 61.25 | 26.05 | 75.91 |
| GPT-5.6-luna Patch | 63.63 | 89.38 | 92.50 | 86.25 | 0.00* | 66.67 |

`*` Luna's free-form flat memory does not match the grouped Gold-memory line
representation and also stores irrelevant sourced facts. Consequently, the
deterministic slot/line scorer reports zero Final-memory F1 even when the
correct durable values remain readable. Luna Composite is therefore
representation-sensitive; Quiz ESM is the more meaningful downstream reference
until Cloud memories receive a canonical normalization or semantic audit.

## Scenario-level Quiz ESM

| Scenario | Granite Patch | Luna Patch |
| --- | ---: | ---: |
| HVE01 | 68.75 | 100.00 |
| HVE02 | 43.75 | 100.00 |
| HVE03 | 56.25 | 43.75 |
| HVE04 | 62.50 | 87.50 |
| HVE05 | 12.50 | 87.50 |
| HVE06 | 68.75 | 100.00 |
| HVE07 | 68.75 | 81.25 |
| HVE08 | 87.50 | 93.75 |
| HVE09 | 93.75 | 100.00 |
| HVE10 | 50.00 | 100.00 |

## Writer diagnostics

| System | Update precision | Update recall | False UPDATEs / 240 NO_OPs | Missed UPDATEs | Invalid outputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Granite 4 1B Patch | 67.53 | 86.67 | 25 | 8 | 9 |
| GPT-5.6-luna Patch | 50.43 | 98.33 | 58 | 1 | 1 |

The easy Quiz wording works as intended, but the full pipeline is not yet a
95% ceiling set. Most remaining difficulty is upstream: separating durable
profile clauses from one-time sourced commands and refusing unrelated NO_OP
facts. Luna's low HVE03 score comes mainly from retaining the pre-replacement
temperature and volume and omitting the co-driver passenger temperature.

The Cloud run used approximately `$0.0673` in total (`$0.0500` memory and
`$0.0172` Quiz).
