# HVE01–HVE20 natural-explicit pilot results

Evaluation date: 2026-09-12 UTC

Both systems performed the same full closed loop: turn-wise Patch memory
generation followed by Quiz answering from their own predicted memory.

- Local model: Granite 4 1B Patch multitask epoch-3 checkpoint as writer and reader.
- Cloud reference: GPT-5.6-luna zero-shot Patch as writer and reader, reasoning effort `low`.
- Set: 20 scenarios, 600 turns, 120 UPDATEs, 480 NO_OPs, and 320 Quizzes
  (160 Turn + 160 Final).
- Composite: `0.60 × Quiz ESM + 0.25 × Final-memory F1 + 0.15 × Update F1`.

## Aggregate results

| System | Composite | Quiz ESM | Turn ESM | Final ESM | Final-memory F1 | Update F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Granite 4 1B Patch | **54.78** | **62.81** | 60.62 | 65.00 | 23.55 | 74.68 |
| GPT-5.6-luna Patch | **63.01** | **88.44** | 89.38 | 87.50 | 0.00* | 66.30 |

`*` Luna writes free-form flat memory rather than the grouped Gold-memory
representation. The deterministic structural scorer therefore reports zero
Final-memory F1 even when the durable value remains readable. Luna Composite is
representation-sensitive; Quiz ESM is the meaningful downstream reference.

## Existing and newly added halves

| System | Scenarios | Composite | Quiz ESM | Turn ESM | Final ESM | Final-memory F1 | Update F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Granite 4 1B Patch | HVE01–10 | 60.57 | 70.62 | 67.50 | 73.75 | 25.10 | 79.45 |
| Granite 4 1B Patch | HVE11–20 | 49.09 | 55.00 | 53.75 | 56.25 | 22.01 | 70.59 |
| GPT-5.6-luna Patch | HVE01–10 | 63.19 | 88.75 | 87.50 | 90.00 | 0.00* | 66.30 |
| GPT-5.6-luna Patch | HVE11–20 | 62.82 | 88.12 | 91.25 | 85.00 | 0.00* | 66.30 |

## Scenario-level Quiz ESM

| Scenario | Granite Patch | Luna Patch |
| --- | ---: | ---: |
| HVE01 | 68.75 | 75.00 |
| HVE02 | 93.75 | 87.50 |
| HVE03 | 100.00 | 93.75 |
| HVE04 | 62.50 | 81.25 |
| HVE05 | 62.50 | 100.00 |
| HVE06 | 75.00 | 81.25 |
| HVE07 | 43.75 | 100.00 |
| HVE08 | 68.75 | 87.50 |
| HVE09 | 56.25 | 93.75 |
| HVE10 | 75.00 | 87.50 |
| HVE11 | 31.25 | 93.75 |
| HVE12 | 18.75 | 100.00 |
| HVE13 | 50.00 | 93.75 |
| HVE14 | 56.25 | 81.25 |
| HVE15 | 50.00 | 81.25 |
| HVE16 | 75.00 | 81.25 |
| HVE17 | 62.50 | 81.25 |
| HVE18 | 100.00 | 93.75 |
| HVE19 | 37.50 | 87.50 |
| HVE20 | 68.75 | 87.50 |

## Interpretation

Naturalizing the original HVE01–10 did not make that half harder for the local
model: Granite Quiz ESM increased from 61.25 on rigid Easy-explicit v1 to 70.62,
while Luna remained similar (89.38 to 88.75). Thus natural wording itself is not
the cause of the lower 20-scenario aggregate.

The decline is concentrated in the newly sourced HVE11–20: Granite falls to
55.00 while Luna remains stable at 88.12. Granite's update recall is 100% on the
new half, but update precision falls from 67.44% to 54.55%, indicating additional
NO_OP facts are being stored and polluting the small model's memory. HVE11,
HVE12, and HVE19 are the principal local-model outliers; HVE18 reaches 100%.
This is therefore a scenario-composition sensitivity rather than a uniform Gold
or Tool-schema failure.
