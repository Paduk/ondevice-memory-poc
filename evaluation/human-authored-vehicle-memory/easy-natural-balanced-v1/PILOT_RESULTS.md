# HVE11–HVE20 balanced Easy pilot results

Evaluation date: 2026-09-12 UTC

This paired rerun measures only HVE11–HVE20. Gold updates, Quiz cutoffs, Quiz
wording, and Tool calls are identical before and after balancing.

- Local model: Granite 4 1B Patch multitask epoch-3 checkpoint as writer and reader.
- Cloud reference: GPT-5.6-luna zero-shot Patch as writer and reader.
- Subset: 10 scenarios, 300 turns, 60 UPDATEs, 240 NO_OPs, 160 Quizzes.
- Composite: `0.60 × Quiz ESM + 0.25 × Final-memory F1 + 0.15 × Update F1`.

## Paired aggregate comparison

| System | HVE11–20 version | Composite | Quiz ESM | Turn ESM | Final ESM | Final-memory F1 | Update F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Granite 4 1B Patch | Natural-explicit | 49.09 | 55.00 | 53.75 | 56.25 | 22.01 | 70.59 |
| Granite 4 1B Patch | **Natural-balanced** | **71.16** | **92.50** | **96.25** | **88.75** | 21.47 | 68.64 |
| GPT-5.6-luna Patch | Natural-explicit | 62.82 | 88.12 | 91.25 | 85.00 | 0.00* | 66.30 |
| GPT-5.6-luna Patch | Natural-balanced | 62.07 | 86.88 | 87.50 | 86.25 | 0.00* | 66.30 |

`*` Luna's free-form memory is incompatible with the grouped deterministic
Final-memory scorer; Quiz ESM is the meaningful Cloud reference.

## Scenario-level Granite Quiz ESM

| Scenario | Before | Balanced | Change |
| --- | ---: | ---: | ---: |
| HVE11 | 31.25 | 68.75 | +37.50 |
| HVE12 | 18.75 | 81.25 | +62.50 |
| HVE13 | 50.00 | 100.00 | +50.00 |
| HVE14 | 56.25 | 87.50 | +31.25 |
| HVE15 | 50.00 | 93.75 | +43.75 |
| HVE16 | 75.00 | 100.00 | +25.00 |
| HVE17 | 62.50 | 100.00 | +37.50 |
| HVE18 | 100.00 | 100.00 | 0.00 |
| HVE19 | 37.50 | 93.75 | +56.25 |
| HVE20 | 68.75 | 100.00 | +31.25 |

The local model improves by 37.5 percentage points on average. The strict
memory F1 and UPDATE-decision F1 do not improve, showing that the gain comes
mainly from making each positive memory statement single-intent and easier to
read, not from eliminating all false UPDATEs. Luna was already robust and
changes by only -1.25 points, which is consistent with ordinary repeated Cloud
run variation.

HVE11 and HVE12 contain the remaining concentrated errors. Further editing
these items after inspecting model outputs would adapt the test to this specific
checkpoint. The recommended protocol is therefore to freeze this version and
use it only as a declared Easy calibration/sanity set, never as the main Human
Test or as evidence of unbiased generalization.
