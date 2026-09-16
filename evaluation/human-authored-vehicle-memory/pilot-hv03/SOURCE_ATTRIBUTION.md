# HVP03 external-adaptation record

HVP03 is an **externally adapted, model-assembled pilot**, not a human-authored
evaluation set. It is excluded from training and from the proposed human test.
Its purpose is to test how much natural external dialogue can be retained while
converting vehicle-setting evidence to the VehicleMemBench V2 contract.

## Direct dialogue sources

| Source | Records used | Origin | License | Turns |
| --- | --- | --- | --- | ---: |
| [Envisioned Voice Assistant Dialogues](https://github.com/SarahTheres/Envisioned-Voice-Assistant-Dialogues) | `66:open`, `172:open`, `9:volume` | Written by study participants | CC BY 4.0 | 25 |
| [Audio2Tool](https://huggingface.co/datasets/RVtech/Audio2Tool) | tier 7: `1981`, `1983`, `1987`, `1988`, `1996`, `1999`; tier 6: `3967`, `3977`, `3979` | LLM-generated query text | CC BY-NC 4.0 | 25 |

The downloaded source files and their SHA-256 hashes are recorded in
`scenario-source.json`. Audio2Tool's non-commercial license means HVP03 must
not be used in a commercial artifact without separate permission.

## Transformation accounting

- 50/50 turns have an external source record.
- 40 turns preserve the source text verbatim; only the local speaker role is
  normalized.
- 10 turns retain the source utterance but add the minimum text needed to make
  a persistent vehicle preference self-contained.
- Each turn stores dataset, record ID, record turn index, origin class, reuse
  mode, and original text under `source_trace`.
- Memory operations and quizzes are newly authored for this pilot and are not
  attributed to either external dataset.

Because only 25 turns originate from human-written data and the scenario was
assembled by a model, HVP03 can support an **external-language transfer pilot**
claim only. It cannot support a real-human-test claim. A final human validation
set still requires independent human writing or substantive human rewriting
and review under a preregistered protocol.

## Inspected but not copied

- [CAR-Bench](https://huggingface.co/datasets/johanneskirmayr/car-bench-dataset)
  informed the vehicle-assistant task framing, but its generated task records
  were not copied into HVP03.
- KVRET informed the multi-turn in-car dialogue structure, but no text was
  copied because the redistribution terms of the transformed copy inspected
  for this pilot were not explicit.
- Sources with access restrictions or no-derivatives terms were excluded.
