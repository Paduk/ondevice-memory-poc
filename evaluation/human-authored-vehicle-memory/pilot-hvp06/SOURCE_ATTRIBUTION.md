# HVP06 external-adaptation record

HVP06 is an **externally adapted, model-assembled pilot**. It is
excluded from training and does not qualify as a human-authored evaluation
set. Its external records do not overlap with any other external-adapted HVP pilot.

| Source | Records used | Origin | License | Turns |
| --- | --- | --- | --- | ---: |
| [Envisioned Voice Assistant Dialogues](https://github.com/SarahTheres/Envisioned-Voice-Assistant-Dialogues) | `81:open`, `110:open`, `196:open`, `132:open` | Written by study participants | CC BY 4.0 | 25 |
| [Audio2Tool](https://huggingface.co/datasets/RVtech/Audio2Tool) | `tier7:2081`, `tier7:2090`, `tier7:2057`, `tier7:2117`, `tier7:2119`, `tier7:2135`, `tier6:4117`, `tier6:3989`, `tier6:4233` | LLM-generated query text | CC BY-NC 4.0 | 25 |

- All 50 turns carry an external source trace.
- 43 turns preserve the source text verbatim
  with only speaker-role normalization.
- 7 turns append the minimum information needed to express
  a persistent, self-contained VehicleMemBench preference.
- The V2 memory operations and quizzes are newly authored for this pilot.
- Audio2Tool's non-commercial license applies to this adaptation.

The exact record IDs, original turn text, downloaded-file SHA-256 values, and
reuse mode are preserved in `scenario-source.json`.
