# HVP05 external-adaptation record

HVP05 is an **externally adapted, model-assembled pilot**. It is
excluded from training and does not qualify as a human-authored evaluation
set. Its external records do not overlap with any other external-adapted HVP pilot.

| Source | Records used | Origin | License | Turns |
| --- | --- | --- | --- | ---: |
| [Envisioned Voice Assistant Dialogues](https://github.com/SarahTheres/Envisioned-Voice-Assistant-Dialogues) | `32:open`, `44:open`, `45:open`, `21:search` | Written by study participants | CC BY 4.0 | 25 |
| [Audio2Tool](https://huggingface.co/datasets/RVtech/Audio2Tool) | `tier7:2072`, `tier7:2078`, `tier7:2042`, `tier7:2043`, `tier7:2056`, `tier7:2105`, `tier6:4115`, `tier6:4269`, `tier6:4273` | LLM-generated query text | CC BY-NC 4.0 | 25 |

- All 50 turns carry an external source trace.
- 43 turns preserve the source text verbatim
  with only speaker-role normalization.
- 7 turns append the minimum information needed to express
  a persistent, self-contained VehicleMemBench preference.
- The V2 memory operations and quizzes are newly authored for this pilot.
- Audio2Tool's non-commercial license applies to this adaptation.

The exact record IDs, original turn text, downloaded-file SHA-256 values, and
reuse mode are preserved in `scenario-source.json`.
