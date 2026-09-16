# HVP13 external-adaptation record

HVP13 is an **externally adapted, model-assembled pilot**. It is
excluded from training and does not qualify as a human-authored evaluation
set. Its external records do not overlap with any other external-adapted HVP pilot.

| Source | Records used | Origin | License | Turns |
| --- | --- | --- | --- | ---: |
| [Envisioned Voice Assistant Dialogues](https://github.com/SarahTheres/Envisioned-Voice-Assistant-Dialogues) | `20:open`, `49:weather`, `10:weather` | Written by study participants | CC BY 4.0 | 25 |
| [Audio2Tool](https://huggingface.co/datasets/RVtech/Audio2Tool) | `tier7:2032`, `tier7:2063`, `tier7:2146`, `tier6:3966`, `tier6:4006`, `tier6:3962`, `tier6:4234`, `tier5:4134`, `tier5:4092`, `tier5:3980`, `tier5:3966`, `tier6:3968`, `tier6:4008` | LLM-generated query text | CC BY-NC 4.0 | 25 |

- All 50 turns carry an external source trace.
- 40 turns preserve the source text verbatim
  with only speaker-role normalization.
- 10 turns append the minimum information needed to express
  a persistent, self-contained VehicleMemBench preference.
- The V2 memory operations and quizzes are newly authored for this pilot.
- Audio2Tool's non-commercial license applies to this adaptation.

The exact record IDs, original turn text, downloaded-file SHA-256 values, and
reuse mode are preserved in `scenario-source.json`.
