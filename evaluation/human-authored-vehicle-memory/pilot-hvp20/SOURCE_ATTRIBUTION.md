# HVP20 external-adaptation record

HVP20 is an **externally adapted, model-assembled pilot**. It is
excluded from training and does not qualify as a human-authored evaluation
set. Its external records do not overlap with any other external-adapted HVP pilot.

| Source | Records used | Origin | License | Turns |
| --- | --- | --- | --- | ---: |
| [Envisioned Voice Assistant Dialogues](https://github.com/SarahTheres/Envisioned-Voice-Assistant-Dialogues) | `102:search`, `71:conversational`, `63:music` | Written by study participants | CC BY 4.0 | 25 |
| [Audio2Tool](https://huggingface.co/datasets/RVtech/Audio2Tool) | `tier7:1907`, `tier7:2006`, `tier7:2139`, `tier5:4204`, `tier4:4097`, `tier4:4102`, `tier4:4122`, `tier6:4236`, `tier4:4224`, `tier6:4004`, `tier5:4206`, `tier6:4208`, `tier6:4210` | LLM-generated query text | CC BY-NC 4.0 | 25 |

- All 50 turns carry an external source trace.
- 40 turns preserve the source text verbatim
  with only speaker-role normalization.
- 10 turns append the minimum information needed to express
  a persistent, self-contained VehicleMemBench preference.
- The V2 memory operations and quizzes are newly authored for this pilot.
- Audio2Tool's non-commercial license applies to this adaptation.

The exact record IDs, original turn text, downloaded-file SHA-256 values, and
reuse mode are preserved in `scenario-source.json`.
