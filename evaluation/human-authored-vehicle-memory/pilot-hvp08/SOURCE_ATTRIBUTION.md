# HVP08 external-adaptation record

HVP08 is an **externally adapted, model-assembled pilot**. It is
excluded from training and does not qualify as a human-authored evaluation
set. Its external records do not overlap with any other external-adapted HVP pilot.

| Source | Records used | Origin | License | Turns |
| --- | --- | --- | --- | ---: |
| [Envisioned Voice Assistant Dialogues](https://github.com/SarahTheres/Envisioned-Voice-Assistant-Dialogues) | `35:search`, `38:weather`, `57:search` | Written by study participants | CC BY 4.0 | 25 |
| [Audio2Tool](https://huggingface.co/datasets/RVtech/Audio2Tool) | `tier7:2024`, `tier7:2088`, `tier7:2145`, `tier5:3963`, `tier5:4003`, `tier5:4111`, `tier5:4239`, `tier4:4096`, `tier5:4272`, `tier4:3988`, `tier5:4005`, `tier5:4113`, `tier5:4002` | LLM-generated query text | CC BY-NC 4.0 | 25 |

- All 50 turns carry an external source trace.
- 40 turns preserve the source text verbatim
  with only speaker-role normalization.
- 10 turns append the minimum information needed to express
  a persistent, self-contained VehicleMemBench preference.
- The V2 memory operations and quizzes are newly authored for this pilot.
- Audio2Tool's non-commercial license applies to this adaptation.

The exact record IDs, original turn text, downloaded-file SHA-256 values, and
reuse mode are preserved in `scenario-source.json`.
