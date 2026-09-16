# HVP17 external-adaptation record

HVP17 is an **externally adapted, model-assembled pilot**. It is
excluded from training and does not qualify as a human-authored evaluation
set. Its external records do not overlap with any other external-adapted HVP pilot.

| Source | Records used | Origin | License | Turns |
| --- | --- | --- | --- | ---: |
| [Envisioned Voice Assistant Dialogues](https://github.com/SarahTheres/Envisioned-Voice-Assistant-Dialogues) | `43:conversational`, `80:weather`, `51:open` | Written by study participants | CC BY 4.0 | 25 |
| [Audio2Tool](https://huggingface.co/datasets/RVtech/Audio2Tool) | `tier7:1902`, `tier7:2003`, `tier7:2021`, `tier5:4107`, `tier5:4203`, `tier4:3992`, `tier4:4101`, `tier4:4221`, `tier4:4223`, `tier4:4257`, `tier5:4110`, `tier6:4111`, `tier5:4205` | LLM-generated query text | CC BY-NC 4.0 | 25 |

- All 50 turns carry an external source trace.
- 40 turns preserve the source text verbatim
  with only speaker-role normalization.
- 10 turns append the minimum information needed to express
  a persistent, self-contained VehicleMemBench preference.
- The V2 memory operations and quizzes are newly authored for this pilot.
- Audio2Tool's non-commercial license applies to this adaptation.

The exact record IDs, original turn text, downloaded-file SHA-256 values, and
reuse mode are preserved in `scenario-source.json`.
