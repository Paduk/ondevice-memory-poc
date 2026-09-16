# HVP04 external-adaptation record

HVP04 is the second **externally adapted, model-assembled pilot**. It is not a
human-authored evaluation set and is excluded from training and the proposed
human test. None of its source records overlap with HVP03.

## Direct dialogue sources

| Source | Records used | Origin | License | Turns |
| --- | --- | --- | --- | ---: |
| [Envisioned Voice Assistant Dialogues](https://github.com/SarahTheres/Envisioned-Voice-Assistant-Dialogues) | `119:open`, `189:open`, `92:search`, `62:open` | Written by study participants | CC BY 4.0 | 25 |
| [Audio2Tool](https://huggingface.co/datasets/RVtech/Audio2Tool) | tier 7: `1982`, `1989`, `1995`, `2000`, `2002`, `2004`; tier 4: `3993`; tier 6: `3965`, `4001`, `4005`, `4007` | LLM-generated query text | CC BY-NC 4.0 | 25 |

The exact downloaded-file SHA-256 values are recorded in
`scenario-source.json`. Audio2Tool's non-commercial license applies to this
adaptation.

## Transformation accounting

- All 50 turns carry an external source trace.
- 40 turns preserve the original text verbatim with only speaker-role
  normalization.
- 10 turns append the minimum information needed to express a persistent,
  self-contained vehicle preference.
- Human-written and synthetic-source turns are counted separately: 25 each.
- `source_trace` preserves dataset, record ID, record turn index, origin, reuse
  mode, and the unmodified source text.
- V2 memory operations and quizzes are newly authored and are not part of the
  external datasets.

HVP04 therefore measures external-language transfer only. It does not qualify
as evidence from a fully human-authored test set.
