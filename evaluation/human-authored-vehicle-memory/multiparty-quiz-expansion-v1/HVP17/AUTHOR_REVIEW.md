# HVP17 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to Summary, Patch, and Delta-v3 outputs.
All inserted turns and 19 added Quiz items were checked against their source,
owner, condition, cutoff, active memory value, and Gold Tool call.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp17-mp01` | PASS with normalization required | Audio2Tool `tier7:68` already says “for the dash” and explicitly chooses soft blue. The dialogue is retained verbatim; HSB hue 210 is normalized to `blue`, while brightness remains turn-local and only persistence is appended. |
| `hvp17-mp02` | PASS with rewrite required | Audio2Tool `tier7:231` explicitly chooses 19°C after a too-hot complaint. The same three-turn dialogue act is rewritten as all-zone vehicle climate and made durable. |
| `hvp17-mp03` | PASS with high-risk rewrite required | Audio2Tool `tier7:480` changes an unsupported halfway request to fully Open. The correction is preserved in a disclosed garage-to-rear-window aperture rewrite, mapped to 100%; the object, child-occupant condition, and safety wording require independent approval. |

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp17-turn-05`–`09` | Coreference resolution | PASS. Each question selects one owner and activates normal climate/lighting, unnamed navigation, heavy fog, stale air, normal driving, or the children-stuffy condition as required. Jordan's combined item occurs after all three inserted memories. |
| `hvp17-turn-10`–`11` | Error correction | PASS. The prompts select Blair's final ambient-color and temperature revisions without revealing target values. The white action is paired with the active temperature to avoid a simulator-default-only target. |
| `hvp17-turn-12`–`18` | Preference conflict | PASS. The questions cover three genuine matched axes in both owner directions: ambient blue versus white, all-zone 19 versus 27°C, and rear-right window 100 versus 60 under the same child-occupant condition. |
| `hvp17-turn-19`–`22` | State shift | PASS. The cutoffs select ambient pink→red→white and the original 18°C temperature before replacement. The final white action is paired with the active 18°C setting. |
| `hvp17-final-07` | Preference conflict | PASS. The final item selects Blair against Jordan on all three same-condition axes. |

## Manual checks and restraint

- Selected three source records unused by the frozen HVP01–HVP20 corpus and all
  previously completed extensions.
- Retained exact source text in provenance even for rewritten turns.
- Retained only explicit targets: blue from HSB hue 210, 19°C, and fully Open.
  Brightness values were excluded from persistent memory.
- Matched Tool, selector, and condition for every owner alternative before
  labeling it as a preference conflict.
- Checked all Quiz cutoffs against active memory, including Blair's two ambient
  replacements and later temperature replacement.
- No added question states a distinguishing color, temperature, or window
  percentage.

## Remaining human-validation requirement

- Audio2Tool query text is synthetic and CC BY-NC 4.0.
- All three sessions require blind independent approval; `hvp17-mp03` must be
  reviewed especially for the object transformation and child-passenger safety.
- Every adaptation and Quiz must receive blind human validation before this
  scenario is described as human-authored or human-validated.
