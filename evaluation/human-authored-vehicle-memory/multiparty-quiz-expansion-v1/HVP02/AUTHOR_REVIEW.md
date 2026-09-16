# HVP02 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This is a model-assisted author review, not evidence that the scenario is
human-authored. Review was performed without Summary, Patch, or Delta-v3
predictions.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp02-mp01` | PASS with note | Audio2Tool record `tier7:548` describes lowering the main speaker to a calm background level and maps that request to level 30. The added final sentence makes 30 an explicit persistent in-car preference. A human should smooth the transition from the original room-speaker context to the vehicle. |
| `hvp02-mp02` | PASS with note | Audio2Tool record `tier7:332` directly establishes locking up before going out. The final sentence maps the home front-door request to Avery's all-vehicle-door departure routine. A human should smooth this cross-domain transition. |
| `hvp02-mp03` | PASS | Audio2Tool record `tier7:2067` directly requests muting the navigation voice. The final sentence makes that vehicle-navigation preference persistent. |

All non-adapted turns are verbatim source text. Only the final user turn of
each inserted session receives a persistence sentence.

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp02-turn-05`–`10` | Coreference resolution | PASS. Each indirect location, object, or pronoun resolves to active Robin memory at its cutoff. The two actions that equal the simulator's default state are paired with another grounded call so every quiz has an observable state transition. |
| `hvp02-turn-11`–`12` | Error correction | PASS. The gold values are Robin's final corrections, 52 rather than 45 and 5 rather than 10. |
| `hvp02-turn-13`–`18` | Preference conflict | PASS. Music volume (30 vs. 5), all-door departure state (locked vs. unlocked), and navigation voice mode (mute vs. detailed) each form a same-Tool, different-owner conflict, tested once in both owner directions. |
| `hvp02-turn-19`–`22` | State shift | PASS. The questions follow Robin's explicit volume sequence 52→25→70→5 and always select the value active at the stated cutoff. |
| `hvp02-final-07` | Coreference resolution | PASS. “They/them” resolves to Avery; the three calls use only Avery's final active memories for music, all-door departure state, and navigation guidance. |

## Remaining human-validation requirement

- Audio2Tool text is synthetic and licensed CC BY-NC 4.0.
- A human must rewrite the room-speaker and home-door transitions, confirm the
  naturalness of every adapted exchange, and independently verify all Quiz
  questions before this can be called a human validation scenario.
- Human reviewers must remain blind to method predictions and scores.
