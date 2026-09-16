# HVP03 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This is a model-assisted author review performed without access to any method's
predictions or scores. It does not make the scenario human-authored.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp03-mp01` | PASS with note | Audio2Tool `tier7:254` explicitly resolves a fan request to speed 7. The persistence sentence scopes it to Jordan's front vehicle zone; a human should smooth the home-fan transition. |
| `hvp03-mp02` | PASS with note | Audio2Tool `tier7:542` explicitly resolves an uncomfortable speaker level to 20. The persistence sentence scopes it to Jordan's after-10pm in-car rule; a human should smooth the room-speaker transition. |
| `hvp03-mp03` | PASS with note | Audio2Tool `tier6:3974` directly corrects passenger-seat cooling from 2 to 1. Its transcript duplicates the separately sourced `tier6:3973` variant used in HVP02, although the source record and audio item are disjoint; the required human rewrite must make this exchange textually distinct. |

Only the final user turn in each inserted session receives a persistence
sentence; all preceding text is verbatim and source-traced.

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp03-turn-05`–`10` | Coreference resolution | PASS. The rear-left child seat, front airflow, all windows, passenger seat, late-night rule, and steering-wheel heater all resolve to active Avery memories at their cutoffs. |
| `hvp03-turn-11`–`12` | Error correction | PASS. The answers retain passenger cooling 2 rather than 3 and steering-wheel heat on rather than the withdrawn off request. |
| `hvp03-turn-13`–`18` | Preference conflict | PASS. Fan speed (7 vs. 3), after-10pm music (20 vs. 4), and passenger ventilation (1 vs. 2) are exact same-Tool and same-target owner conflicts, each tested in both directions. |
| `hvp03-turn-19`–`22` | State shift | PASS. The selected values match the active snapshots across passenger cooling 3→2, front fan 5→3, and steering-wheel heat off→on. |
| `hvp03-final-07` | Coreference resolution | PASS. Jordan is explicitly the active user and “they/them/that seat” resolves to Jordan and the front passenger seat; all three answers come from Jordan's final memory. |

## Remaining human-validation requirement

- Audio2Tool text is synthetic and CC BY-NC 4.0.
- A human must rewrite the two cross-device transitions and the duplicated
  passenger-cooling transcript, then independently verify every Quiz.
- Human reviewers must remain blind to Summary, Patch, and Delta-v3 outputs.
