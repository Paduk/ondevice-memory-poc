# HVP09 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to Summary, Patch, and Delta-v3 outputs.
All inserted turns and 19 added Quiz items were manually read against their
source records, owner, condition, cutoff, and active memory value.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp09-mp01` | PASS with note | Audio2Tool `tier7:2054` resolves an invalid brightness request to 90 and soft blue `#3399FF`. Only the explicitly restated durable color is normalized to VehicleMemBench `blue`; brightness is intentionally not stored. A human should approve the added romantic-mood scope. |
| `hvp09-mp02` | PASS with note | Audio2Tool `tier4:4258` implies closing an open rear trunk. The adaptation makes Cameron's no-automatic-open rule explicit for heavy shopping, rather than claiming that a closed trunk performs loading. This condition extension needs human approval. |
| `hvp09-mp03` | PASS with note | Audio2Tool `tier6:4138` explicitly corrects voice guidance from off to on. The persistence addition explicitly chooses `detailed` for Cameron during audiobooks; a human should approve this Boolean-to-categorical normalization. |

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp09-turn-05`–`09` | Coreference resolution | PASS. Current fan, departure access, cold-drive heat, bright-day roof, shopping trunk, and each owner's audiobook/lighting choices are active at their cutoffs. |
| `hvp09-turn-10`–`12` | Error correction | PASS. The answers consistently use fan 8 rather than 6, unlocked rather than locked at departure, and enabled rather than disabled wheel heat. |
| `hvp09-turn-13`–`18` | Preference conflict | PASS. Romantic lighting (blue vs. red), heavy-shopping trunk behavior (closed-until-requested vs. open), and audiobook guidance (detailed vs. mute) are same-scope owner distinctions tested in both directions. |
| `hvp09-turn-19`–`21` | State shift | PASS. The first two questions fall on opposite sides of the locked→unlocked door replacement; the last uses the later enabled wheel-heat state. |
| `hvp09-final-07`–`08` | Coreference / preference conflict | PASS. Cameron maps to blue/detailed/closed, whereas Jordan maps to red/mute/open; all six facts remain active in final memory. |

## Manual safeguards applied

- Did not store Audio2Tool brightness 90 because Cameron only makes color the
  durable preference.
- Worded Cameron's cargo prompts to require an explicit open request, avoiding
  an impossible interpretation of loading through a closed trunk.
- Paired default-state calls with another state-changing action where needed so
  simulator success cannot be obtained from a pure no-op target.

## Remaining human-validation requirement

- Audio2Tool query text is synthetic and CC BY-NC 4.0.
- An independent human must approve or rewrite the romantic-mood, cargo, and
  guidance-mode adaptations and verify all Quiz items while blind to model
  predictions before this is called human-authored or human-validated.
