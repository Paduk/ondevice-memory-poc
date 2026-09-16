# HVP04 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to method predictions and scores. It is
not a substitute for independent human validation.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp04-mp01` | PASS with note | Audio2Tool `tier7:2102` explicitly requests 28°C in a vehicle cabin. The persistence sentence scopes the value to Casey's driver zone; a human should confirm that this zone narrowing reads naturally. |
| `hvp04-mp02` | PASS with note | Audio2Tool `tier7:241` explicitly resolves a fan request to speed 8. The persistence sentence maps the room fan to Casey's front vehicle zone; this transition requires a human rewrite. |
| `hvp04-mp03` | PASS with note | Audio2Tool `tier7:340` directly requests locking an entrance before leaving. The persistence sentence maps it to Casey's all-door vehicle rule when parked at home; this cross-domain transition requires a human rewrite. |

All source turns except the final persistence sentence are retained verbatim
with record and turn provenance.

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp04-turn-05`–`10` | Coreference resolution | PASS. Indirect references resolve to Jordan's driver zone, front trunk, steering wheel, circulation, driver window, fan, and doors at the stated cutoffs. The default unlocked-door action is paired with a grounded temperature call. |
| `hvp04-turn-11`–`12` | Error correction | PASS. Fan level 5 supersedes 9, and the final window action is 90% after discarding the unrelated suspension and parking-assist requests. |
| `hvp04-turn-13`–`18` | Preference conflict | PASS. Driver temperature (28 vs. 22), front fan (8 vs. 5), and parked-at-home door state (locked vs. unlocked) are exact same-target owner conflicts tested in both directions. |
| `hvp04-turn-19`–`22` | State shift | PASS. The driver-window snapshots correctly follow 20→0→55→90. The zero-degree checkpoint is paired with the active fan setting so the simulator observes a state change. |
| `hvp04-final-07` | Coreference resolution | PASS. “They/their” consistently denotes Casey across the driving-to-home sequence, and all three calls use Casey's final active memory. |

## Remaining human-validation requirement

- Audio2Tool text is synthetic and CC BY-NC 4.0.
- A human must smooth the zone narrowing and two cross-domain adaptations,
  then independently verify every Quiz while blind to model outputs.
