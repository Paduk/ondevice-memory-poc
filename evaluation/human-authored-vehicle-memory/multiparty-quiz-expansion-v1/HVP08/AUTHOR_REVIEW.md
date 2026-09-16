# HVP08 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to Summary, Patch, and Delta-v3 outputs.
It included source-by-source reading and a manual audit of all 19 added Quiz
items against their exact cutoff memories.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp08-mp01` | PASS with note | Audio2Tool `tier7:2103` explicitly resolves climate to 22°C and AUTO. The durable addition retains only the explicitly restated 22°C driver-zone preference; AUTO remains an immediate source action and is not silently added to memory. |
| `hvp08-mp02` | PASS with note | Audio2Tool `tier7:2055` explicitly selects bright red ambient lighting. The persistence sentence stores only red, not brightness, so the memory label does not overclaim an unstated durable brightness preference. |
| `hvp08-mp03` | PASS with note | Audio2Tool `tier5:4270` explicitly closes the rear trunk to conceal bags. The adaptation was revised to a plausible persistent rule: Blake does not want the trunk opened automatically when loading bulky items and requires an explicit open request. A human should approve this condition extension. |

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp08-turn-05`–`09` | Coreference resolution | PASS. The current window, destination, departure doors, and each owner's temperature/light facts are active and unambiguous at their cutoffs. |
| `hvp08-turn-10`–`12` | Error correction | PASS. The targets correctly use window 10% rather than 60%, the library rather than the dog park, and unlocked rather than locked doors. The default-state unlock is paired with navigation so the simulator measures a state change. |
| `hvp08-turn-13`–`18` | Preference conflict | PASS. Driver temperature (22 vs. 18), ambient color (red vs. blue), and rear-trunk behavior (explicit-open-only/closed vs. automatically open) are same-target owner distinctions tested both ways. |
| `hvp08-turn-19`–`21` | State shift | PASS. The first two questions straddle the 60%→10% window replacement; the third uses the later library destination. |
| `hvp08-final-07`–`08` | Coreference / preference conflict | PASS. The two final questions deliberately mirror the owners: Blake maps to 22/red/closed and Avery to 18/blue/open. |

## Manual corrections made

- Reworded Blake's trunk persistence sentence and related questions to avoid the
  physically odd implication that a closed trunk itself loads an item.
- Kept source brightness and AUTO mode out of durable memory because only color
  and temperature were explicitly restated as persistent.
- Confirmed every owner-conflict pair uses different values for the same
  VehicleMemBench Tool and compatible argument scope.

## Remaining human-validation requirement

- Audio2Tool query text is synthetic and CC BY-NC 4.0.
- An independent human must approve or rewrite the three persistence additions,
  especially the trunk condition, and verify all Quiz items while blind to
  model predictions before this is called human-authored or human-validated.
