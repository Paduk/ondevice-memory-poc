# HVP10 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to Summary, Patch, and Delta-v3 outputs.
All inserted turns and 19 added Quiz items were manually checked against their
source, owner, condition, cutoff, and active memory value.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp10-mp01` | PASS with rewrite required | Audio2Tool `tier4:4194` contains the human-like implicit complaint “too cold,” while its released label resolves the action to 28°C heat. The adaptation makes 28°C and the passenger zone explicit before declaring persistence. Because those details are label-derived rather than verbatim, an independent human must rewrite or explicitly approve this turn. |
| `hvp10-mp02` | PASS | Audio2Tool `tier7:2118` is a multi-turn repair that ends with both front and rear defrost. The added sentence preserves that scope and only makes it durable for rear-visibility needs. |
| `hvp10-mp03` | PASS | Audio2Tool `tier7:1990` gives a detailed, vehicle-specific reason for choosing steering-wheel heat instead of cabin or seat heat. The persistence sentence directly matches the stated cold-hands intent. |

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp10-turn-05`–`09` | Coreference resolution | PASS after correction. The fan, two window scopes, sunroof, defrost, passenger temperature, and wheel heat are active at cutoff. Two prompts were rewritten so a passenger was not incorrectly asked to use steering-wheel heat. |
| `hvp10-turn-10`–`12` | Error correction | PASS. The target values are fan 7 rather than 1, sunroof 30% rather than 10%, and passenger window 10% rather than 60%. |
| `hvp10-turn-13`–`18` | Preference conflict | PASS. Only two genuine differing axes are used: passenger temperature (28 vs. 23) and rear-visibility defrost scope (both vs. rear only). The last pair combines those two distinctions; no false conflict is claimed for wheel heat. |
| `hvp10-turn-19`–`21` | State shift | PASS. Two items straddle Taylor's 10%→30% sunroof replacement and the third uses the later 10% passenger-window value. |
| `hvp10-final-07`–`08` | Coreference / preference conflict | PASS. Dakota's final item includes the shared wheel-heat preference as a memory-binding check; Taylor's final conflict uses only the two values that genuinely differ. |

## Manual corrections and restraint

- Rejected the tempting but false construction of making Dakota's steering-
  wheel preference differ from Taylor's: the available unused source supports
  `enabled=true`, the same value Taylor uses.
- Corrected two prompts that initially placed Dakota in the passenger seat while
  asking for steering-wheel heat; Dakota now drives while setting the passenger
  zone for the companion.
- Reused two genuine owner-conflict axes in combined items instead of inventing
  a third unsupported difference merely to increase apparent variety.

## Remaining human-validation requirement

- Audio2Tool query text is synthetic and CC BY-NC 4.0.
- `hvp10-mp01` specifically requires independent human rewriting or approval;
  all other adaptations and Quiz items also require blind human validation
  before this is called human-authored or human-validated.
