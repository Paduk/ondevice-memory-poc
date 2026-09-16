# HVP15 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to Summary, Patch, and Delta-v3 outputs.
All inserted turns and 19 added Quiz items were manually checked against their
source, owner, condition, cutoff, and active memory value.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp15-mp01` | PASS with rewrite required | Audio2Tool `tier7:248` contains a five-turn distraction-and-return exchange ending in explicit fan speed 3. The dialogue acts and value are preserved in a disclosed home-to-vehicle rewrite; the wording requires independent approval. |
| `hvp15-mp02` | PASS with rewrite required | Audio2Tool `tier7:225` explicitly selects 23°C. The original exchange is retained, but assigning the value to the vehicle driver zone and making it durable require an independent in-car rewrite. |
| `hvp15-mp03` | PASS with rewrite required | Audio2Tool `tier7:526` explicitly selects Pause. Its three-turn request/confirmation/action structure is preserved in a disclosed stopped-audiobook rewrite so that the condition matches Rowan's competing resume preference. |

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp15-turn-05`–`09` | Coreference resolution | PASS. Each question names one owner and explicitly activates normal driving, normal fan, stopped audiobook, missing gloves, obvious directions, or normal rear-trunk conditions as required. Morgan's item occurs only after all three inserted updates. |
| `hvp15-turn-10`–`11` | Error correction | PASS. The questions select Rowan's later fan and temperature records without exposing either value. |
| `hvp15-turn-12`–`18` | Preference conflict | PASS. Three balanced owner-direction pairs compare the same axes: fan 3 versus 8, driver temperature 23 versus 18, and leaving a stopped audiobook paused versus resuming it. The combined Morgan item covers all three axes. |
| `hvp15-turn-19`–`22` | State shift | PASS. The first pair straddles Rowan's outside→inside circulation change, and the second pair straddles the fan 7→8 change. The inside-circulation item is paired with the active fan value so the official simulator observes a real state change. |
| `hvp15-final-07` | Preference conflict | PASS. The final question activates Rowan's current fan, temperature, and stopped-audiobook rule and selects all three against Morgan's alternatives. |

## Manual corrections and restraint

- Selected only previously unused records whose final values—3, 23°C, and
  Pause—are explicit in the source dialogue; no value was inferred from tone.
- Rewrote the long fan exchange turn by turn instead of appending an unrelated
  vehicle preference to home-device wording. Exact source text remains in every
  provenance trace under an explicit rewrite mode.
- Made both owners' playback condition identical before treating Pause versus
  Resume as a preference conflict.
- Paired Morgan's paused action and Rowan's revised inside-circulation action
  with another active memory action where a Boolean/enum action alone would
  match the simulator's initial state.
- No added question states a distinguishing fan speed, temperature, or playback
  state.

## Remaining human-validation requirement

- Audio2Tool query text is synthetic and CC BY-NC 4.0.
- All three inserted sessions require blind independent in-vehicle rewrite or
  approval, with special attention to the condition added for persistence.
- Every adaptation and Quiz must still receive blind human validation before
  this scenario is described as human-authored or human-validated.
