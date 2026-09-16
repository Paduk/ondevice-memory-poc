# HVP13 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to Summary, Patch, and Delta-v3 outputs.
All inserted turns and 19 added Quiz items were manually checked against their
source, owner, condition, cutoff, and active memory value.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp13-mp01` | PASS with rewrite required | Audio2Tool `tier7:270` is a three-turn clarification that explicitly selects an always-on fan at speed 8. The speed is preserved, but the living-room fan is adapted to the all-zone cabin fan and needs an independent in-vehicle rewrite. |
| `hvp13-mp02` | PASS with rewrite required | Audio2Tool `tier7:220` explicitly raises a thermostat from 18.5°C to 24°C. The value is preserved, but the home thermostat is adapted to the vehicle's rear-right zone and must be rewritten independently. |
| `hvp13-mp03` | PASS with normalization approval required | Audio2Tool `tier7:529` explicitly corrects Stop to Pause. Pause is mapped to VehicleMemBench `music_switch=false`, and the durable rule to leave interrupted audio paused requires independent approval. |

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp13-turn-05`–`09` | Coreference resolution | PASS. Cameron's questions activate normal driving, rear-right climate, frost, storytelling, stopped audio, and cold hands as needed. Parker's item activates all three new memories after their insertion. |
| `hvp13-turn-10`–`12` | Error correction | PASS. The two fan questions distinguish the first and second corrections, and the window question selects the final position. No corrected value appears in a prompt. |
| `hvp13-turn-13`–`18` | Preference conflict | PASS. Three balanced owner-direction pairs cover fan speed 8 versus 3, rear-right temperature 24 versus 27, and leaving stopped audio paused versus restarting it. The latter is a true same-condition behavioral difference, not merely two unrelated media contexts. |
| `hvp13-turn-19`–`22` | State shift | PASS. Three fan snapshots respectively select 5 before correction, 1 after the first correction, and 3 after the second; the last item selects the revised driver-window state. |
| `hvp13-final-07` | Preference conflict | PASS. The final prompt activates Parker's normal fan, normal rear-right temperature, and stopped-audio rule and selects all three against Cameron's alternatives. |

## Manual corrections and restraint

- Rejected the initially considered Audio2Tool `tier6:3963` rear-right value of
  22.5°C because the official VehicleMemBench temperature Tool accepts integers;
  rounding it would have changed the source value.
- Replaced it with `tier7:220`, whose explicit integer 24°C target can be
  preserved exactly; only the controlled zone is adapted and flagged for rewrite.
- Rejected `tier7:254` after the source-disjointness gate showed it was already
  reserved by an earlier pilot; the final fan source `tier7:270` is disjoint.
- Used the stopped-audio contrast only after making both owners' conditions
  identical: Cameron restarts interrupted audio, while Parker leaves it paused.
- No question states the distinguishing fan speed, temperature, or playback state.

## Remaining human-validation requirement

- Audio2Tool query text is synthetic and CC BY-NC 4.0.
- `hvp13-mp01` and `hvp13-mp02` require independent in-vehicle rewrites;
  `hvp13-mp03` requires explicit normalization approval.
- Every adaptation and Quiz must still receive blind human validation before
  this scenario is described as human-authored or human-validated.
