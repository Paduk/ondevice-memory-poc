# HVP11 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to Summary, Patch, and Delta-v3 outputs.
All inserted turns and 19 added Quiz items were manually checked against their
source, owner, condition, cutoff, and active memory value.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp11-mp01` | PASS with normalization approval required | Audio2Tool `tier6:4136` is vehicle-specific and its correction changes the intended action to voice guidance on. Mapping guidance-on to VehicleMemBench `detailed` preserves the action, but the audiobook condition and persistence are additions that require independent approval. |
| `hvp11-mp02` | PASS with rewrite required | Audio2Tool `tier4:125` explicitly encodes red in the labeled HSB action. The adaptation preserves red but changes a room light into vehicle ambient lighting; a human must rewrite or approve that domain transfer. |
| `hvp11-mp03` | PASS with rewrite required | Audio2Tool `tier7:243` is a coherent five-turn repair ending at fan speed 5. The exact value is preserved, but the home fan is adapted into an all-zone cabin fan and must be independently rewritten. |

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp11-turn-05`–`09` | Coreference resolution | PASS. Each prompt names one owner and activates the selected seat, climate, lighting, voice, access, or destination conditions at its cutoff. Emery's early item occurs only after all three Emery updates. |
| `hvp11-turn-10`–`12` | Error correction | PASS after correction. The Gold actions use Quinn's final fan speed, departure access rule, and unnamed-destination default. The access item includes the active fan action because unlock is already the simulator's initial state; both calls remain memory-grounded and conversationally compatible. |
| `hvp11-turn-13`–`18` | Preference conflict | PASS. The six items are balanced owner-direction pairs over three genuine same-Tool differences: detailed versus muted guidance, red versus blue ambient light, and fan speed 5 versus Quinn's final speed 8. No prompt states the distinguishing value. |
| `hvp11-turn-19`–`21` | State shift | PASS. Two items straddle Quinn's fan replacement (6→8); the third queries the locked parked-access state immediately before the later departure replacement. |
| `hvp11-final-07`–`08` | Coreference / preference conflict | PASS. Both final items activate normal lighting and fan plus audiobook guidance. The conflict item selects Quinn's final values across all three genuine conflict axes. |

## Manual corrections and restraint

- Reopened the earlier source audit without relaxing value grounding: the two
  cross-device records retain their explicit red and speed-5 actions and are
  clearly marked as adaptations.
- Rejected rear-seat and rear-zone records as conflicts with Quinn's driver-seat
  and passenger-zone memories because their targets differ.
- Rejected lock-state queries and route-cancel records because they do not
  establish alternative durable settings.
- Reworded the two-person comfort prompt to identify the remembered setting
  types without revealing their values.
- Expanded the departure correction item with a compatible, memory-grounded fan
  action after the official simulator correctly rejected a no-state-change-only
  unlock target.
- Made the pre-revision access state-shift prompt explicitly say that Quinn has
  parked, rather than relying on the memory-slot label to imply its condition.

## Remaining human-validation requirement

- Audio2Tool query text is synthetic and CC BY-NC 4.0.
- `hvp11-mp02` and `hvp11-mp03` require independent in-vehicle rewrites;
  `hvp11-mp01` requires explicit normalization approval.
- Every adaptation and Quiz must still receive blind human validation before
  this scenario is described as human-authored or human-validated.
