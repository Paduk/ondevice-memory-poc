# HVP18 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to Summary, Patch, and Delta-v3 outputs.
All inserted turns and 18 added Quiz items were checked against their source,
owner, condition, cutoff, active memory value, and Gold Tool call.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp18-mp01` | PASS with rewrite required | Audio2Tool `tier7:70` explicitly requests blue, labeled HSB `240,100,100`. Its three-turn confirmation is rewritten as vehicle ambient lighting; only color is durable and brightness remains turn-local. |
| `hvp18-mp02` | PASS with rewrite required | Audio2Tool `tier7:227` explicitly chooses 20°C and excludes a different room. The exchange is rewritten as an all-zone vehicle request and made durable. |
| `hvp18-mp03` | PASS with normalization required | Audio2Tool `tier7:560` is already a driving exchange: music masks traffic, the user briefly checks the exit, then explicitly returns to volume 20. All five turns are retained verbatim and only in-car persistence is appended. |

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp18-turn-05`–`09` | Coreference resolution | PASS. Each item selects one owner and activates normal lighting/climate/music, side-window fog, a stuck front trunk, distracting noise, or rear visibility as required. Riley's combined item occurs after all three inserted updates. |
| `hvp18-turn-10`–`11` | Error correction | PASS. The questions select Harper's active ambient and final temperature revisions without revealing either target. |
| `hvp18-turn-12`–`18` | Preference conflict | PASS. Every item has a genuine alternative on ambient blue versus red, temperature 20 versus 28°C, or both. Riley's volume and Harper's seat ventilation are treated only as owner-specific companion facts, not as competing values. One initially ambiguous query was rewritten to state that boundary explicitly. |
| `hvp18-turn-19`–`21` | State shift | PASS. The cutoffs select Harper's original pink ambient color, original 24°C temperature, and intermediate 21°C temperature before the final replacement. |
| `hvp18-final-07` | Preference conflict | PASS. The final item selects Harper over Riley on both genuine axes and adds Harper's owner-specific rear-right seat cooling. |

## Manual checks and restraint

- Selected three source records unused by the frozen HVP01–HVP20 corpus and all
  previously completed extensions.
- Preserved exact original text in every provenance trace, including all five
  turns of the already vehicle-situated volume record.
- Retained only explicit targets: blue from HSB hue 240, 20°C, and volume 20.
- Deliberately did not convert an unrelated fan-speed source into seat
  ventilation merely to manufacture a third user-conflict axis.
- Reworded `hvp18-turn-18` after semantic review so music volume is not presented
  as having a Harper alternative, and a later condition audit added the normal
  drive and usual-music triggers for all three selected memories.
- Removed Harper's seat-cooling companion call from `hvp18-turn-17`; the FINAL
  item retains it, so the TURN and FINAL items no longer duplicate the same
  owner, cutoff, and complete Gold call set.
- No added question states a distinguishing color, temperature, or volume.

## Remaining human-validation requirement

- Audio2Tool query text is synthetic and CC BY-NC 4.0.
- All adaptations and persistence statements require blind independent approval.
- Every Quiz must receive blind human validation before this scenario is
  described as human-authored or human-validated.
