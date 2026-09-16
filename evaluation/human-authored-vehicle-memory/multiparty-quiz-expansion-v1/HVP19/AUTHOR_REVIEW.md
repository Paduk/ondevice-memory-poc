# HVP19 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to Summary, Patch, and Delta-v3 outputs.
All inserted turns and 18 added Quiz items were checked against their source,
owner, condition, cutoff, active memory value, and Gold Tool call.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp19-mp01` | PASS with rewrite required | Audio2Tool `tier7:67` explicitly rejects red and chooses blue at a restrained brightness. Its three-turn clarification is rewritten as vehicle ambient lighting; HSB hue 240 is normalized to `blue`, and brightness is excluded from memory. |
| `hvp19-mp02` | PASS with high-risk rewrite required | Audio2Tool `tier7:345` explicitly locks the front door when leaving. The security intent and confirmation are preserved in a parked-vehicle all-door rewrite; object, scope, and persistence require independent approval. |
| `hvp19-mp03` | PASS with rewrite required | Audio2Tool `tier7:547` explicitly chooses speaker volume 50. The same three-turn exchange is rewritten as in-car music and made durable. |

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp19-turn-05`–`09` | Coreference resolution | PASS after one cutoff correction. Each question selects one owner and activates normal lighting/air/music, parked access, background noise, front condensation, coat retrieval, or rear-right seat comfort as required. `hvp19-turn-05` uses `s8`, not `s7`, so both Gold facts exist. |
| `hvp19-turn-10`–`11` | Error correction | PASS. The questions select Emerson's returned-to-white ambient value and final outside-air replacement without exposing either answer. White is paired with active seat ventilation to avoid a simulator-default-only target. |
| `hvp19-turn-12`–`18` | Preference conflict | PASS. Every item contains a genuine difference on normal ambient blue versus white, parked all-door locked versus unlocked, or both. Taylor's volume and Emerson's circulation/radio/seat settings remain owner-specific companion facts. |
| `hvp19-turn-19`–`21` | State shift | PASS. The cutoffs select Emerson's ambient white→red→white sequence. Both white targets are paired with active rear-right seat ventilation. |
| `hvp19-final-07` | Preference conflict | PASS. The final item selects Emerson over Taylor on both matched axes and adds only Emerson-specific circulation and seat facts. |

## Manual checks and corrections

- Selected three source records unused by the frozen HVP01–HVP20 corpus and all
  previously completed extensions.
- Retained explicit targets only: blue from HSB hue 240, Locked, and volume 50.
- Matched the parked condition and all-door selector exactly for the competing
  lock states.
- Corrected `hvp19-turn-19` from cutoff `s7` to `s8` because rear-right seat
  ventilation is not active until `s8`; then rebuilt and revalidated.
- Did not present Taylor's music volume as having an Emerson alternative; a
  later condition audit also added normal-drive, music, and parked context to
  `hvp19-turn-18` so all three selected memories are active.
- No added question states a distinguishing color, lock state, or volume.

## Remaining human-validation requirement

- Audio2Tool query text is synthetic and CC BY-NC 4.0.
- All three vehicle adaptations require blind independent rewrite or approval;
  the home-entry to all-vehicle-door transformation needs the closest review.
- Every Quiz must receive blind human validation before this scenario is
  described as human-authored or human-validated.
