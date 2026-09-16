# HVP14 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to Summary, Patch, and Delta-v3 outputs.
All inserted turns and 19 added Quiz items were manually checked against their
source, owner, condition, cutoff, and active memory value.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp14-mp01` | PASS with high-risk rewrite required | Audio2Tool `tier7:458` supplies a three-turn Open–confirmation exchange. Its dialogue acts and binary Open target are preserved, while the garage is explicitly rewritten as the vehicle sunroof and full opening is normalized to 100%. The full exchange must receive an independent rewrite/approval. |
| `hvp14-mp02` | PASS with normalization approval required | Audio2Tool `tier6:4119` explicitly corrects “strip mall” to `Suburban Mall`. The location is preserved exactly; promoting that corrected stop to Sawyer's unnamed-navigation default requires independent approval. |
| `hvp14-mp03` | PASS with high-risk rewrite required | Audio2Tool `tier7:454` supplies a three-turn Close–clarification exchange. Its dialogue acts and Closed target are preserved, while the garage is explicitly rewritten as the front trunk. The added cargo-loading persistence condition must be independently rewritten/approved. |

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp14-turn-05`–`09` | Coreference resolution | PASS. Every prompt activates the stated owner's stored conditions at its cutoff. Sawyer's three-memory item appears only after all three inserted updates; Parker's final item uses the revised roof, wheel-heat, and destination values. |
| `hvp14-turn-10`–`11` | Error correction | PASS. The questions select Parker's later roof and destination values without stating either answer. |
| `hvp14-turn-12`–`18` | Preference conflict | PASS. The owner-direction pairs compare the same behavior axes: sunroof 100 versus 18, unnamed destination `Suburban Mall` versus `Home`, and cargo-loading front-trunk closed versus open. Each pair differs in a Tool argument rather than merely in wording or condition. |
| `hvp14-turn-19`–`22` | State shift | PASS. The cutoffs select roof 30 before replacement, roof 18 after replacement, wheel heat off before replacement, and wheel heat on after replacement. |
| `hvp14-final-07` | Preference conflict | PASS. The final prompt selects Parker's three current values against Sawyer's alternatives and does not expose the distinguishing values. |

## Manual corrections and restraint

- Rejected direct vehicle records `tier5:4276` and `tier4:4220` after the
  source-disjointness gate showed that they already belong to the frozen HVP
  corpus; no external record was duplicated to improve this scenario.
- Replaced the earlier awkward garage sentence followed by an unrelated vehicle
  preference with coherent, turn-by-turn vehicle-domain rewrites. Each trace now
  retains the exact source text and marks the rewrite explicitly as
  `vehicle_domain_rewrite` or `vehicle_domain_rewrite_with_persistence`.
- Preserved the source dialogue acts (request, confirmation/clarification, final
  choice) and Open/Closed polarity; only the controlled object and persistence
  condition were changed.
- Confirmed that all added preference-conflict questions use facts visible at
  their cutoffs and that no query states the distinguishing numeric, Boolean, or
  destination answer.
- Removed the literal Gold mode name from `hvp14-turn-08` in a second-pass
  audit; the prompt now supplies only the rear-visibility trigger and asks for
  the remembered climate response.
- Reframed cargo prompts as preparation rather than completed loading. Sawyer's
  prompts also state that no explicit open request was made, so keeping the
  compartment closed remains physically coherent while Parker's automatic-open
  preference remains a genuine competing behavior.
- Added the cabin-venting trigger to `hvp14-turn-21`, whose no-op wheel state is
  paired with a roof action; both selected memories are now explicitly active.

## Remaining human-validation requirement

- Audio2Tool query text is synthetic and CC BY-NC 4.0.
- `hvp14-mp01` and `hvp14-mp03` are intentionally marked high-risk
  vehicle-domain rewrites; `hvp14-mp02` requires normalization approval.
- Every adaptation and Quiz must still receive blind independent human
  validation before this scenario is described as human-authored or
  human-validated.
