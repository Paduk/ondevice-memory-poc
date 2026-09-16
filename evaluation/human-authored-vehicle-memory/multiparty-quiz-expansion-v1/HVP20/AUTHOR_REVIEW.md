# HVP20 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to Summary, Patch, and Delta-v3 outputs.
All inserted turns and 18 added Quiz items were checked against their source,
owner, condition, cutoff, active memory value, and Gold Tool call.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp20-mp01` | PASS with rewrite required | Audio2Tool `tier7:73` rejects bright blue/orange and explicitly settles on soft yellow, labeled HSB `60,70,80`. Its clarification structure is rewritten as vehicle ambient lighting; hue is durable and brightness is excluded. |
| `hvp20-mp02` | PASS with rewrite required | Audio2Tool `tier7:229` explicitly chooses 25°C after a strong cold complaint. The exchange is rewritten as all-zone vehicle climate and made durable. |
| `hvp20-mp03` | PASS with high-risk rewrite required | Audio2Tool `tier7:337` explicitly locks the front door when leaving for security. The intent and confirmation are preserved in a parked-vehicle all-door rewrite; object and scope require independent approval. |

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp20-turn-05`–`09` | Coreference resolution | PASS. Each question names one owner and activates normal lighting/climate, parked access, unnamed navigation, frequent prompts, rear visibility, or stale-air headache as required. Quinn's combined item occurs after all three inserted memories. |
| `hvp20-turn-10`–`11` | Error correction | PASS. The questions select Finley's intermediate and final temperature revisions at the correct cutoffs without revealing their values. |
| `hvp20-turn-12`–`18` | Preference conflict | PASS. Three matched axes are covered in both directions: ambient yellow versus white, all-zone temperature 25 versus 21°C, and parked all-door locked versus unlocked. |
| `hvp20-turn-19`–`21` | State shift | PASS. The three cutoffs select Finley's temperature sequence 18→27→24 before the final 21°C value used by the correction and final items. |
| `hvp20-final-07` | Preference conflict | PASS. The final item selects Finley against Quinn on all three same-condition axes. |

## Manual checks and restraint

- Selected three source records unused by the frozen HVP01–HVP20 corpus and all
  previously completed extensions.
- Preserved exact source text in every provenance trace while marking each
  vehicle-domain rewrite explicitly.
- Retained explicit targets only: yellow from HSB hue 60, 25°C, and Locked.
- Matched Tool, selector, and condition exactly for all three owner-conflict
  axes before assigning the preference-conflict label.
- Checked every temperature cutoff across all four active values.
- Removed the parked-door companion call from `hvp20-turn-09`; the FINAL item
  retains the complete three-axis comparison, eliminating a same-cutoff Gold
  duplicate while preserving the Quiz quota.
- No added question states a distinguishing color, temperature, or lock state.

## Remaining human-validation requirement

- Audio2Tool query text is synthetic and CC BY-NC 4.0.
- All three adaptations require blind independent rewrite or approval; the
  home-entry to all-vehicle-door transformation needs the closest review.
- Every Quiz must receive blind human validation before this scenario is
  described as human-authored or human-validated.
