# HVP16–HVP20 second-pass audit

Status: **PASS for model-assisted author review; independent human validation
remains pending.**

This audit was performed without inspecting Summary, Patch, or Delta-v3 model
predictions. It covers the five scenarios added in the HVP16–HVP20 pass.

## Scope

- 5 completed scenarios
- 15 disjoint external records and 15 inserted UPDATE turns
- 47 inserted source-trace turns
- 92 added Quiz items: 25 coreference, 10 error-correction, 35 TURN
  preference-conflict, 17 state-shift, and 5 FINAL preference-conflict
- 65 owner-sensitive Quiz items and 40 genuine preference-conflict items
- 15/15 external target values directly grounded in the source record's final
  user turn and/or structured Tool target

## Matched owner-conflict axes

Only axes with the same Tool, selector, and normalized condition but different
values were treated as competing preferences.

| Scenario | Matched axis | Primary user | Added user | Review |
| --- | --- | --- | --- | --- |
| HVP16 | normal sunroof | Reese: 0 | Avery: 100 | PASS |
| HVP16 | stopped playback | Reese: resume | Avery: pause | PASS |
| HVP17 | normal ambient color | Blair: white | Jordan: blue | PASS |
| HVP17 | normal all-zone temperature | Blair: 27°C | Jordan: 19°C | PASS |
| HVP17 | children report stuffy rear cabin, rear-right window | Blair: 60 | Jordan: 100 | PASS; safety wording queued |
| HVP18 | normal ambient color | Harper: red | Riley: blue | PASS |
| HVP18 | normal all-zone temperature | Harper: 28°C | Riley: 20°C | PASS |
| HVP19 | normal ambient color | Emerson: white | Taylor: blue | PASS |
| HVP19 | parked all-door access | Emerson: unlocked | Taylor: locked | PASS |
| HVP20 | normal ambient color | Finley: white | Quinn: yellow | PASS |
| HVP20 | normal all-zone temperature | Finley: 21°C | Quinn: 25°C | PASS |
| HVP20 | parked all-door access | Finley: unlocked | Quinn: locked | PASS |

The HVP16 temperature, HVP18 music volume, and HVP19 music volume are
owner-specific companion facts. They were deliberately not claimed as conflict
axes because the primary user has no matched alternative.

## Adversarial review findings

1. **HVP16:** no sufficiently grounded third conflict axis was available, so the
   design retained two real axes rather than manufacturing a pseudo-conflict.
2. **HVP17:** the 100% rear-right-window adaptation is deterministically valid
   but its child-passenger safety framing was explicitly escalated to independent
   review.
3. **HVP18:** one Quiz originally allowed Riley's music volume to sound like a
   Harper conflict. It was rewritten to label ambient and temperature as the
   competing facts and volume as a companion fact. A final condition audit also
   added normal-drive and usual-music triggers to that prompt. Harper's
   seat-cooling companion was removed from a same-cutoff TURN duplicate and
   retained only in the richer FINAL item.
4. **HVP19:** one state-shift Quiz initially used cutoff `s7` while its seat fact
   begins at `s8`. The cutoff was corrected to `s8`, rebuilt, and revalidated.
   A later audit added normal-drive, music, and parked triggers to Taylor's
   three-memory prompt.
5. **HVP20:** all three owner-conflict axes were checked in both directions and
   at the final snapshot. A redundant parked-door companion was removed from
   the same-cutoff coreference item, leaving the full three-axis combination in
   FINAL only.

## Reproducibility and leakage checks

- A clean rebuild of HVP16–HVP20 produced byte-identical SHA-256 hashes for all
  five `scenario-source.json` files, attribution files, manifests, snapshots,
  and dialogue/Patch/Quiz JSONL files.
- Every scenario passed the official builder and the dedicated deterministic
  validator after that rebuild.
- Every source Quiz exactly matches its derived TURN/FINAL JSONL row on
  `quiz_id`, query text, and Gold calls.
- All 92 added Quiz queries are exact-text unique in this five-scenario pass and
  participate in the corpus-wide 374/374 uniqueness result.
- A targeted scan plus direct reading found no query that states its
  distinguishing numeric, color, lock, circulation, or playback answer.

## Human-validation boundary

This document records model-assisted author review, not real human validation.
Audio2Tool query text is synthetic. The 15 inserted sessions and 92 Quiz items
remain in `HUMAN_REVIEW_QUEUE.md` for blind independent rewrite, approval, and
Gold-call verification before publication as human-validated data.
