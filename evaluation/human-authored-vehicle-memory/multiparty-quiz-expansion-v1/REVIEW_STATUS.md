# Multi-participant Quiz expansion review status

Snapshot date: 2026-09-12 UTC

## Scope completed

- Completed through author-level semantic review: `HVP01`–`HVP20`
- No scenario is left partially authored or only automatically checked in this
  release.

The completed twenty scenarios contain:

- 1,174 dialogue turns: 251 UPDATE and 923 NO_OP
- 430 TURN Quiz and 144 FINAL Quiz (574 total)
- 374 newly added Quiz items
- 374/374 newly added Quiz queries are exact-text unique across `HVP01`–`HVP20`
- 807 Gold Tool calls
- 60 newly inserted Audio2Tool records, all unique across the extensions and
  disjoint from the frozen 20-scenario source corpus
- 174 inserted source-trace turns: 78 retain source text exactly, 43 append only
  a persistence sentence, and 53 in `HVP14`–`HVP20` carry an explicit
  vehicle-domain rewrite mode while retaining the exact original text in
  provenance
- all 1,748 derived Patch/Quiz rows are isolated as `pilot_excluded`; all 1,174
  Patch rows also have `train_eligible: false`

Added Quiz distribution:

| Quiz type | Reasoning type | Count |
| --- | --- | ---: |
| TURN | coreference resolution | 106 |
| TURN | error correction | 47 |
| TURN | preference conflict | 127 |
| TURN | state shift | 70 |
| FINAL | coreference resolution | 11 |
| FINAL | preference conflict | 13 |

## Validation result

Every completed scenario passed all of the following after a clean rebuild:

- source contract and original-source hash preservation
- chronology and unique identifiers
- all-turn Patch replay and NO_OP memory identity
- active evidence at each Quiz cutoff
- official Tool JSON Schema validation
- official VehicleWorld simulator execution
- FINAL Quiz use of the final snapshot
- per-scenario addition quota
- multiple memory owners
- external record disjointness
- exact external record/turn provenance and reuse-mode-specific text checks
- zero-shot Test isolation from train-eligible data
- same-Tool alternative evidence for every preference-conflict item
- a genuinely different stored value or target scope for the other owner in all
  140 preference-conflict items (same-Tool/same-value pseudo-conflicts fail)
- consistent named owner and single-owner Gold evidence in all 257 newly added
  coreference-resolution and preference-conflict items

Each scenario also has an `AUTHOR_REVIEW.md` recording a manual source and Quiz
review. The review caught issues not detectable by the deterministic checks,
including an invalid filtration-to-recirculation equivalence, a pure no-op
simulator target, physically awkward trunk wording, and a passenger being asked
to use steering-wheel heat. A second pass over `HVP11`–`HVP15` also removed one
playback-direction leak from `hvp12-turn-05`, removed the literal Gold climate
mode from `hvp14-turn-08`, made six selected-memory triggers explicit, and
reframed HVP14 cargo actions as preparation rather than mid-load closure. For
`HVP14`, review rejected two otherwise suitable
but already-used vehicle records and replaced awkward cross-device sentence
splices with explicitly traced, coherent vehicle-domain rewrites. `HVP15`
likewise preserves explicit source values and dialogue acts across its fan and
Pause rewrites. `HVP16` deliberately retains only two genuine user-conflict
axes instead of inventing a third one from its separate temperature fact.
`HVP17` contributes three fully matched conflict axes; `HVP18` similarly avoids
misrepresenting its owner-specific music-volume fact as a conflict and now
states all three triggers in the combined prompt. `HVP19` keeps its music volume
as a companion fact with parked/music context explicit, while `HVP20` closes the
corpus with three fully matched axes. Direct answer values were also removed from
preference-conflict prompts. All corrected items were rebuilt before inclusion.

## Authorship boundary

This is **model-assisted author review**, not independent human validation.
Audio2Tool query text is synthetic and CC BY-NC 4.0. The source scenarios retain
`human_review_status: PENDING` intentionally. Before publication, an independent
human reviewer should:

1. rewrite or approve every persistence addition without seeing model outputs;
2. independently verify owner, condition, cutoff, and Gold Tool call for every
   added Quiz;
3. resolve every `PASS with note` or `rewrite required` item in the per-scenario
   review files; and
4. sign and version a frozen human-validated release.

The consolidated `HUMAN_REVIEW_QUEUE.md` contains all 60 inserted sessions:
36 require rewriting, 16 require explicit normalization approval, and 8 are
lower-risk approval checks. None is pre-signed.

`HVP11_HVP15_SOURCE_VALUE_AUDIT.md` and the scenario-specific HVP16–HVP20
source-value audits separately record the direct comparison between every
resumed source record's external Tool target and its V2 memory target; all 30
values are grounded, while object/scope/condition rewrites remain queued for
independent review.

`HVP16_HVP20_SECOND_PASS_AUDIT.md` additionally records the 12 matched
owner-conflict axes, the two deliberately non-conflicting companion-fact
patterns, four author-review interventions, query leakage checks, and a
byte-identical rebuild of every newly completed scenario.

`HVP11_HVP15_SECOND_PASS_AUDIT.md` records the independent re-audit of 15
matched conflict axes, two corrected answer-leakage prompts, the high-risk
HVP14/HVP15 adaptation boundaries, and the clean-rebuild checks.

## Reproduction commands

From the PalmClaw repository root, rebuild and validate one completed scenario:

```bash
PYTHONPATH=/home/hj153lee/PalmClaw:/home/hj153lee/PalmClaw/ubuntu/src \
  /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python \
  memory_training/scripts/build_multiparty_hvp_quiz_expansion.py \
  --scenario HVP20

PYTHONPATH=/home/hj153lee/PalmClaw:/home/hj153lee/PalmClaw/ubuntu/src \
  /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python \
  memory_training/scripts/validate_multiparty_hvp_quiz_expansion.py \
  --scenario HVP20
```

Then validate the completed corpus and regenerate `CORPUS_VALIDATION.json`:

```bash
/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python \
  memory_training/scripts/validate_multiparty_hvp_corpus.py \
  --expected-scenarios 20
```
