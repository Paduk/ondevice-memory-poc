# HVP11–HVP15 second-pass audit

Status: **PASS for model-assisted author review; independent human validation
remains pending.**

This audit was performed without inspecting Summary, Patch, or Delta-v3 model
predictions. It independently reopened the five previously completed scenarios
and checked the external source, adaptation, owner/condition semantics, Quiz
cutoff, Gold action, and answer leakage.

## Scope

- 5 scenarios, 15 disjoint external records, and 15 inserted UPDATE turns
- 45 inserted source-trace turns: 20 verbatim/role-normalized, 11 minimal
  persistence adaptations, and 14 explicit vehicle-domain rewrite turns
- 95 added Quiz items: 26 coreference, 13 error-correction, 37 preference
  conflict, and 19 state-shift items
- 15/15 external target actions or values grounded in the source record's final
  user turn and structured Tool target; schema normalization and changed
  persistence conditions remain explicitly disclosed

## Matched owner-conflict axes

Only memories with the same Tool, target selector, and normalized condition but
different actions were accepted as competing preferences.

| Scenario | Matched axis | Primary user | Added user | Review |
| --- | --- | --- | --- | --- |
| HVP11 | audiobook guidance | Quinn: mute | Emery: detailed | PASS |
| HVP11 | normal ambient color | Quinn: blue | Emery: red | PASS |
| HVP11 | normal all-zone fan | Quinn: 8 | Emery: 5 | PASS |
| HVP12 | normal driver temperature | Sydney: 19°C | Rowan: 23°C | PASS |
| HVP12 | audiobook-listening playback | Sydney: play | Rowan: pause | PASS |
| HVP12 | parked all-door access | Sydney: locked | Rowan: unlocked | PASS |
| HVP13 | normal all-zone fan | Cameron: 3 | Parker: 8 | PASS |
| HVP13 | normal rear-right temperature | Cameron: 27°C | Parker: 24°C | PASS |
| HVP13 | stopped-audio playback | Cameron: resume | Parker: pause | PASS |
| HVP14 | normal sunroof | Parker: 18 | Sawyer: 100 | PASS |
| HVP14 | unnamed-navigation destination | Parker: Home | Sawyer: Suburban Mall | PASS |
| HVP14 | cargo-loading front trunk | Parker: open | Sawyer: closed | PASS |
| HVP15 | normal all-zone fan | Rowan: 8 | Morgan: 3 | PASS |
| HVP15 | normal driver temperature | Rowan: 18°C | Morgan: 23°C | PASS |
| HVP15 | stopped-audiobook playback | Rowan: resume | Morgan: pause | PASS |

## Second-pass findings and corrections

1. **HVP12 answer leakage:** `hvp12-turn-05` originally said Sydney
   “resumes” a stopped audiobook, which disclosed the Gold playback direction.
   It now states only that the audiobook stopped unexpectedly and asks for the
   remembered response.
2. **HVP14 answer leakage:** `hvp14-turn-08` originally named “rear defrost,”
   which repeated the Gold `mode=defrost`. It now gives the rear-visibility
   trigger and asks for the remembered climate response.
3. **HVP14 semantic risk:** both garage-to-vehicle conversions preserve source
   dialogue acts and Open/Closed polarity, but remain high-risk object rewrites
   requiring independent human rewriting or approval. Cargo questions were
   reframed as preparation; Sawyer's explicitly says no open request was made,
   avoiding the physically awkward reading of closing a compartment mid-load.
4. **HVP15 semantic risk:** the stopped-audiobook condition in the Pause
   exchange is an explicit adaptation used to create a matched condition, not a
   fact present in the Audio2Tool source. It remains queued for blind human
   rewrite and must not be described as verbatim human dialogue.
5. **Review-queue consistency:** the HVP12 Pause row incorrectly described the
   added condition as an interrupted audiobook. The data says audiobook
   listening; the queue now uses that exact scope and no longer calls the
   source action a correction. Because “listening while paused” is not a natural
   durable preference, its disposition was also raised from normalization
   approval to mandatory human rewrite.
6. **Condition completeness:** two HVP12 historical prompts now state normal
   driving for their temperature action, and `hvp14-turn-21` now states cabin
   venting for its paired roof action. Three access prompts now explicitly state
   that the owner has parked. These changes do not alter Gold or cutoff.
7. No other prompt disclosed a distinguishing number, color, destination,
   Boolean state, playback direction, or climate mode. Cutoff replay confirmed
   that every Gold fact was active and owned by the named user.

## Validation and human boundary

The corrected scenarios were rebuilt before deterministic validation. A second
clean rebuild produced byte-identical SHA-256 hashes for all 40 generated source,
attribution, manifest, snapshot, dialogue, Patch, and Quiz files. This document
also confirmed exact `quiz_id`/query/Gold-call parity between each source Quiz
and its derived TURN/FINAL JSONL row. It is model-assisted author review, not
real human validation; all 15 inserted
sessions and 95 added Quiz items remain subject to the blind independent
workflow in `HUMAN_REVIEW_QUEUE.md`.
