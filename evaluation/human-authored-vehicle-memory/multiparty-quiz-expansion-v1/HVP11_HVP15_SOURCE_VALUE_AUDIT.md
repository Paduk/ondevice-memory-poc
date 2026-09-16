# HVP11–HVP15 source-value audit

Status: **PASS for value grounding; independent human rewrite/approval pending.**

This second model-assisted pass compared every newly inserted session directly
with the external record's `expected_tool_call`, extracted arguments, and final
user turn. It did not inspect Summary, Patch, or Delta-v3 predictions.

| Scenario / session | External record target | V2 memory target | Audit result |
| --- | --- | --- | --- |
| `hvp11-mp01` | `tier6:4136`: voice guidance `muted=false` | guidance mode `detailed` | PASS; Boolean-on normalization |
| `hvp11-mp02` | `tier4:125`: HSB `0,100,85` (“red mood”) | ambient color `red` | PASS; device rewrite pending |
| `hvp11-mp03` | `tier7:243`: fan On, speed 5 | all-zone fan speed 5 | PASS; device rewrite pending |
| `hvp12-mp01` | `tier7:221`: temperature 23°C | driver-zone temperature 23°C | PASS; scope rewrite pending |
| `hvp12-mp02` | `tier7:523`: playback Pause | music switch `false` | PASS for target; durable-condition rewrite pending |
| `hvp12-mp03` | `tier7:335`: front door Unlocked | all vehicle doors unlocked | PASS; object/scope rewrite pending |
| `hvp13-mp01` | `tier7:270`: fan On, speed 8 | all-zone fan speed 8 | PASS; device rewrite pending |
| `hvp13-mp02` | `tier7:220`: temperature 24°C | rear-right temperature 24°C | PASS; scope rewrite pending |
| `hvp13-mp03` | `tier7:529`: corrected action Pause | music switch `false` | PASS; Pause normalization |
| `hvp14-mp01` | `tier7:458`: garage state Open | sunroof degree 100 | PASS for polarity; high-risk object rewrite pending |
| `hvp14-mp02` | `tier6:4119`: corrected stop `Suburban Mall` | unnamed-navigation default `Suburban Mall` | PASS; persistence approval pending |
| `hvp14-mp03` | `tier7:454`: garage state Closed | front-trunk switch `false` | PASS for polarity; high-risk object rewrite pending |
| `hvp15-mp01` | `tier7:248`: fan On, speed 3 | all-zone fan speed 3 | PASS; device rewrite pending |
| `hvp15-mp02` | `tier7:225`: temperature 23°C | driver-zone temperature 23°C | PASS; scope rewrite pending |
| `hvp15-mp03` | `tier7:526`: playback Pause | music switch `false` | PASS; condition rewrite pending |

No target value was rounded or inferred from conversational tone. The table
checks value grounding only: owner, vehicle object, scope, persistence, and
condition transformations still require the independent actions listed in
`HUMAN_REVIEW_QUEUE.md`.
