# HVP16 source-value audit

Status: **PASS for value grounding; independent human rewrite/approval pending.**

This model-assisted pass compared every newly inserted session directly with
the external record's `expected_tool_call`, extracted arguments, and final user
turn. It did not inspect Summary, Patch, or Delta-v3 predictions.

| Scenario / session | External record target | V2 memory target | Audit result |
| --- | --- | --- | --- |
| `hvp16-mp01` | `tier7:459`: garage state fully Open after rejecting halfway | sunroof degree 100 | PASS for Open/full-extent polarity; high-risk object rewrite pending |
| `hvp16-mp02` | `tier7:537`: playback Pause | music switch `false` | PASS; Pause normalization and condition approval pending |
| `hvp16-mp03` | `tier7:240`: thermostat 20°C | driver-zone temperature 20°C | PASS; scope and persistence rewrite pending |

No target value was rounded or inferred from conversational tone. The table
checks value grounding only: owner, vehicle object, scope, persistence, and
condition transformations remain in the independent review queue.
