# HVP20 source-value audit

Status: **PASS for value grounding; independent human rewrite/approval pending.**

This model-assisted pass compared every newly inserted session directly with
the external record's `expected_tool_call`, extracted arguments, and final user
turn. It did not inspect Summary, Patch, or Delta-v3 predictions.

| Scenario / session | External record target | V2 memory target | Audit result |
| --- | --- | --- | --- |
| `hvp20-mp01` | `tier7:73`: soft yellow, HSB `60,70,80` | ambient color `yellow` | PASS for hue normalization; brightness excluded |
| `hvp20-mp02` | `tier7:229`: thermostat 25°C | all-zone cabin temperature 25°C | PASS; scope and persistence rewrite pending |
| `hvp20-mp03` | `tier7:337`: front door Locked | all vehicle doors locked | PASS for lock polarity; high-risk object/scope rewrite pending |

No target value was rounded or inferred from conversational tone. The table
checks value grounding only: owner, object, scope, condition, and persistence
transformations remain in the independent review queue.
