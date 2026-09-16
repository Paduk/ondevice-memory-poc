# HVP18 source-value audit

Status: **PASS for value grounding; independent human rewrite/approval pending.**

This model-assisted pass compared every newly inserted session directly with
the external record's `expected_tool_call`, extracted arguments, and final user
turn. It did not inspect Summary, Patch, or Delta-v3 predictions.

| Scenario / session | External record target | V2 memory target | Audit result |
| --- | --- | --- | --- |
| `hvp18-mp01` | `tier7:70`: blue, HSB `240,100,100` | ambient color `blue` | PASS for hue normalization; brightness excluded |
| `hvp18-mp02` | `tier7:227`: thermostat 20°C | all-zone cabin temperature 20°C | PASS; scope and persistence rewrite pending |
| `hvp18-mp03` | `tier7:560`: in-traffic speaker volume 20 | in-car music volume 20 | PASS; vehicle context already explicit, persistence approval pending |

No target value was rounded or inferred from conversational tone. The table
checks value grounding only: owner, object, scope, and persistence
transformations remain in the independent review queue.
