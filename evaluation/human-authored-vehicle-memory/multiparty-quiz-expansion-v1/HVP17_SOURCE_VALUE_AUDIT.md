# HVP17 source-value audit

Status: **PASS for value grounding; independent human rewrite/approval pending.**

This model-assisted pass compared every newly inserted session directly with
the external record's `expected_tool_call`, extracted arguments, and final user
turn. It did not inspect Summary, Patch, or Delta-v3 predictions.

| Scenario / session | External record target | V2 memory target | Audit result |
| --- | --- | --- | --- |
| `hvp17-mp01` | `tier7:68`: dash soft blue, HSB `210,40,40` | ambient color `blue` | PASS for hue normalization; brightness excluded |
| `hvp17-mp02` | `tier7:231`: thermostat 19°C | all-zone cabin temperature 19°C | PASS; scope and persistence rewrite pending |
| `hvp17-mp03` | `tier7:480`: fully Open after rejecting halfway | rear-right window degree 100 | PASS for full-open polarity; high-risk object/condition rewrite pending |

No target value was rounded or inferred from conversational tone. The table
checks value grounding only: owner, object, scope, persistence, condition, and
safety transformations remain in the independent review queue.
