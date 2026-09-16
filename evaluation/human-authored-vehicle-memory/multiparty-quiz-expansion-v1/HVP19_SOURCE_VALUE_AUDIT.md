# HVP19 source-value audit

Status: **PASS for value grounding; independent human rewrite/approval pending.**

This model-assisted pass compared every newly inserted session directly with
the external record's `expected_tool_call`, extracted arguments, and final user
turn. It did not inspect Summary, Patch, or Delta-v3 predictions.

| Scenario / session | External record target | V2 memory target | Audit result |
| --- | --- | --- | --- |
| `hvp19-mp01` | `tier7:67`: blue, HSB `240,100,50` | ambient color `blue` | PASS for hue normalization; brightness excluded |
| `hvp19-mp02` | `tier7:345`: front door Locked | all vehicle doors locked | PASS for lock polarity; high-risk object/scope rewrite pending |
| `hvp19-mp03` | `tier7:547`: speaker volume 50 | in-car music volume 50 | PASS; object and persistence rewrite pending |

No target value was rounded or inferred from conversational tone. The table
checks value grounding only: owner, object, scope, condition, and persistence
transformations remain in the independent review queue.
