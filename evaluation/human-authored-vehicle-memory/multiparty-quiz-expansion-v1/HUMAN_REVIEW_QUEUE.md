# Independent human review queue

This queue is for a reviewer who has not seen Summary, Patch, or Delta-v3
predictions. Checking a box means the reviewer inspected the source trace,
rewrote or approved the adaptation, and independently verified the resulting
memory operation. It must not be pre-checked by the model-assisted author.

## Dialogue adaptation queue

| Priority | Session | Human action required | Sign-off |
| --- | --- | --- | --- |
| Rewrite | `hvp01-mp01` | Replace the room-speaker framing with natural in-car music dialogue while preserving source intent and volume 25. | ☐ |
| Rewrite | `hvp01-mp02` | Replace the home-thermostat framing with natural all-zone vehicle climate dialogue; retain automatic mode. | ☐ |
| Approve | `hvp01-mp03` | Check the vehicle-specific rear-trunk/groceries persistence sentence. | ☐ |
| Rewrite | `hvp02-mp01` | Replace room-speaker language with in-car music dialogue; retain volume 30. | ☐ |
| Rewrite | `hvp02-mp02` | Replace house entrance-lock language with natural all-vehicle-door departure dialogue. | ☐ |
| Approve | `hvp02-mp03` | Check the vehicle-navigation mute persistence sentence. | ☐ |
| Rewrite | `hvp03-mp01` | Replace home-fan framing with front-zone vehicle airflow dialogue; retain speed 7. | ☐ |
| Rewrite | `hvp03-mp02` | Replace room-speaker framing with an after-10pm in-car music dialogue; retain volume 20. | ☐ |
| Rewrite | `hvp03-mp03` | Rewrite the exchange so it is textually distinct from the duplicate source variant while preserving passenger cooling speed 1. | ☐ |
| Normalize | `hvp04-mp01` | Approve the narrowing of the vehicle-cabin 28°C request to Casey's driver zone. | ☐ |
| Rewrite | `hvp04-mp02` | Replace room-fan framing with front-zone vehicle airflow dialogue; retain speed 8. | ☐ |
| Rewrite | `hvp04-mp03` | Replace building-entrance language with a natural parked-at-home all-vehicle-door rule. | ☐ |
| Rewrite | `hvp05-mp01` | Replace room-speaker framing with natural in-car music dialogue; retain volume 50. | ☐ |
| Rewrite | `hvp05-mp02` | Replace non-vehicle door framing with a natural all-door store-parking preference. | ☐ |
| Approve | `hvp05-mp03` | Check the vehicle-specific 50% warm-drive sunroof dialogue and trim only if needed. | ☐ |
| Approve | `hvp06-mp01` | Check the front-only cold-drive defrost persistence sentence. | ☐ |
| Normalize | `hvp06-mp02` | Confirm that emergency filtration remains a one-off source request and ordinary recirculation is a separate durable preference. | ☐ |
| Approve | `hvp06-mp03` | Check the corrected rear-trunk-close luggage routine. | ☐ |
| Normalize | `hvp07-mp01` | Approve mapping Boolean voice-guidance-on to explicit `detailed` guidance. | ☐ |
| Approve | `hvp07-mp02` | Check the fresh-air → outside-circulation normalization. | ☐ |
| Normalize | `hvp07-mp03` | Confirm that the source play/pause repair supports the explicitly paused durable playlist state. | ☐ |
| Normalize | `hvp08-mp01` | Confirm that only 22°C is durable; AUTO remains an immediate, non-memory action. | ☐ |
| Normalize | `hvp08-mp02` | Confirm that only red is durable; brightness 100 remains an immediate, non-memory action. | ☐ |
| Normalize | `hvp08-mp03` | Approve the no-automatic-open rear-trunk rule for bulky-item loading. | ☐ |
| Normalize | `hvp09-mp01` | Approve soft-blue → `blue` and the added romantic-cabin condition; brightness 90 remains non-durable. | ☐ |
| Normalize | `hvp09-mp02` | Approve the no-automatic-open rear-trunk rule for heavy shopping. | ☐ |
| Normalize | `hvp09-mp03` | Approve mapping Boolean voice-guidance-on to `detailed` during audiobooks. | ☐ |
| Rewrite | `hvp10-mp01` | Independently rewrite the implicit “too cold” complaint into a natural explicit 28°C front-passenger request; the value and scope currently come from the source label/adaptation. | ☐ |
| Approve | `hvp10-mp02` | Check the multi-turn repair to both front and rear defrost. | ☐ |
| Approve | `hvp10-mp03` | Check the cold-hands steering-wheel-heat persistence sentence. | ☐ |
| Normalize | `hvp11-mp01` | Approve mapping Boolean vehicle voice-guidance-on to `detailed` guidance during audiobooks. | ☐ |
| Rewrite | `hvp11-mp02` | Replace room-light framing with natural in-car ambient-light dialogue; retain red. | ☐ |
| Rewrite | `hvp11-mp03` | Replace home-fan framing with all-zone vehicle airflow dialogue; retain speed 5. | ☐ |
| Rewrite | `hvp12-mp01` | Replace home-thermostat framing with natural driver-zone vehicle climate dialogue; retain 23°C. | ☐ |
| Rewrite | `hvp12-mp02` | Rewrite the explicit Pause action as a natural durable in-car audiobook rule; retain `music_switch=false` without implying that listening is always paused. | ☐ |
| Rewrite | `hvp12-mp03` | Replace home front-door language with a natural parked-vehicle all-door unlock rule. | ☐ |
| Rewrite | `hvp13-mp01` | Replace living-room fan framing with all-zone vehicle airflow dialogue; retain speed 8. | ☐ |
| Rewrite | `hvp13-mp02` | Replace home-thermostat framing with natural rear-right vehicle climate dialogue; retain 24°C. | ☐ |
| Normalize | `hvp13-mp03` | Approve mapping the Stop→Pause correction to leaving interrupted vehicle audio paused. | ☐ |
| Rewrite | `hvp14-mp01` | Independently rewrite/approve the disclosed garage-open dialogue as a fully open vehicle-sunroof exchange; retain the Open polarity and confirmation structure. | ☐ |
| Normalize | `hvp14-mp02` | Approve promoting the corrected `Suburban Mall` trip stop to Sawyer's unnamed-navigation default. | ☐ |
| Rewrite | `hvp14-mp03` | Independently rewrite/approve the disclosed garage-close dialogue as a front-trunk-close exchange and review the cargo-loading persistence condition. | ☐ |
| Rewrite | `hvp15-mp01` | Independently rewrite/approve the disclosed home-fan exchange as natural all-cabin vehicle dialogue; preserve speed 3 and its distraction-and-return structure. | ☐ |
| Rewrite | `hvp15-mp02` | Rewrite the thermostat exchange as natural driver-zone vehicle climate dialogue; retain 23°C. | ☐ |
| Rewrite | `hvp15-mp03` | Independently approve the Pause exchange rewritten as an unexpectedly stopped in-car audiobook and its no-resume persistence rule. | ☐ |
| Rewrite | `hvp16-mp01` | Independently rewrite/approve the disclosed fully Open garage-door exchange as a fully open vehicle-sunroof exchange; preserve the rejection of a half-open state and review the 100% mapping. | ☐ |
| Normalize | `hvp16-mp02` | Approve mapping the explicit Pause request to `music_switch=false` and the added stopped-playback persistence rule. | ☐ |
| Rewrite | `hvp16-mp03` | Rewrite/approve the thermostat exchange as natural driver-zone vehicle climate dialogue; retain 20°C. | ☐ |
| Normalize | `hvp17-mp01` | Approve HSB hue 210 → `blue`, excluding brightness from memory, and the appended normal ambient-light persistence statement. | ☐ |
| Rewrite | `hvp17-mp02` | Rewrite/approve the thermostat exchange as natural all-zone vehicle climate dialogue; retain 19°C. | ☐ |
| Rewrite | `hvp17-mp03` | Independently rewrite/approve the fully Open correction as a 100% rear-right-window preference; review both the child-occupant condition and safety wording. | ☐ |
| Rewrite | `hvp18-mp01` | Rewrite/approve the generic blue-light exchange as vehicle ambient-light dialogue; retain blue, exclude brightness, and approve persistence. | ☐ |
| Rewrite | `hvp18-mp02` | Rewrite/approve the room-thermostat exchange as all-zone vehicle climate dialogue; retain 20°C. | ☐ |
| Normalize | `hvp18-mp03` | Approve promoting the already driving-situated volume-20 request to Riley's normal in-car music preference. | ☐ |
| Rewrite | `hvp19-mp01` | Rewrite/approve the generic blue-light exchange as vehicle ambient-light dialogue; retain blue, exclude brightness, and approve persistence. | ☐ |
| Rewrite | `hvp19-mp02` | Independently rewrite/approve the home-entry lock exchange as a parked-vehicle all-door rule; preserve Locked and review scope. | ☐ |
| Rewrite | `hvp19-mp03` | Rewrite/approve the speaker-volume exchange as in-car music dialogue; retain volume 50 and approve persistence. | ☐ |
| Rewrite | `hvp20-mp01` | Rewrite/approve the soft-yellow light exchange as vehicle ambient-light dialogue; retain yellow, exclude brightness, and approve persistence. | ☐ |
| Rewrite | `hvp20-mp02` | Rewrite/approve the thermostat exchange as natural all-zone vehicle climate dialogue; retain 25°C. | ☐ |
| Rewrite | `hvp20-mp03` | Independently rewrite/approve the home-entry lock exchange as a parked-vehicle all-door rule; preserve Locked and review scope. | ☐ |

## Quiz review protocol

Review all 374 added Quiz items directly in each `scenario-source.json`; do not
review model predictions or scores at this stage. For every item, record PASS or
a corrected question/Gold call after checking:

1. the named owner and all `evidence_slot_ids` refer to one person;
2. the fact is active at `cutoff_turn_id`, not merely present earlier or later;
3. the condition in the question activates every selected memory fact;
4. the question does not state the distinguishing answer value or scope;
5. every Gold Tool and argument is the intended action;
6. multi-Tool combinations are physically and conversationally plausible; and
7. preference-conflict items contain a genuine other-owner alternative, not a
   same-value pseudo-conflict.

After edits, rebuild and rerun the deterministic validator before signing a
versioned human-validated release.
