# HVP07 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to Summary, Patch, and Delta-v3 outputs.
Every inserted source utterance and all 18 added Quiz items were read against
the memory state at their exact cutoff.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp07-mp01` | PASS with note | Audio2Tool `tier6:4137` explicitly corrects muted guidance to voice guidance on. VehicleMemBench has categorical voice modes rather than a Boolean unmute, so the added sentence explicitly selects `detailed`; a human should approve that normalization. |
| `hvp07-mp02` | PASS | Audio2Tool `tier6:3990` corrects Bioweapon Defense to fresh air. Mapping fresh air to outside circulation is direct, and the added sentence only makes Avery's preference durable. |
| `hvp07-mp03` | PASS with note | Audio2Tool `tier6:4097` starts from a pause request and resolves to the play/pause control. The added sentence removes toggle-state ambiguity by explicitly stating Avery's durable paused-playlist preference. A human should confirm that this small clarification preserves the source intent. |

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp07-turn-05`–`09` | Coreference resolution | PASS. The roof value, cabin-air source, playback state, equipment compartment, and Avery-owned settings all resolve to active facts at the stated cutoffs. |
| `hvp07-turn-10`–`12` | Error correction | PASS. The target values are Morgan's corrected 18% sunroof and outside air, plus Avery's corrected spoken guidance. |
| `hvp07-turn-13`–`18` | Preference conflict | PASS. Guidance (detailed vs. mute), cabin air (outside vs. inside), and playlist playback (paused vs. playing) are true owner conflicts at cutoff `hvp07-s9-t03`, each tested in both directions. The air conflict intentionally precedes Morgan's later switch to outside air. |
| `hvp07-turn-19`–`21` | State shift | PASS. The two cabin-air questions straddle Morgan's inside→outside replacement, and the roof question uses the later 18% value rather than 50%. |
| `hvp07-final-07` | Coreference resolution | PASS. Avery is the only named owner; detailed guidance, outside air, and paused playback remain active in the final snapshot. |

## Manual corrections made

- Selected the fresh-air correction (`tier6:3990`) instead of treating an
  emergency filtration mode as synonymous with recirculation.
- Kept the cabin-air owner conflict at the earlier cutoff where Avery and
  Morgan genuinely have different active values.
- Paired Avery's default-state paused call with a fresh-air call so the official
  simulator evaluates a real state change rather than accepting a no-op Quiz.

## Remaining human-validation requirement

- Audio2Tool query text is synthetic and CC BY-NC 4.0.
- An independent human must approve or rewrite the guidance-mode and play/pause
  normalizations and independently check all Quiz items while blind to model
  predictions before this scenario is called human-authored or human-validated.
