# HVP12 multi-participant author review

Status: **PASS for pilot use; independent human review remains pending.**

This model-assisted review was blind to Summary, Patch, and Delta-v3 outputs.
All inserted turns and 19 added Quiz items were manually checked against their
source, owner, condition, cutoff, and active memory value.

## Inserted dialogue

| Session | Review | Finding |
| --- | --- | --- |
| `hvp12-mp01` | PASS with rewrite required | Audio2Tool `tier7:221` is a coherent three-turn clarification that explicitly selects 23°C. The value is preserved, but the home thermostat is adapted to a vehicle driver-zone setting and needs an independent in-car rewrite. |
| `hvp12-mp02` | PASS for pilot; rewrite required | Audio2Tool `tier7:523` explicitly resolves a media command to Pause. The Tool mapping is grounded, but the durable “listening while paused” rule reads unnaturally and must be independently rewritten rather than merely approved. |
| `hvp12-mp03` | PASS with rewrite required | Audio2Tool `tier7:335` returns from a weather distraction to an explicit front-door unlock. The unlock value is preserved, but the home entry lock is adapted to all parked vehicle doors and must be rewritten independently. |

## Added Quiz review

| Quiz IDs | Primary type | Review finding |
| --- | --- | --- |
| `hvp12-turn-05`–`09` | Coreference resolution | PASS. The questions explicitly activate normal driving, stopped or playing audiobook, frequent prompts, frost, parked access, and unnamed navigation as needed. Every item names exactly one memory owner. |
| `hvp12-turn-10`–`12` | Error correction | PASS. The items select the post-replacement steering-wheel, parked-door, and destination records; none reveals the corrected value. |
| `hvp12-turn-13`–`18` | Preference conflict | PASS after wording correction. Three owner-direction pairs cover different driver temperatures, audiobook playback states, and parked-door states. Sydney's playback prompt now says the audiobook has stopped, while Rowan's says it is playing, so each desired action is pragmatically grounded. |
| `hvp12-turn-19`–`22` | State shift | PASS after wording correction. The two steering-wheel items and two door items lie on opposite sides of their respective replacements. The pre-door-replacement prompt now states that Sydney is parked, so the stored condition is explicitly active. |
| `hvp12-final-07` | Preference conflict | PASS. A single final item selects Rowan's three preferences against Sydney's alternatives, and the normal-drive, audiobook-playing, and parked conditions are all present. |

## Manual corrections and restraint

- Kept the exact external values—23°C, Pause, and Unlocked—rather than deriving
  new values from conversational tone.
- Added a temperature action to Rowan's pause and unlock combinations where the
  latter actions alone match the simulator's initial state; every added action
  is still selected from Rowan's active memory and activated by the prompt.
- Rephrased Sydney's playback conflict to require resuming a stopped audiobook,
  avoiding an unnatural instruction to start audio already described as playing.
- Rephrased `hvp12-turn-05` in a second-pass audit so that it states only the
  stopped-audiobook trigger, rather than leaking the remembered resume action.
- Rephrased the historical temperature items to activate normal driving, and
  the earlier parked-door item to activate both normal driving and its parked
  condition explicitly.
- Made both post-revision access-only prompts explicitly state that Sydney is
  parked, rather than using “parked rule” as a proxy for the trigger.
- No question states the distinguishing temperature, playback state, or lock
  state.

## Remaining human-validation requirement

- Audio2Tool query text is synthetic and CC BY-NC 4.0.
- All three inserted sessions require independent in-vehicle rewrites;
  `hvp12-mp02` specifically needs a natural durable audiobook condition while
  preserving its explicit Pause action.
- Every adaptation and Quiz must still receive blind human validation before
  this scenario is described as human-authored or human-validated.
