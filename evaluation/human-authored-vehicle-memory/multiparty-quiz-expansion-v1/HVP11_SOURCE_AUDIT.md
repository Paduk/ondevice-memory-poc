# HVP11 source-availability audit

Status: **RESOLVED after a broader source-policy audit.**

The first pass stopped before construction because it considered only unused
vehicle-specific Audio2Tool records. Under that narrower filter, only one
same-target owner conflict was available. The resumed pass checked the policy
already used by `HVP01`–`HVP10`: an external action and value may be retained
while its controlled device is minimally adapted to the vehicle, provided the
adaptation is explicit and queued for independent human rewriting.

## Accepted source support

- `tier6:4136` is vehicle-specific and directly supports navigation guidance
  being turned on. It is normalized to `detailed` and conditioned on audiobook
  listening, contrasting with Quinn's `mute` preference.
- `tier4:125` explicitly encodes a red light action. Red is retained while the
  room light is adapted to the vehicle's ambient light, contrasting with
  Quinn's blue setting.
- `tier7:243` explicitly settles on fan speed 5. The value is retained while the
  home fan is adapted to the all-zone cabin fan, contrasting with Quinn's final
  speed 8.

These three records are disjoint from the frozen pilots and all earlier
extensions. No value was invented, no different seat or zone was mislabeled as
a conflict, and no query-only action was converted into an update.

## Remaining boundary

The latter two device adaptations are intentionally marked for independent
human rewriting. This resolution makes `HVP11` valid as a model-assembled,
`pilot_excluded` scenario; it does not make the data human-authored or
publication-ready.
