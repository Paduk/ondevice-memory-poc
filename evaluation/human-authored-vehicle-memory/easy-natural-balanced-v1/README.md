# Easy natural-balanced HVE scenarios

This calibration variant lowers the HVE11–HVE20 difficulty identified in the
first natural-explicit evaluation while preserving all Gold semantics.

- HVE01–HVE10 dialogue and Quiz text is unchanged from
  `easy-natural-explicit-v1`.
- HVE11–HVE20 UPDATE turns contain one natural, explicit durable preference and
  no unrelated one-time vehicle command.
- The first human turn of every HVE11–HVE20 NO_OP session naturally establishes
  that it is a one-off request at home.
- Owners, values, ADD/REPLACE operations, timestamps, Quiz cutoffs, and Gold
  Tool calls remain unchanged.

This remains a model-assisted calibration pilot, not independently human-
authored evaluation data.
