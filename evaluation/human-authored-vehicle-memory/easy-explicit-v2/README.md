# Easy explicit v2 calibration set

Easy-v2 is a ceiling/sanity-check revision of Easy-v1. It preserves every
scenario identity, owner, timestamp, memory operation, cutoff, and Gold Tool
call while simplifying only the observable language.

## Changes from v1

- UPDATE turns contain one durable vehicle-profile fact and no competing
  one-time vehicle command.
- REPLACE turns explicitly say to use the latest value and discard the old one.
- Every human NO_OP says that it is a one-time, non-vehicle request.
- Quiz requests use direct, uniform owner-and-setting wording.
- The 30-turn, two-owner, `UPDATE 6 : NO_OP 24`, `ADD 4 : REPLACE 2`, and
  `Turn Quiz 8 : Final Quiz 8` structure is unchanged.

## Interpretation boundary

This is intentionally label-transparent calibration data. It should be used to
test whether the pipeline can reach a high ceiling under unambiguous inputs,
not as evidence of natural conversational generalization. It remains
model-assembled, `pilot_excluded`, and ineligible for a human-authored-test
claim. Easy-v1 remains preserved for before/after comparison.

The external record ID and original text remain in every turn's `source_trace`.
The changed `reuse_mode` distinguishes single-intent UPDATE rewrites and
explicit non-vehicle wrappers from verbatim reuse.

