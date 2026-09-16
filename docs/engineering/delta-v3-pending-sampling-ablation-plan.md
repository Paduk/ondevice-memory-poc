# Delta-v3 pending-depth sampling ablation

## Current setup

- `memory_training.train` defaults Delta-v3 to pending-depth weights
  `40/30/15/10/5` for depths `d=0..4`.
- The current Granite 4 1B Delta-v3 compact k=5 result used in the main table was
  trained with explicit weights `1/1/1/1/1`. Its run configuration predates the
  explicit sampling-mode field, so that field is absent. On the fixed test it has
  Quiz ESM 0.7073, Final-memory F1 0.4922, UPDATE F1 0.8422, and composite 0.6738.
- This ablation does not treat the existing uniform checkpoint as either new
  condition. It compares unstratified random sampling against the proposed weighted
  sampling while preserving the existing uniform result as a reference.

## Comparison

| Item | Unstratified random | Depth-weighted |
|---|---|---|
| Independent NO_OP sampling | No depth information | `40/30/15/10/5` for `d=0..4` |
| Base model | Granite 4 1B | Same |
| Model/dropout training seeds | 45, 46, 47 | Paired 45, 46, 47 |
| Data/trajectory sampling seed | Fixed at 45 | Same |
| UPDATE examples | All eligible examples | Same |
| NO_OP budget | 5 per UPDATE | Same |
| Adjacent NO_OP target | 30% | Same |
| Trajectory target | 20%, paired seed | Identical windows for the same seed |
| Quiz exposure and optimization | Fixed | Same |
| Delta runtime and k | Compact k=5 | Same |

The primary intervention is whether independent NO_OP selection uses pending depth.
Actual Memory-SFT example and token exposures are reported because trajectories can
duplicate turns and changing the depth mix can change encoded input length.

## Realized exposure audit

The pre-run audit confirms that both conditions contain 49,437 Memory-SFT exposures,
including 42,764 NO_OP and 6,673 UPDATE exposures. UPDATE counts and fixed trajectory
windows are identical. After trajectory exposures are included, the NO_OP counts for
`d=0..4` are:

- unstratified: `9,061/8,610/8,341/9,567/7,185`;
- weighted: `15,314/11,681/6,976/5,452/3,341`.

The encoded target-token count is identical at 836,771, while total encoded tokens are
29,990,693 for unstratified and 28,276,089 for weighted sampling (-5.72%). This input
length difference is reported rather than removed by changing the selected examples.
Full counts are in `docs/engineering/results/delta-v3-pending-sampling-audit.json`.

## Evaluation

Two evaluations are kept separate.

1. Fixed gold-state diagnostic: the same balanced UPDATE and NO_OP rows at each
   pending depth are supplied to every checkpoint. Report per-depth UPDATE recall and
   NO_OP false-update rate, followed by their micro aggregates.
2. Closed-loop test: report Final-memory F1, Quiz ESM, UPDATE F1, and false-update
   rate on the fixed 24-scenario test set. This captures accumulated errors but is not
   used to interpret per-depth behavior because model errors can alter later states.

Validation checkpoint selection uses only the six fixed validation scenarios and the
existing `0.60 Quiz ESM + 0.25 Final-memory F1 + 0.15 UPDATE F1` composite. Test
results do not select checkpoints.

## Commands

First audit the realized training distribution:

```bash
/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python \
  -m memory_training.audit_delta_pending_sampling \
  --data-root /mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2-delta-v3-compact-k5-v1 \
  --workspace /mnt/data/hj153lee/PalmClaw/on-device-memory-training \
  --output docs/engineering/results/delta-v3-pending-sampling-audit.json
```

Run the six paired jobs by assigning an available GPU to each command:

```bash
memory_training/scripts/run_granite4_1b_pending_sampling_ablation.sh GPU unstratified 45 r1
memory_training/scripts/run_granite4_1b_pending_sampling_ablation.sh GPU unstratified 46 r1
memory_training/scripts/run_granite4_1b_pending_sampling_ablation.sh GPU unstratified 47 r1
memory_training/scripts/run_granite4_1b_pending_sampling_ablation.sh GPU depth-weighted 45 r1
memory_training/scripts/run_granite4_1b_pending_sampling_ablation.sh GPU depth-weighted 46 r1
memory_training/scripts/run_granite4_1b_pending_sampling_ablation.sh GPU depth-weighted 47 r1
```

When two GPUs are idle, the paired queue alternates conditions across the devices and
runs one seed pair at a time:

```bash
memory_training/scripts/run_granite4_1b_pending_sampling_ablation_queue.sh GPU_A GPU_B r1
```

After all jobs finish:

```bash
/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python \
  -m memory_training.aggregate_delta_pending_ablation \
  --workspace /mnt/data/hj153lee/PalmClaw/on-device-memory-training \
  --seeds 45 46 47 \
  --run-tag r1 \
  --output docs/engineering/results/delta-v3-pending-sampling-ablation.json
```

## Interpretation rule

The weighted checkpoint replaces the current main Delta-v3 result only if the paired
runs show a consistent benefit in the gold-state depth diagnostic, no material
regression in closed-loop Final-memory F1 or Quiz ESM, and no regression against the
existing single-seed uniform reference. Otherwise, retain the current main-table
result and describe `1/1/1/1/1` as its actual training configuration. If the paper
still needs a general pending-aware-versus-unaware claim in that case, add uniform
seeds 46 and 47 and compare uniform against unstratified; do not claim that the
specific `40/30/15/10/5` schedule is necessary.
