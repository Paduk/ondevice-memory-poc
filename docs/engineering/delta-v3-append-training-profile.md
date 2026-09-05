# Delta-v3 append-only training profile

Status: implementation ready for a training canary.

## Contract

- Keep the existing `delta_v3` method and checkpoints unchanged.
- Train the new method key `delta_v3_append` from the existing aligned `delta.jsonl` view.
- Start one cache epoch with `base_summary` and the first `current_turn`.
- Append later user turns and gold assistant `NO_OP`/`UPDATE` outputs without moving deltas into a rebuilt input object.
- Reset the epoch after five UPDATEs, a discontinuous source turn, or 32 retained turns.
- Mask the complete history and train loss only on the latest assistant output.

The 32-turn bound prevents NO_OP-heavy trajectories from growing without limit. At an early reset, the current canonical materialized memory becomes the next epoch base.

## Training

The profile derives histories at dataset load time, so no copied training JSONL is required.

```bash
bash memory_training/scripts/run_granite4_1b_delta_v3_append_train_first.sh 0 4 45 r1
```

The initial runner uses length 4096, batch size 2, and gradient accumulation 8 because append-only examples are longer than the single-turn Delta-v3 examples.

An initial Granite tokenizer preview over all 16,576 validation rows produced a 2,055-token mean and 4,021-token maximum with the 32-turn bound. A uniform 1,024-row preview over the 278,775-row training split produced a 2,055-token mean, 3,430-token p95, and 3,732-token maximum. No previewed example reached the 4,096-token truncation limit.

Generation validation is intentionally disabled in the generic training validator for this first profile. The dedicated HF prefix-cache benchmark now retains the exact user/assistant transcript, generated-token cache identity, and matching epoch boundaries for latency comparison; full-trajectory append closed-loop quality validation remains separate.
