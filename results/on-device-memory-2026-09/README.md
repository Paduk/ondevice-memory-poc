# On-device Memory Evaluation Results (2026-09)

This bundle contains the compact, commit-friendly evaluation artifacts referenced by the latest results document.

- Archive: `results.tar.gz`
- Files in archive: 253 JSON/CSV files
- Uncompressed size: 44,196,148 bytes
- Compressed size: 2,074,307 bytes
- Layout: `runs/...` and `benchmarks/...`

Included:

- Final fixed-ratio validation and Test Composite summaries for the 350M–2B models
- Patch, Summary, Delta-v3, and compact-memory results for k=2, 5, and 10
- KV-cache and prefill benchmark summaries, manifests, comparisons, and CSV tables

Excluded:

- Model checkpoints and training state
- Raw `turns.jsonl` traces
- Training/evaluation logs and progress files

The JSON/CSV contents are unchanged, including existing absolute source-path strings.

Extract and verify:

```bash
sha256sum -c results.tar.gz.sha256
tar -xzf results.tar.gz
```

See `docs/engineering/on-device-memory-latest-results.md` for the human-readable result summary.
