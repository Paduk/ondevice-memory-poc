# Delta-v3 pending-depth sampling audit

Counts include independent examples and trajectory-window exposures. The JSON artifact retains per-seed and per-epoch details.

| Condition | Decision | d=0 | d=1 | d=2 | d=3 | d=4 | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| unstratified | NO_OP | 9061 | 8610 | 8341 | 9567 | 7185 | 42764 |
| unstratified | UPDATE | 1473 | 1417 | 1363 | 1249 | 1171 | 6673 |
| depth-weighted | NO_OP | 15314 | 11681 | 6976 | 5452 | 3341 | 42764 |
| depth-weighted | UPDATE | 1473 | 1417 | 1363 | 1249 | 1171 | 6673 |

## Encoded Memory-SFT token exposures

| Condition | Examples | Input tokens | Target tokens | Total tokens | At max length |
|---|---:|---:|---:|---:|---:|
| unstratified | 49437 | 29153922 | 836771 | 29990693 | 6 |
| depth-weighted | 49437 | 27439318 | 836771 | 28276089 | 6 |

Depth-weighted total tokens differ by -5.72% from unstratified sampling; the example count is held fixed.

The two conditions use identical UPDATE, total independent NO_OP, adjacent-NO_OP, trajectory, and Quiz budgets; only independent NO_OP selection uses pending depth in the depth-weighted condition.
