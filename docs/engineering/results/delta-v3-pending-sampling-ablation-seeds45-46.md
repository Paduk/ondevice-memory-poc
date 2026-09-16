# Delta-v3 pending-depth sampling ablation

All values are mean ± sample standard deviation across paired seeds.

| Metric | Unstratified random | Weighted 40/30/15/10/5 | Paired Δ |
|---|---:|---:|---:|
| test_composite | 0.6760 ± 0.0099 | 0.6414 ± 0.0146 | -0.0346 ± 0.0047 |
| closed_loop_quiz_esm | 0.7167 ± 0.0177 | 0.6661 ± 0.0184 | -0.0505 ± 0.0007 |
| closed_loop_final_state_f1 | 0.4903 ± 0.0054 | 0.4976 ± 0.0163 | +0.0073 ± 0.0217 |
| closed_loop_update_f1 | 0.8226 ± 0.0043 | 0.7819 ± 0.0035 | -0.0407 ± 0.0078 |
| closed_loop_false_update_rate | 0.0165 ± 0.0032 | 0.0119 ± 0.0018 | -0.0046 ± 0.0050 |
| diagnostic_update_recall | 0.7641 ± 0.0243 | 0.7000 ± 0.0088 | -0.0641 ± 0.0331 |
| diagnostic_false_update_rate | 0.0125 ± 0.0044 | 0.0078 ± 0.0022 | -0.0047 ± 0.0066 |

Paired Δ is weighted minus unstratified for the same seed.

## Existing uniform-depth reference

The current `1/1/1/1/1` result is a single unpaired reference, not a third condition in this ablation.

| Test composite | Quiz ESM | Final-memory F1 | UPDATE F1 | False-update rate |
|---:|---:|---:|---:|---:|
| 0.6738 | 0.7073 | 0.4922 | 0.8422 | 0.0142 |
