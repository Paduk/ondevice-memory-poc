# Section 5 UPDATE-cycle efficiency

`applied_update=true`인 턴의 decode와 동일 시나리오의 바로 다음 턴에서
실제로 계산한 prefill을 결합한 재현 결과다.

| Model | Method | Cycles | UPDATE decode tokens | Next evaluated prefill tokens | UPDATE-cycle latency (s) | Relative (Summary=100) | All-turn model tokens | All-turn latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Granite 350M | Summary | 33 | 264.88 | 196.21 | 6.54 | 100.0 | 127.35 | 1.26 |
| Granite 350M | Patch | 48 | 82.29 | 189.00 | 2.96 | 45.3 | 115.47 | 1.03 |
| Granite 350M | Delta-v3 (k=5) | 38 | 79.71 | 156.32 | 2.67 | 40.9 | 96.03 | 0.87 |
| Qwen 0.8B | Summary | 39 | 272.69 | 367.21 | 15.28 | 100.0 | 187.24 | 3.46 |
| Qwen 0.8B | Patch | 64 | 94.11 | 338.06 | 8.38 | 54.9 | 175.66 | 3.03 |
| Qwen 0.8B | Delta-v3 (k=5) | 32 | 86.09 | 306.16 | 7.62 | 49.9 | 136.62 | 2.30 |
| Granite 1B | Summary | 59 | 260.54 | 192.12 | 23.36 | 100.0 | 143.39 | 6.55 |
| Granite 1B | Patch | 59 | 85.02 | 187.27 | 12.47 | 53.4 | 117.93 | 5.00 |
| Granite 1B | Delta-v3 (k=5) | 41 | 77.78 | 162.98 | 11.08 | 47.4 | 95.98 | 4.05 |
| Llama 3.2 1B | Summary | 34 | 247.44 | 196.47 | 18.00 | 100.0 | 126.79 | 4.17 |
| Llama 3.2 1B | Patch | 55 | 83.71 | 181.27 | 9.31 | 51.7 | 117.61 | 3.70 |
| Llama 3.2 1B | Delta-v3 (k=5) | 35 | 76.43 | 167.34 | 8.55 | 47.5 | 95.08 | 2.96 |
| Llama 3.2 3B | Summary | 27 | 231.52 | 176.70 | 47.63 | 100.0 | 121.53 | 11.98 |
| Llama 3.2 3B | Patch | 45 | 84.27 | 193.20 | 28.95 | 60.8 | 115.40 | 11.13 |
| Llama 3.2 3B | Delta-v3 (k=5) | 33 | 76.00 | 159.91 | 24.82 | 52.1 | 94.70 | 9.13 |
| Qwen 2B | Summary | 28 | 303.07 | 382.32 | 36.38 | 100.0 | 182.09 | 8.16 |
| Qwen 2B | Patch | 53 | 90.75 | 350.87 | 20.46 | 56.3 | 172.83 | 7.51 |
| Qwen 2B | Delta-v3 (k=5) | 42 | 87.36 | 279.05 | 17.32 | 47.6 | 138.72 | 6.05 |
