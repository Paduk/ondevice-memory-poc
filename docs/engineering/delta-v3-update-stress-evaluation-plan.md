# Delta-v3 Update Stress 평가 계획

## 목적

자연 workload(`NO_OP:UPDATE = 5:1`)에서 확인된 Delta-v3의 KV-cache 이점이
UPDATE 누적에 따라 어떻게 변하는지 Patch 및 Delta-v3 `k=2/5/10`과 비교한다.
주 결과와 stress case-study를 분리해 cherry-picking으로 해석되지 않게 한다.

## 평가 데이터

- **개발 기준선:** 고정 Validation의 `S81-S85` 원본 trajectory
- **Validation Stress:** `S81-S85 × {20, 40, 60, 80 total UPDATE}` = 20 trajectories
- **Final Test Stress:** 규칙 동결 후 `S86-S90`에 같은 4개 규모를 한 번 적용한다.
- 원본 고정 데이터의 legacy memory catalog는 `S86-S90`을 `validation`으로 기록하지만,
  최신 Quiz/평가 protocol은 이를 `test`로 사용한다. 생성 artifact에는
  `memory_split`과 `quiz_split`을 각각 기록하고 최종 실행은 scenario ID를 명시한다.
- 기존 UPDATE/NO_OP과 순서는 보존하고, 목표 수까지 synthetic ADD만 추가한다.
- Synthetic ADD는 vehicle tool schema를 따르는 짧고 명시적인 단일 조건부 선호로
  만든다. 새로운 key만 사용해 ADD 여부가 모호하지 않고 기존 Quiz 정답과 충돌하지
  않게 한다.
- Synthetic turn은 trajectory 전체에 고르게 배치하고, memory 삽입 위치는 앞/중간을
  1:1로 고정한다. 생성 seed와 master update pool을 고정해 재현성을 보장한다.
- 같은 cell에서는 Patch와 모든 Delta k가 동일한 turn, operation, 최종 memory를 쓴다.

## Quiz와 정확도

- 원본 시나리오당 40개 Quiz와 Gold Tool-call label을 유지한다.
- 삽입된 turn에 맞춰 Quiz의 `memory_ref`와 snapshot hash를 새 trajectory로 재매핑한다.
- 기존 Quiz는 장기 업데이트 후 원래 기억의 보존·검색 정확도를 평가한다.
- 합성 사실 자체는 UPDATE F1, intermediate/final-state F1로 평가한다.
- Latency의 주 비교는 **controlled replay**, 실제 품질과 운영 비용은
  **predicted closed loop**로 별도 보고한다.

## 실행 매트릭스

- 모델: Qwen3.5 0.8B, seed45 best checkpoint
- 방법: Patch, Delta-v3 `k=2`, `k=5`, `k=10`
- Cache: OFF, ON, ON + background prefill
- 동일 GPU·precision·context limit·greedy decoding을 사용하고 warm-up 후 3회 이상
  반복한다. 실행 전 context 초과와 truncation을 검사한다.

## 측정 및 보고

각 cell에서 `mean / p50 / p95 / total`을 산출한다.

- logical, reused, evaluated prefill tokens와 cache reuse ratio
- decode tokens, prefill/decode/TTFT/end-to-end latency, cache 관리 시간
- foreground/background/total 비용, KV-cache bytes, peak CUDA memory
- 전체 turn, NO_OP, UPDATE 직후, compaction/non-compaction 구간
- UPDATE F1, intermediate/final-state F1, Quiz ESM
- HF 실측과 `30 prefill tok/s` on-device projection을 분리 표기

최종 표는 update 수에 따른 **누적 비용**, **turn 평균**, **p95 latency**, **accuracy**를
함께 보여주고, Patch 대비 절감률과 각 k의 compaction trade-off를 제시한다.

## 구현 순서

1. Stress generator를 5개 시나리오와 4개 UPDATE 규모로 확장한다.
2. 원본 Quiz를 재매핑하고 데이터 정합성·최종 memory 동일성을 검증한다.
3. Validation 결과로 생성 규칙·k·지표·report 형식을 확정하고 동결한다.
4. 동결된 규칙으로 Test Stress를 한 번 생성·실행한다.
5. 전체 실행 매트릭스를 자동화하고 원시 `turns.jsonl`/`summary.json`을 저장한다.
6. latency·token·memory·accuracy를 하나의 JSON/Markdown report로 집계한다.

## 완료 조건

- 자연 기준선 5개와 stress 20개 trajectory 검증 통과
- Cache OFF/ON decision 및 state-transition 동등성 점검 완료
- 모든 방법·k·cache mode의 반복 실행 완료
- 평균/p95/누적 비용과 closed-loop accuracy가 포함된 최종 비교표 생성

## 현재 상태 (2026-09-04)

- Validation S81-S85의 cache-ON controlled replay canary를 반복 1회로 완료했다.
- 20/40/60/80 UPDATE × Patch/k2/k5/k10의 16개 job이 모두 성공했다.
- 같은 16개 cell의 predicted closed-loop memory와 Quiz 200개 평가를 완료해 정식
  Composite를 계산했다.
- 초기 결과와 해석은
  [Delta-v3 Update Stress KV-cache Canary 결과](delta-v3-update-stress-canary-results.md)에
  기록했다.
- cache OFF/background, latency 반복 측정과 Test 실행은 아직 수행하지 않았다.
