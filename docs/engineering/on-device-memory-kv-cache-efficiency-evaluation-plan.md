# On-device Memory KV-cache 효율 비교 계획

## 상태와 목적 (2026-09-02)

**상태: Runner ready; append checkpoint training 중.** Summary, Patch, Delta-v3-append의
정확도와 별개로 turn-wise latency와 연산 비용을 비교한다. `NO_OP:UPDATE = 5:1`
Validation 540턴 고정 manifest, batch-1 HF prefix-cache runner, append-only cache epoch,
background prefill 계측 및 세 관점 비교 report 구현을 완료했다.

기존 HF 평가는 `generate(use_cache=True)`로 한 요청 안의 decode cache만 사용한다.
전용 runner는 시나리오별 KV cache를 다음 turn까지 유지하며, 세 방법 모두에 동일한
cache 정책을 적용한다.

## 구현 계획

1. 각 시나리오가 자신의 `past_key_values`와 이전 prompt token을 보유한다.
2. 다음 prompt와 이전 prompt의 token LCP(longest common prefix)를 구하고, cache를
   LCP까지 crop한 뒤 나머지 suffix만 prefill한다.
3. Summary/Patch/기존 Delta-v3는 생성 output을 다음 prompt에서 제거한다.
   Delta-v3-append는 생성된 assistant token까지 cache identity와 transcript에 보존한다.
4. 우선 `batch_size=1`로 구현·검증하고, 이후 시나리오별 cache를 유지하는 동적 batch를
   추가한다.
5. cache OFF/ON에서 greedy output과 state transition이 동일한지 자동 검증한다.
6. `--background-prefill`은 UPDATE 직후 안정적인 memory prefix KV를 만들고, 이 비용을
   다음 turn foreground TTFT와 분리하되 전체 token/compute cost에는 포함한다.

Delta-v3는 UPDATE 1~4회 동안 `base_summary + 기존 pending_updates`를 재사용한다.
5번째 UPDATE compaction 뒤에는 변경된 `base_summary`를 다시 prefill한다. Patch와
Summary도 같은 LCP 규칙을 사용하되, memory 변경 위치에 따라 실제 재사용량이 결정된다.

## 실험 계획

- 모델 크기별로 `350M`, `1B`, `2B`를 분리 비교한다.
- 동일 크기 안에서는 같은 base model, precision, device, context limit, decoding 설정을
  사용한다.
- 각 `Summary / Patch / Delta-v3`에 대해 `cache OFF / cache ON`을 모두 실행한다.
- 각 시나리오는 가능하면 10회 이상 UPDATE를 포함해 Delta-v3 compaction 주기를 두 번
  이상 관찰한다.
- warm-up 이후 단일 시나리오 순차 실행을 기본 latency 측정으로 사용한다.

두 실행 모드를 함께 보고한다.

- **Controlled replay:** gold state를 사용해 방법 자체의 prompt/cache 비용을 비교한다.
- **Predicted closed loop:** 모델이 만든 state를 이어서 실제 운영 비용을 측정한다.

## 측정과 집계

매 turn마다 다음 값을 기록한다.

- logical prompt tokens, reused prefix tokens, actually evaluated prefill tokens
- decode tokens, cache reuse ratio
- TTFT, prefill/decode/end-to-end latency, cache 관리 및 executor overhead
- peak KV-cache memory
- Delta-v3 pending depth와 compaction 발생 여부

결과는 `mean / p50 / p95`와 표본 수를 포함해 세 관점으로 집계한다.

1. **전체 turn 평균:** 5:1 workload의 실제 평균 비용
2. **Gold UPDATE 평균:** 세 방법에 공통인 중요 turn의 통제 비교
3. **Predicted UPDATE 평균:** false/invalid/apply failure를 포함한 실제 고비용 경로

로컬 HF의 cost는 우선 `evaluated prefill tokens + decode tokens`, latency, peak memory로
정의한다. 이후 on-device 단계에서 energy per turn과 thermal 영향을 추가한다. Delta-v3의
compaction 비용은 compaction 직후 cache rebuild와 5-UPDATE 주기당 상각 비용으로 함께
표시한다.

Background prefill 실행에서는 `foreground prefill + background prefill + decode`를 총
model cost로 사용한다. 따라서 TTFT 개선과 실제 연산량 증가/감소를 혼동하지 않는다.

## 실행

세 checkpoint가 결정된 뒤 GPU, Summary, Patch, Delta-v3-append checkpoint 순서로 실행한다.

```bash
bash memory_training/scripts/run_granite4_1b_three_method_kv_benchmark.sh \
  0 /path/to/summary/checkpoint /path/to/patch/checkpoint \
  /path/to/delta-v3-append/checkpoint v1
```

각 방법을 cache OFF / ON / ON+background로 실행하고 controlled/predicted 각각에 대해
`all_turns`, `gold_update`, `predicted_update` 비교 JSON과 Markdown을 만든다. 30 token/s
열은 on-device 예상치일 뿐이며 HF 실측 latency와 구분된다.

고정 5:1 데이터는 원본 turn 일부를 생략했으므로 성능 비종속 비용 실험에서는 이를
합성 연속 workload로 취급한다. 정확도/상태 검증에는 생략 없는 full trajectory를 쓴다.

## 완료 조건

- cache OFF/ON 결과 동일성 검증 통과
- 세 방법과 세 모델 크기의 동일 조건 측정 완료
- 전체 turn, Gold UPDATE, Predicted UPDATE 결과 및 Delta-v3 compaction-cycle 결과 산출
