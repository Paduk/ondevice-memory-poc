# On-device Combined 프로필 구현 계획

## 목표

현재 `Patch` baseline은 그대로 두고, 기존 Turn-wise Combined를 재현하는 다섯 번째
프로필 `combined`를 추가한다.

```text
Temporal-aware Patch
+ deterministic apply
+ soft-30 periodic compaction
```

`soft-30`의 정확한 기존 설정은 **누적 ADD 64회 또는 memory 1,000 tokens 도달 시**
전체 memory rewrite를 호출하고, 안전할 때 **기존 대비 30% 축약(target ratio 0.70)** 을
시도하는 것이다. 목표 미달을 이유로 사실을 억지로 삭제하지 않으며, 결과가 더 짧지
않으면 Patch 적용 직후 memory를 유지한다.

## 선행 감사 결과

현재 Hybrid S1–S100은 시나리오당 최대 ADD 13회, 최대 memory 527 tokens다. 따라서
기존 임계값에서는 compaction이 **0회** 발동한다. S91–S100 본 평가는 Temporal Patch
효과는 볼 수 있지만 soft-30 효과를 검증하지 못하므로, 실제 과거 Combined trace 또는
별도 long-memory stress set으로 compaction을 따로 검증한다.

## 구현 순서

### 1. Combined 학습 view

- `combined.jsonl`: 기존 Patch 입력에 대해 UPDATE operation을
  `op/target/content/identity_key/temporal_action/temporal_cue`로 확장한다.
- `identity_key`와 명시적인 condition/state 변화는 Stage2 및 Hybrid provenance에서
  결정론적으로 연결한다.
- `temporary_override/end_temporary`는 원문에 명시적 cue가 있을 때만 부여하고,
  불확실한 시간 관계는 만들지 않는다.
- `combined_compaction.jsonl`: 기존 Combined 실행에서 실제 발동한 compaction의
  pre/post memory를 추출한다. 부족하면 학습 split 내부의 long-memory replay만 별도
  생성하며 Test trajectory는 사용하지 않는다.
- export 시 schema, evidence, before/after hash와 temporal class 분포를 manifest에 남긴다.

### 2. Method와 executor

- `memory_training/methods/combined.py`에 `CombinedMethod`를 추가한다.
- Turn 출력은 `NO_OP` 또는 extended temporal operations만 허용한다.
- runtime은 temporal 규칙을 검증한 뒤 `op/target/content`를 원자적으로 적용한다.
- 상태에 `adds_since_compaction`을 보존하고 UPDATE 뒤에만 다음 조건을 검사한다.
  - 누적 ADD `>=64`
  - memory가 처음으로 `1,000 tokens`를 넘음
  - memory character limit 초과
- 발동 시 같은 모델에 명시적인 `COMPACTION` task를 주어 전체 memory를 생성한다.
  더 짧고 유효한 결과만 채택하고, 그렇지 않으면 patched memory를 유지한다.

### 3. 학습·평가 연결

- method registry/CLI를 `summary|patch|delta|summary_reason|combined`로 확장한다.
- Combined SFT에는 Turn Patch와 compaction 표본을 `task_type`으로 구별해 함께 넣는다.
- 기존 UPDATE:NO_OP sampler는 Turn Patch에만 적용하고 compaction 표본은 별도 비율로
  고정한다.
- HF Validation, checkpoint resume, dashboard, Ollama export/Test가 `combined`를 같은
  run contract로 읽게 한다.

### 4. 검증과 첫 실행

1. 단위 테스트: durable/current/temporary/conditional cycle, invalid cue, atomic apply,
   NO_OP counter 유지, 64-ADD/1000-token trigger, 축약 실패 시 원본 유지.
2. 전체 JSONL replay: Patch 적용 및 hash-chain 100% 일치.
3. compaction stress replay: trigger/채택/보존 정확도와 fact retention 측정.
4. Qwen3.5 4B 짧은 canary 후 기존 Patch와 동일 설정으로 학습.
5. S21 Train diagnostic와 S91·S92 Test에서 closed-loop/teacher-forced 비교 후
   S91–S100으로 확장한다.

## 비교 지표와 예상

- 품질: Update F1, temporal-action F1, schema/apply 성공률, final State F1, Quiz ESM/Arg.
- 효율: Turn input/output tokens, compaction 횟수·tokens·latency, 최종 memory 길이,
  Quiz prefill/latency.
- 예상: pure Patch보다 operation 출력은 조금 길어진다. 현재 짧은 S1–S100에서는
  compaction 이득이 거의 없고 Temporal 처리 차이가 주효하다. 긴 trajectory에서는
  간헐적인 rewrite 비용을 내는 대신 memory와 Quiz prefill을 줄이는 것이 목표다.

## 완료 기준

- 기존 네 방법의 데이터·동작·결과를 변경하지 않는다.
- Combined temporal schema와 deterministic replay가 모두 통과한다.
- 본 Test와 compaction stress 결과를 분리 보고해, compaction이 발동하지 않은 결과를
  Combined 전체 효과로 오해하지 않게 한다.
