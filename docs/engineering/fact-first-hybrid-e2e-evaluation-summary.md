# Fact-first vs Structured Hybrid: E2E 평가 및 병목 요약

상태: `Scenario 1–10 평가 및 Scenario 6–10 Oracle 진단 완료`

최종 업데이트: 2026-07-30

## 핵심 결론

동결 설정으로 처음 평가한 VehicleMemBench Scenario 6–10에서 Ours
`cloud_fact_patch`의 ESM은 `0.50`, Baseline `cloud_structured_hybrid`는
`0.56`이었다. Ours의 가장 큰 병목은 과거 대화의 정답 Fact가 DB에 생성되지
않는 문제이며, 그다음은 필요한 Fact 검색과 검색된 Fact를 정확한
Tool·argument·복합 호출로 바꾸는 과정이다.

Oracle 진단에서 정답 Fact를 모두 복구하면 ESM이 `0.50→0.80`, 이후
Routing·Retrieval·Binding까지 정답으로 통제하면 `0.80→1.00`이 됐다. 이는
현재 방법의 성능이 아니라 각 단계가 완벽할 때의 상한이다.

## 1. Ours 전체 평가 방법

### 평가 조건

- 데이터: VehicleMemBench 전체 50 scenario·500 task 중 Scenario 6–10의
  50 task
- 비율: 전체 task의 10%
- Agent: `gpt-5.6-terra`
- Memory: `gpt-5.6-luna`
- Embedding: `text-embedding-3-small`, 256 dimensions
- Gold Memory·Gold Tool·Gold argument: 일반 평가에서 Agent에 제공하지 않음
- 한 scenario의 History로 Memory를 한 번 생성하고, 같은 scenario의 10개
  task는 이 읽기 전용 Memory snapshot을 공유
- 각 task query와 Tool 결과는 평가 중 Memory에 다시 저장하지 않음

Ours는 다음 방식으로 Memory를 만든다.

1. History를 최대 32 turns 또는 4,096 source tokens의 batch로 나눈다.
2. Cloud Memory LLM이 향후 차량 동작에 영향을 주는 명시적·영속 Fact를
   추출한다. 이 prompt에는 111개 Tool schema 전체를 넣지 않는다.
3. Fact를 `entity·predicate·value·conditions·evidence` 형태로 정규화한다.
4. 규칙 기반 Validation과 불확실 후보의 batch semantic validation을 거쳐
   SQLite에 `ADD·UPDATE·NOOP` 및 version 형태로 저장한다.
5. 질문 시 BM25·embedding 기반 Top-k Fact를 검색하고, 현재 질문과 역할을
   이용해 Tool schema 및 runtime argument로 late binding한다.
6. Agent가 선택된 Memory와 Tool을 사용해 Vehicle simulator에 Tool call을
   실행한다.

### 재현 명령

아래 명령은 Ours와 Baseline을 동일 조건에서 함께 실행한다. 호환되는
Memory cache가 있으면 재사용하고, 없으면 Cloud Memory를 먼저 생성한다.

```bash
cd ubuntu

export VEHICLEMEMBENCH_ROOT=/path/to/VehicleMemBench
export PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS=2048
export PALMCLAW_MODEL_TIMEOUT_SECONDS=120

palmclaw eval vehicle \
  --benchmark-root "${VEHICLEMEMBENCH_ROOT}" \
  --mode live \
  --profiles cloud_structured_hybrid,cloud_fact_patch \
  --scenario 6 \
  --scenario-limit 5 \
  --task-limit 10 \
  --model gpt-5.6-terra \
  --memory-model gpt-5.6-luna \
  --embedding-model text-embedding-3-small \
  --memory-cache-dir ./evaluation/vehiclemembench-memory
```

평가기는 profile별로 매 task에 새로운 predicted/reference VehicleWorld를
만든다. Agent의 Tool calls와 Gold calls를 각각 실행한 후 VehicleMemBench
scorer로 최종 상태와 Tool 호출을 비교한다. 결과는
`ubuntu/evaluation/vehiclemembench/RUN_ID/`에 저장된다.

주요 구현 위치:

- [CLI와 실행 설정](../../ubuntu/src/palmclaw_ubuntu/cli.py)
- [데이터셋 검증·로딩](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/dataset.py)
- [Memory 생성·검색](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/memory.py)
- [Agent 평가 실행](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/runner.py)
- [공식 scorer 연결](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/scoring.py)

## 2. Baseline: Structured Hybrid 평가 방법

여기서 Baseline은 VehicleMemBench의 공식 방법명이 아니라 PalmClaw에 먼저
구현한 비교 profile `cloud_structured_hybrid`다.

1. 같은 Scenario History를 약 8,000-token batch로 나눈다.
2. Cloud Memory LLM이 대화에서 구조화된 개별 사실을 추출한다.
3. 중요도·중복 판단을 거친 Memory record를 SQLite cache에 저장한다.
4. 질문과 Memory를 BM25와 embedding으로 각각 점수화한 뒤 결합하여
   Top-k를 선택한다.
5. 선택 Memory와 동적으로 발견한 Tool schema를 Agent에 제공한다.
6. Ours와 동일한 Agent, simulator, task, scorer로 평가한다.

Baseline에는 Ours의 Fact patch identity, 별도 semantic candidate review,
Tool-aware late binding 단계가 없다. 공정 비교에서는 두 profile을 같은
명령에 넣어 dataset·모델·task·scorer를 동일하게 유지한다.

## 3. 현재 성능

### 주 결과: 동결 Holdout Scenario 6–10

총 50 task로, 전체 VehicleMemBench 500 task의 10%다.

| 방법 | ESM | State F1 | Tool F1 | Argument exact | Recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structured Hybrid | **0.56** | **0.751** | **0.607** | **0.48** | **0.567** |
| Ours: Fact-first | 0.50 | 0.661 | 0.507 | 0.42 | 0.490 |

Ours는 Scenario 8에서는 `0.80 대 0.40`, Scenario 10에서는
`0.60 대 0.50`으로 앞섰지만, Scenario 6·7·9에서 뒤져 전체 평균은
Baseline보다 낮았다.

### 보조 결과: Scenario 1–10 누적

Scenario 1–5와 동결 Holdout 6–10을 단순 합산한 100 task 결과다. 전체
benchmark의 20%이며, 500-task 전체 평가는 아직 수행하지 않았다.

| 방법 | ESM | State F1 | Tool F1 | Argument exact | Recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structured Hybrid | **0.56** | **0.812** | **0.617** | **0.49** | **0.518** |
| Ours: Fact-first | 0.51 | 0.680 | 0.531 | 0.45 | 0.470 |

지표 의미:

- `ESM`: 예측한 최종 차량 상태가 정답 상태와 완전히 같은 task 비율
- `State F1`: 차량 상태 필드·값의 부분 일치 성능
- `Tool F1`: 호출한 Tool 이름 집합의 정밀도와 재현율
- `Argument exact`: 정답 Tool argument와 완전히 일치한 task 비율
- `Recall@k`: 필요한 Memory가 선택된 Top-k 안에 포함된 비율

주 결과 artifact:

- [Holdout 비교 결과](../../ubuntu/evaluation/vehiclemembench/ffdd8096-c990-46f6-ab9f-0644f61df56a/results.md)
- [Holdout 분석 문서](tool-schema-on-device-memory-holdout-s6-s10-results.md)

## 4. E2E Workflow

```text
Scenario History
  ├─ Structured Hybrid
  │    → 8K-token batch 구조화 요약
  │    → SQLite Memory
  │
  └─ Ours: Fact-first
       → 32-turn/4K-token batch Fact 추출
       → 정규화·Validation·ADD/UPDATE
       → SQLite versioned Fact

Task Query
  → Routing
  → BM25 + Embedding Top-k Retrieval
  → 선택 Memory 조립
  → Tool discovery 및 argument late binding
  → Agent LLM
  → Tool call → VehicleWorld 상태 변경
  → 결과를 Agent에 반환하는 반복 Loop
  → 최종 상태·Tool·argument 평가
```

Agent 입력에는 시스템 지침, 현재 질문, 선택된 Memory, 사용 가능한 Tool
schema와 실행 결과가 포함된다. 전체 원본 History와 선택되지 않은 장기
Memory는 제공하지 않는다.

## 5. Ours의 주요 병목과 Oracle 근거

Oracle은 특정 단계만 검토된 정답으로 교체해 그 단계가 완벽할 때의 상한을
측정한다. 일반 성능과 구분해야 한다.

### 5.1 Fact 생성·저장 누락 — 최우선

50 task 중 필요한 정답 Fact가 active DB에 존재한 것은 `23/50`뿐이었다.
나머지 27 task의 원인은 다음과 같다.

| 원인 | Task 수 | 의미 |
| --- | ---: | --- |
| Extraction 누락 | 20 | 필요한 사실 후보 자체를 LLM이 생성하지 않음 |
| Structure 오류 | 6 | 후보는 있지만 entity·predicate·condition이 잘못됨 |
| Gate 보류 | 1 | 올바른 후보가 review 상태로 남아 active DB에 없음 |

27 task를 복구하는 데 필요한 것은 공유 Fact를 포함해 26개의 고유
record였다. 모든 관련 Fact를 검토된 정답으로 복구한 `Full Memory Oracle`은
ESM을 `0.50→0.80`, State F1을 `0.661→0.869`로 높였다.

따라서 현재 Fact-first부터 Full Pipeline까지 남아 있던 ESM 개선 가능 폭
`0.50` 중 약 60%인 `+0.30`이 Fact 생성·구조화·저장 문제와 연결된다.

### 5.2 Retrieval·Routing 오류

- 원래 DB에 정답 Fact가 있던 23 task 중 일반 Top-k가 선택한 것은
  `21/23`이었다.
- Retrieval Oracle로 빠진 두 record를 강제 Top-k에 넣자 두 task 모두
  실패에서 성공으로 바뀌었다.
- 모든 Fact를 복구한 Full Memory에서도 일반 검색은 `46/50`만 정확한
  reviewed record를 Top-k에 포함했다.
- Gold Tool route만 제공한 O1에서는 Tool selector miss가 `15→0`,
  ESM이 `0.50→0.56`으로 개선됐다.

즉 Fact가 DB에 있어도 질문의 표현과 Tool domain을 잘못 연결하거나 검색
순위에서 밀리면 Agent가 사용할 수 없다. 다만 Retrieval 단독 전체 ESM
증가량은 Cloud Agent 변동의 영향을 받아, 현재는 record-level hit 변화와
해당 task의 복구 여부를 주 근거로 사용한다.

### 5.3 Tool·argument·복합 호출 변환 오류

정답 Fact가 검색된 21 task에 Gold binding을 적용하고 Retrieval까지 누적한
23-task subset에서는 다음과 같이 개선됐다.

- ESM: `19/23→22/23`
- Argument exact: `15/23→19/23`
- 실행 hint rejection: 전체 기준 `177→99`

Full Memory Oracle에서도 관련 record가 검색됐지만 8 task가 실패했다.
대표 원인은 다음과 같다.

- `zone=driver`와 `zone=all` 변환 오류
- radio와 music Tool 혼동
- `switch → set → play` 같은 복합 호출 중 일부 생략
- display·language 관련 Tool의 과잉 호출

Full Pipeline Oracle에서 reviewed Top-k, Gold Tool boundary, Gold argument
binding을 함께 적용하면 ESM은 `0.80→1.00`이 됐다. 이 `+0.20`은 전체
Oracle 개선 가능 폭의 약 40%지만, Retrieval·Routing·Binding·Agent 실행을
함께 통제한 값이므로 복합 호출 변환 하나의 기여도로 해석하면 안 된다.

| Oracle 단계 | ESM | State F1 | Tool F1 | Argument exact |
| --- | ---: | ---: | ---: | ---: |
| 현재 Fact-first | 0.50 | 0.661 | 0.507 | 0.42 |
| 모든 관련 Fact 복구 | 0.80 | 0.869 | 0.763 | 0.64 |
| Retrieval·Tool·Binding까지 통제 | 1.00 | 1.000 | 0.993 | 0.98 |

Full Pipeline의 Tool F1과 Argument exact가 1.0보다 약간 낮은 이유는 한
task에서 이미 닫힌 창을 다시 닫는 Gold 호출을 Agent가 생략했기 때문이다.
최종 상태는 같아 ESM은 성공했다.

Oracle 근거:

- [단계별 Oracle 평가](fact-first-oracle-evaluation-plan.md)
- [O6 Full Oracle 결과](../../ubuntu/evaluation/vehiclemembench-oracle/ede6d720-0f0b-4a65-91c0-247a496d4f7c/results.md)

## 개선 우선순위

1. **Fact 생성 recall 개선:** 필요한 과거 사실을 후보로 빠짐없이 만들고
   올바른 entity·predicate·condition으로 저장한다.
2. **Retrieval 개선:** 저장된 Fact를 질문·인물·상황·Tool 의미에 맞게
   Top-k에 포함한다.
3. **실행 변환 개선:** 여러 Fact와 현재 질문을 결합해 정확한
   Tool·enum·argument·복합 호출 계획을 만든다.

500-task 전체 확대 평가는 이 세 단계를 개선한 뒤 같은 Holdout에서 회귀를
확인하고 수행한다.
