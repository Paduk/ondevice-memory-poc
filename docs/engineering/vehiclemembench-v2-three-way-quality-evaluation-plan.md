# VehicleMemBench V2 3-way 품질 비교 계획

## 목적

동일한 frozen Stage 2를 사용하는 `post_hoc`, `hybrid`, `native_turnwise`의
5개 paired scenario에서 **실제 Quiz 성능**, **Quiz answerability**, **turn-wise
memory label 품질**, **생성 비용**을 분리해 비교한다. 이 평가는 5개 scenario
scale gate이며 통계적 우위의 최종 근거로 사용하지 않는다.

각 method/scenario는 Turn Quiz 30개와 Final Quiz 10개, 총 40개를 가져야 한다.
실행 전에 15개 artifact의 `source_stage2_sha256`, Quiz 수, simulator/hash-chain
통과 여부를 manifest로 고정한다. 같은 scenario의 세 source hash가 다르면 평가하지
않는다.

## 1. 전체 Quiz 실행 평가

각 Quiz는 생성 method의 **해당 cutoff memory snapshot**만 Agent에 제공한다.
Turn Quiz는 그 turn의 memory, Final Quiz는 final memory를 사용하며 대화 원문은
제공하지 않는다. 입력은 `memory + query + 동일 Tool schema`로 고정한다.

- 기본 Agent: `gpt-5.6-luna`, 설정과 prompt version 고정
- 600문항(`3 × 5 × 40`)을 2회 fresh 실행
- 지표: ESM, Tool F1, Argument Exact, 호출 tokens, latency
- 집계: method/scenario, Turn 30/Final 10, immediate/delayed/composite,
  reasoning type별
- method별 query가 다르므로 이 결과는 memory뿐 아니라 **end-to-end 데이터 생성
  품질**을 함께 측정한 값으로 명시한다.

## 2. Quiz answerability 감사

동일 600문항에 대해 Judge가 `query + gold calls + Tool schema + cutoff memory`만
보고 정답 복원 가능성을 판정한다. 기존 `qa_coverage.py`의 stable line ID와
결정론적 evidence grounding을 일반화한다. 동일 memory hash를 사용하는 Quiz는
한 호출로 묶어 memory 중복 입력을 줄인다.

판정은 다음 네 가지다.

- `FULL_SUPPORT`: 모든 gold Tool과 Argument를 유일하게 복원 가능
- `PARTIAL_SUPPORT`: 일부 필드만 복원 가능하거나 조건·주체가 모호함
- `UNSUPPORTED`: 핵심 값 또는 Tool을 복원할 수 없음
- `CONTRADICTED`: memory가 gold와 명시적으로 충돌

각 Tool call과 Argument마다 `SUPPORTED_BY_MEMORY`, `SUPPORTED_BY_QUERY`,
`SUPPORTED_BY_BOTH`, `MISSING`, `CONTRADICTED`를 기록하고 다음 값을 계산한다.

```text
support_coverage = supported gold fields / total gold fields
```

Judge는 evidence line ID를 반드시 반환한다. 존재하지 않는 ID, gold만 근거로 한
판정, memory에 없는 인용은 결정론적으로 거부한다. 기본 Judge는
`gpt-5.6-terra`; `PARTIAL/UNSUPPORTED/CONTRADICTED`와 무작위 `FULL_SUPPORT` 10%는
Human Review 대상으로 보낸다.

Quiz 실행과 answerability를 교차 집계한다.

| Answerability | Agent 결과 | 해석 |
| --- | --- | --- |
| FULL_SUPPORT | 정답 | 정상 |
| FULL_SUPPORT | 오답 | Agent 추론 실패 |
| PARTIAL/UNSUPPORTED | 오답 | Memory 또는 Quiz 생성 실패 |
| PARTIAL/UNSUPPORTED | 정답 | 추측·query 단서 의존 가능성 |
| CONTRADICTED | 무관 | 데이터 결함 우선 수정 |

## 3. UPDATE 중심 turn-wise memory 감사

모든 turn을 LLM으로 검사하지 않는다. Memory는 UPDATE에서만 변하므로 다음 범위를
검사하면 모든 고유 memory state와 누락 위험을 함께 볼 수 있다.

1. 전체 turn에 schema, memory hash-chain, Patch atomicity, causal evidence cutoff를
   결정론적으로 검사
2. 생성된 모든 UPDATE와 UPDATE 직후 memory snapshot을 Judge로 전수 검사
3. Stage 2에서 preference update가 예정된 모든 event를 event 단위로 검사하여
   누락, 중복, spurious update와 earliest evidence boundary를 확인
4. 모든 Quiz cutoff memory를 2절의 answerability 감사로 전수 검사
5. preference update가 없는 event는 method/scenario/reason code별 고정 seed로 10%를
   층화 표본 검사

Event Judge 출력은 `PASS/PARTIAL/FAIL`, expected update별
`FOUND/MISSING/WRONG_VALUE/WRONG_SUBJECT/WRONG_CONDITION`, 생성 UPDATE의 spurious
여부, earliest valid evidence turn과 label turn 차이를 포함한다. 표본 NO_OP에서
문제가 발견되면 해당 stratum 전체로 검사를 확대한다.

## 4. 최종 비교표

단일 종합점수로 즉시 합치지 않고 다음 Pareto 표를 사용한다.

- Quiz: ESM, Tool F1, Argument Exact
- Answerability: FULL/PARTIAL/UNSUPPORTED/CONTRADICTED 비율, 평균 field coverage
- Memory: expected-update recall, spurious-update rate, evidence-boundary offset,
  Human Review pass rate
- Efficiency: input/cached/output tokens, 호출 수, retry, latency, 비용

시나리오별 paired 결과와 5개 macro average를 함께 표시한다. Hard gate는 simulator,
hash-chain, causal cutoff 100%와 `CONTRADICTED=0`이다. 품질 차이가 2%p 이내면 더
낮은 token·latency 방법을 우선한다.

## 구현 순서

1. **Artifact adapter와 readiness manifest**: 세 schema에서 Quiz cutoff memory를
   hash로 복원하고 15개 source hash를 검증한다.
2. **Quiz runner**: 40문항 × 2회 실행, raw response·usage·latency·ESM/F1 저장.
3. **Answerability/UPDATE Judge**: structured schema, evidence grounding,
   snapshot batching, resumable checkpoint 구현.
4. **Aggregation/dashboard**: paired 표, 실패 교차표, Human Review queue와 기존 웹
   화면 연결.

모든 cache key에는 method, scenario, source/quiz/memory SHA-256, model,
prompt/schema version을 포함한다. 입력 artifact는 수정하지 않고 평가 결과는 별도
`vehiclemembench-v2-three-way-evaluation/` 아래에 저장한다.
