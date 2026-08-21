# VehicleMemBench V2 UPDATE/NO_OP 품질 감사 구현 계획

## 목적

동일한 frozen Stage 2를 사용하는 `post_hoc`, `hybrid`, `native_turnwise`의
turn-wise label이 학습 데이터로 신뢰 가능한지 비교한다. Quiz 성능과 별도로
**필요한 UPDATE를 정확한 시점에 만들었는지**, **불필요한 UPDATE를 만들지
않았는지**, **UPDATE 결과 memory가 증거를 충실히 보존하는지**를 측정한다.

Stage 2의 `preference_updates`는 예상 target을 알려주는 oracle로 사용하되, 그 값
자체를 대화 증거로 간주하지 않는다. Judge는 반드시 해당 event의 causal dialogue
prefix에서 근거를 찾아야 한다.

## 1. 공통 감사 입력 만들기

세 artifact schema를 다음 공통 `TurnAuditRecord`로 변환한다.

```text
method, scenario, event_id, turn_id, global/event turn index
dialogue_prefix, current_turn
before_memory, candidate decision(UPDATE/NO_OP), Patch operations, after_memory
source_update_indexes, Stage 2 expected updates
source/artifact/memory SHA-256
```

- `post_hoc`와 `hybrid`: `V2HybridArtifact.event_checkpoints[].turn_labels`
- `native_turnwise`: `V2NativeScenarioArtifact.event_artifacts[].turn_checkpoints`
- 원본 artifact는 수정하지 않고 평가용 record만 별도 저장한다.
- 동일 scenario의 세 방법이 같은 `source_stage2_sha256`를 가질 때만 paired 비교한다.

## 2. 결정론적 전수 검사

LLM 호출 전에 모든 turn을 검사한다.

- schema 및 artifact/checkpoint hash 검증
- `before_memory → Patch → after_memory`의 atomic 재현
- NO_OP에서 memory hash가 바뀌지 않았는지 확인
- UPDATE의 evidence turn/quote가 현재 causal prefix 안에 실제 존재하는지 확인
- `source_update_indexes`의 범위, 중복, event 간 중복 commit 검사
- 미래 turn을 근거로 사용한 label은 즉시 `FAIL` 처리

결과는 `PASS/FAIL + error_codes`로 기록하며, 실패 record는 LLM Judge 결과와
무관하게 hard failure로 집계한다.

## 3. Dual Judge → SOL → Human

같은 record를 독립적인 두 Judge가 먼저 검사한다.

- Judge A: `gpt-5.6-luna`
- Judge B: `gpt-5.6-terra`
- 두 호출은 동일한 versioned input을 사용하되 response나 reasoning을 공유하지 않는다.
- 입력: `dialogue prefix + 직전 memory + candidate label/Patch + UPDATE 후 memory +
  expected update fields`
- 두 Judge 모두 동일한 structured output schema를 사용한다.

Luna 두 번처럼 같은 모델의 반복 호출만 사용하지 않고 Luna와 Terra를 조합하여
동일 모델의 공통 오류가 합의로 오인될 가능성을 낮춘다.

### 검사 범위

1. 생성된 모든 UPDATE
2. Stage 2에 expected update가 존재하는 모든 event
3. expected update가 없는 NO_OP event는 method/scenario/reason code별 고정 seed로
   10% 층화 표본
4. NO_OP 표본에서 오류가 나오면 해당 stratum 전체로 자동 확대

### Judge 판정 schema

```text
verdict: PASS | PARTIAL | FAIL
expected_updates[]:
  FOUND | MISSING | UNSUPPORTED_BY_DIALOGUE
  | WRONG_VALUE | WRONG_SUBJECT | WRONG_CONDITION
candidate_update: VALID | SPURIOUS | DUPLICATE | PREMATURE | LATE
earliest_evidence_turn, labeled_turn, boundary_offset
memory_result: FAITHFUL | INFORMATION_LOSS | CONTRADICTED | MALFORMED
evidence_turn_ids[], short_reason
```

Judge가 반환한 turn ID와 인용문은 입력 prefix에 대해 결정론적으로 grounding한다.
존재하지 않는 증거, 순서가 다른 expected update, 잘못된 hash는 해당 Judge만
bounded retry한다.

### 합의 기준

두 결과를 deterministic normalize한 뒤 다음 material fields의 **구조화된 의미가
같을 때** 합의로 채택한다.

```text
verdict
expected update별 status
candidate_update
earliest_evidence_turn과 boundary_offset
memory_result
```

다음 차이는 합의 여부에서 무시한다.

- `short_reason`의 문장 표현
- 같은 turn을 가리키는 서로 다른 유효 evidence quote
- casing, 공백, field 순서, 명백한 enum·단위 alias

반대로 `PASS/FAIL`만 같고 오류 field, normalized value/subject/condition 또는 evidence
boundary가 다르면 의미가 다른 것이므로 SOL로 보낸다. 의미가 같은지를 자유 텍스트
유사도로 추정하지 않고 structured field와 grounding 결과로 판정한다.

### SOL adjudication

두 Judge가 불일치하거나 한쪽 결과가 grounding/retry에 실패하면
`gpt-5.6-sol`에 다음을 제공한다.

```text
원래 audit input
Judge A/B의 구조화된 결과와 grounded evidence
두 결과의 material diff
결정론적 검사 결과
```

SOL은 `RESOLVE`와 최종 audit 판정 전체를 반환하거나, 근거가 부족하거나 복수 해석이
가능하면 `DEFER_HUMAN`을 반환한다. 숫자형 confidence만으로 자동 채택하지 않는다.
SOL의 evidence도 동일하게 grounding하며 실패 시 Human Review로 보낸다.

### Human Review

SOL이 `DEFER_HUMAN`을 반환하거나 SOL 응답이 bounded retry 후에도 schema/grounding을
통과하지 못한 경우만 기존 V2 Human Review 웹에서 검토한다. 결정론적 hard failure는
명시적 `FAIL`로 기록하고 Human queue에 넣지 않는다. 기존 SQLite queue의
상태·revision·hash 검증 패턴은 재사용하되, Gold 생성 correction과 섞이지 않도록
별도 `UPDATE_AUDIT` request type과 queue DB를 사용한다.

Human은 원본 record와 A/B/SOL 판정을 보고 최종 audit schema를 제출한다. 이 감사는
read-only이므로 원본 memory artifact를 자동 수정하지 않는다. 수정이 필요하다고
판정된 record만 별도 regeneration 대상 목록에 넣는다.

기존 V2 pipeline에서는 candidate 생성 로직이 아니라 다음 기반만 재사용한다.

- consensus 결과 정규화·material diff 패턴
- model별 checkpoint, retry, usage 및 hash signature 패턴
- `human_review.py`의 queue lifecycle·revision 검증
- `serve_vehiclemembench_v2_reviews.py`의 목록·상세·제출 UI

감사 결과는 Gold 생성 queue와 별도 DB에 저장해 기존 실행 상태에 영향을 주지 않는다.

## 4. 지표와 결과물

방법·시나리오별로 다음을 집계한다.

- expected-update recall 및 field-level 정확도
- generated-update precision과 spurious/duplicate rate
- NO_OP accuracy와 sampled error rate
- evidence-boundary offset의 평균·분포·premature/late 비율
- UPDATE 후 memory faithful/information-loss/contradiction 비율
- 결정론적 hard-gate 통과율
- A/B 합의율, SOL escalation율, Human escalation율
- 모델별 input/cached/output tokens, 호출 수, retry, latency, 예상 비용

결과는 아래에 저장한다.

```text
vehiclemembench-v2-three-way-evaluation/update-audit/
  manifest.json
  records/{method}/sXX/*.json
  judge/{method}/sXX/{luna,terra,sol,consensus}.json
  update-audit-review-queue.sqlite
  review-queue.json
  update-audit-summary.json
```

`PARTIAL/FAIL` 자체는 품질 측정 결과이므로 자동으로 Human에게 보내지 않는다. 두
Judge의 불일치는 먼저 SOL이 판정하며, SOL도 해결하지 못한 경우만 Human Review
queue에 넣는다. 최종 표에서는 Quiz/Answerability 결과와 합치되 단일 점수로 섞지
않고 별도 열로 비교한다.

## 구현 회차

### 1회차 — Adapter와 결정론적 검사

- 공통 `TurnAuditRecord` adapter 구현
- hash-chain, Patch, NO_OP, causal evidence 검사 구현
- readiness/dry-run manifest에 검사 대상 수와 예상 Judge 호출 수 출력
- 세 schema fixture 기반 단위 테스트

### 2회차 — Dual Judge와 consensus

- Luna/Terra 공통 structured Judge prompt/schema와 evidence grounding 구현
- UPDATE 전수, expected-update event 전수, NO_OP 10% 표본 선택 구현
- material-field normalize와 strict consensus 구현
- Judge별 bounded retry, atomic checkpoint, cache signature 구현

### 3회차 — SOL·Human 연결과 S1 smoke test

- [완료] disagreement용 SOL adjudicator와 `DEFER_HUMAN` 구현
- [완료] 별도 UPDATE audit review queue와 기존 웹의 audit 화면 구현
- [완료] S1 세 방법 end-to-end smoke test
- [완료] NO_OP 오류 stratum 자동 확대 검증

S1 smoke에서는 최초 51 event를 검사했고, 표본 오류가 나온
`NO_NEW_VEHICLE_FACT` 층을 `hybrid`와 `post_hoc`에서 자동 전수 확대한 뒤 총
177 event를 완료했다. Luna/Terra 354회와 SOL 21회를 사용했으며 provider 실패와
Human defer는 모두 0건이었다. 최종 PASS는 `post_hoc 76/80`, `hybrid 77/80`,
`native_turnwise 13/17`이었고, 이는 방법론 순위를 확정하는 본 실험이 아니라 감사
파이프라인의 end-to-end 동작을 확인한 smoke 결과다.

### 4회차 — 전체 실행·집계·대시보드

- [완료] 준비된 artifact부터 병렬 실행하고 Native S2–S5 완료 후 부족분만 재개
- [완료] paired scenario 및 method별 지표·모델별 비용 집계
- [완료] 기존 V2 review 웹에 UPDATE/NO_OP 감사 열과 상세 evidence 표시
- [완료] Quiz 성능·Answerability·UPDATE 감사의 교차 오류표 추가
- [완료] 전체 실행 checkpoint와 결과 재현 명령 검증

S1–S5 전체 실행은 Agent 1,200 tasks, Answerability 600 quizzes, UPDATE 감사
512 events에서 provider failure 0건으로 완료했다. Post-hoc S3의 한 event는 SOL이
`DEFER_HUMAN`으로 판정하여 별도 UPDATE audit queue에서 검토 대기 중이다. 따라서
자동 집계는 `PROVISIONAL`로 표시하며, Human 제출 후 동일 summarizer를 다시 실행한다.

## 완료 조건

- 세 방법의 모든 turn이 결정론적 검사를 통과하거나 명시적 실패로 기록됨
- 모든 generated UPDATE와 expected update event가 Judge 결과를 가짐
- NO_OP 표본이 고정 seed로 재현되고 오류 stratum이 자동 확대됨
- A/B material consensus와 SOL escalation이 deterministic하게 재현됨
- SOL `DEFER_HUMAN`을 웹에서 검토할 수 있음
- source/hash/prompt/schema/model/judge role이 checkpoint signature에 포함됨
- 원본 생성 artifact는 변경되지 않음
- S1–S5 paired 결과와 비용을 한 표로 재생성할 수 있음
