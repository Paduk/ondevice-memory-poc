# VehicleMemBench V2 Consensus Gold 실행 계획

상태: `S1 완료, S2–S10 실행 중, Human Review Web·자동 재개 구현 완료`

요약 문서: [V2_plan.md](../assets/V2_plan.md)

## 1. 목표와 고정 원칙

정답 ledger 없이 각 scenario의 첫 turn부터 adjudicated Gold Memory를 만든다.

```text
확정된 Gold Memory M(t-1) + Current Turn t
→ Luna Combined temporal patch 독립 생성 3회
→ Semantic Consensus
→ Evidence / Replay hard-rule 검증
→ 필요 시 Terra → SOL → Human
→ 단일 Gold Memory M(t) + Combined Update Record + 한 문장 Reason
→ 동일 M(t)를 Summary full-rewrite target으로 export
```

고정 원칙:

- Luna 입력 형식과 history 범위는 V1과 동일하게 유지한다.
- Luna에는 V2 출력 schema instruction만 추가한다.
- QA `gold_memory`, query, gold answer, 기존 trace와 미래 turn은 생성·검증에
  사용하지 않는다.
- `M(t)`가 확정되기 전에는 같은 scenario의 `t+1`을 시작하지 않는다.
- unresolved turn은 NO_OP으로 대체하지 않고 해당 scenario를 중단한다.
- scenario끼리는 독립적으로 병렬 실행할 수 있다.
- Gold trajectory는 방법별로 나누지 않고 하나만 만든다.
- canonical update 형식은 Combined temporal patch로 고정한다.
- 같은 turn의 A/B/C 후보는 모두 Combined 형식으로 생성한다.
- 확정된 동일 `M(t)`에서 Combined와 Summary 학습 view를 export한다.
- 공식 QA는 scenario 생성 완료 후 외부 평가에만 사용하며 label을 수정하는
  근거로 되먹이지 않는다.

이 문서에서 Gold는 multi-sample과 escalation을 통해 확정한
**adjudicated Gold**를 의미한다.

## 2. 현재 구현된 기반

| 영역 | 현재 상태 | V2에서의 사용 |
| --- | --- | --- |
| V1 Turn-wise 입력 | `Current Memory + New Conversation Turn` 조립과 Summary/Patch Tool 호출 구현 | 동일 입력을 보장하는 공통 builder로 분리 |
| Turn 처리 | history entry 파싱, 시간순 정렬, turn별 batch 구현 | scenario source turn 생성에 재사용 |
| Memory 적용 | Summary full rewrite, Combined exact temporal Patch·periodic compaction 구현 | 방식별 hard rule에 재사용 |
| Model 응답 검증 | strict schema, semantic retry, usage 수집 패턴 구현 | 새 Candidate/Terra/SOL schema에 맞게 확장 |
| Checkpoint | SQLite transaction, resume, memory hash-chain 패턴 구현 | 새 Gold checkpoint 설계에 참고 |
| Controller·artifact | scenario별 실행, marker, manifest, JSONL writer 패턴 구현 | 새 output root와 schema로 재작성 |
| 테스트 환경 | 관련 infrastructure 테스트 42개 통과 | 새 pipeline 회귀 테스트의 기반 |
| V2 core | V1-equivalent input, Combined/compaction schema, replay gate, 3-way consensus, Terra/SOL, checkpoint와 S1 runner 구현 | cloud canary에 사용 |

새 pipeline은 turn별 3-candidate 생성, 제한 retry, consensus, Terra/SOL escalation,
atomic commit/resume, 30-add compaction consensus와 완료 후 QA coverage까지 연결되었다.
Human Review는 중앙 SQLite queue, local Web UI와 tmux 자동 재개 controller로 구현했다.
audit/학습 export는 아직 구현 전이다.
기존 label, trace, critic 결과와 이전 smoke artifact는 새 S1의 입력·정답·완료
기준으로 사용하지 않는다.

## 3. 구현해야 할 것

### 3.1 Versioned schema

다음 객체를 정의한다.

- `GoldGenerationInput`: scenario, turn, `M(t-1)`, current utterance, source hash
- `CombinedTurnCandidate`: sample ID, decision, temporal patch operations,
  system이 계산한 `next_memory`, reason, evidence spans
- `CompactionCandidate`: sample ID, post-patch memory를 rewrite한 compacted memory
- `ConsensusResult`: 정규화된 비교 필드, agreement와 차이
- `GateResult`: `PASS / FAIL`, 실패한 hard rule과 error code
- `GoldTurnLabel`: 확정 memory, Combined update record, reason, evidence,
  validation path

모든 record에는 dataset, prompt, schema, model version, input hash,
before/after memory hash를 저장한다. Combined operations는 실행 가능한 배열이고
NO_OP은 빈 배열로 표현한다. Summary는 별도 candidate가 아니라 확정된 `M(t)`를
complete rewrite target으로 사용하는 export view다.

### 3.2 V1-equivalent 입력 builder

V1 provider 내부의 turn-wise 입력 조립을 공통 함수로 분리한다.

- 같은 memory와 turn이면 V1과 V2의 user input text가 동일해야 함
- V2의 차이는 instruction과 출력 Tool schema뿐이어야 함
- generator request에 QA, 기존 trace, future history가 들어가면 테스트 실패
- candidate 세 개가 같은 input hash를 사용해야 함
- 입력 hash와 redaction 결과를 candidate별로 저장

### 3.3 Luna 3-sample Combined generator

동일한 `M(t-1)`과 current turn으로 세 후보를 독립 생성한다.

- 호출을 `sample_id=A/B/C`로 구분
- 세 호출의 model ID, prompt, instruction, schema, reasoning effort와 sampling
  설정은 동일한 version으로 고정
- Luna reasoning effort는 `medium`으로 고정
- `previous_response_id`나 후보 간 응답을 공유하지 않고 별도 request로 호출
- 현재 Responses API에는 `seed`를 전제하지 않는다. 서로 다른 seed보다 동일 설정의
  독립 호출 3회를 기본으로 사용
- prompt나 temperature를 후보별로 다르게 주지 않음. prompt variant는 Gold 생성과
  분리된 robustness 실험에서만 사용
- request parameter, response ID와 input/prompt hash를 후보별로 저장
- 후보 하나가 실패하면 해당 후보만 retry하고 완료 후보는 재사용
- Luna가 minimal temporal patch operations를 반환하고 시스템이 이를
  `M(t-1)`에 적용해 `next_memory`를 계산함
- 30-add compaction이 발생하면 확정 post-patch memory로 Luna rewrite 후보 3개를
  독립 생성함. 3/3 일치는 자동 채택하고 불일치는 Terra 없이 SOL이 선택·수정함
- Summary용 별도 Luna 호출은 하지 않고 최종 `M(t)`를 full-rewrite target으로 export
- `operation_reason`은 current-turn evidence에 근거한 한 문장만 허용
- 세 후보가 모두 준비되기 전에는 consensus와 다음 turn을 실행하지 않음

### 3.4 Semantic consensus

초기 구현은 별도 LLM consensus judge를 두지 않고 구조화 필드를 비교한다.

```text
operation + subject + attribute + normalized value/unit
+ condition/temporal scope + supersedes/delete target
+ evidence support + resulting canonical memory state
```

- casing, whitespace, field order와 명백한 enum/unit alias만 deterministic normalize
- 동일한 patch 효과에서 내부 추적용 `identity_key` 차이는 무시하되, operation,
  memory text/value, target, temporal action/cue 차이는 유지
- patch 표현이 달라도 deterministic apply 후 `next_memory` hash와
  구조화된 의미 필드가 같으면 동등하게 처리
- compaction 후보는 정규화된 memory가 같을 때만 동등하게 처리하고, 그 이상으로
  문장이 다르면 rule로 의미 동등성을 추정하지 않고 SOL로 바로 보냄
- **3/3 의미 일치:** 대표 후보를 두 gate로 보냄
- **2/3 일치 + 나머지 1개가 형식 차이뿐:** normalize 후 3/3으로 간주해 두 gate로 보냄
- **2/3 일치 + 나머지 1개가 의미적으로 다름:** 다수 후보를 preferred로 기록하되
  auto-accept하지 않고 Terra로 보냄
- **세 후보 모두 의미적으로 다름:** Terra로 보냄
- schema 실패 또는 Combined deterministic apply 실패 후보는 해당 후보만 제한적으로
  재생성하고, 유효 후보 간 불일치가 남으면 Terra로 보냄
- vote count와 정확한 material diff를 Terra 입력과 audit에 전달
- consensus만으로 Gold를 확정하지 않고 두 gate를 반드시 통과

### 3.5 두 rule-based validation gate

두 gate는 Gold의 의미를 판단하지 않고 **명백한 invalid output만 차단**한다. 코드로
확실하게 판정할 수 없는 조건은 FAIL로 만들지 않는다.

**Gate 1 — Evidence Integrity**

- UPDATE evidence quote가 허용된 current-turn user 원문에 exact match함
- evidence의 message ID와 quote/offset이 실제 source와 일치함
- 과거·미래 turn, QA, assistant/tool message를 evidence로 참조하지 않음
- UPDATE의 필수 evidence field가 누락되지 않음

**Gate 2 — Combined Patch/Compaction Integrity**

- patch schema, type, operation enum과 필수 field가 유효함
- patch를 `M(t-1)`에 atomic·deterministic apply할 수 있음
- replace/delete target이 현재 memory에 정확히 한 번 존재함
- patch apply 결과의 hash가 candidate `next_memory` hash와 일치함
- NO_OP은 operations가 비어 있고 memory hash가 변하지 않음
- 30-add compaction trigger와 counter가 deterministic하게 재현됨
- compaction rewrite는 non-empty·character limit·hash·실제 길이 감소만 검사하고 의미 보존을
  rule로 판단하지 않음

**Summary export check**

- UPDATE이면 확정 `M(t)`를 그대로 complete `new_memory` target으로 사용
- NO_OP이면 tool call 없이 `M(t) = M(t-1)`로 export
- serialization, character limit과 source Gold hash 일치만 확인
- 별도 Summary 의미 판단이나 operation replay를 수행하지 않음

다음 항목은 rule gate에서 판정하지 않는다.

- 해당 정보가 durable memory로 중요한지
- compaction rewrite가 이전 사실을 의미적으로 보존했는지
- UPDATE와 NO_OP 중 어느 쪽이 의미적으로 맞는지
- value, condition, attribution과 temporal scope의 해석이 맞는지
- 표현이 다른 두 memory가 의미적으로 완전히 같은지

이 판단은 Luna consensus, Terra/SOL과 human audit에 맡긴다. 따라서 gate 결과는
`PASS / FAIL`만 사용하며, FAIL은 재현 가능한 rule violation을 반드시 포함한다.

### 3.6 Terra → SOL → Human escalation

다음 경우 Terra로 보낸다.

- 2/3 다수결이 있어도 한 후보와 의미적으로 불일치함
- 세 후보가 모두 의미적으로 다름
- retry 후에도 gate가 실패함
- consensus 과정에서 UPDATE/NO_OP, attribution 또는 scope 판단이 해결되지 않음

Terra에는 후보 순서에 따른 편향을 줄이도록 A/B/C를 중립 ID로 제시하고, vote count,
material diff와 gate 결과를 함께 준다. 다수 후보는 preferred이지만 **근거가 확인될
때만** 선택할 수 있다. 일반 turn은 current turn과 `M(t-1)`을 사용한다. QA, 기존
trace, 미래 turn은 보지 않는다.

- Terra와 SOL reasoning effort는 각각 `high`, `high`로 고정
- Terra 출력은 `SELECT / CORRECT / UNCERTAIN`으로 제한
- Terra가 선택하거나 수정한 후보는 두 gate를 다시 통과해야 함
- Terra가 해결하지 못하면 SOL로 escalation
- SOL 결과도 두 gate를 다시 통과해야 함
- SOL이 해결하지 못하면 scenario를 `PAUSED_REVIEW`로 중단
- Human correction이 commit된 뒤에만 같은 scenario를 resume
- escalation 결과가 이전 Gold Memory를 바꾸면 이후 turn은 아직 실행되지 않았으므로
  rollback 대신 확정 상태에서 바로 계속함

Human Review request에는 current turn, 확정 `M(t-1)`, 후보별 operation·gate 결과·적용
후 memory와 Terra/SOL 시도를 저장한다. 로컬 Web UI에서 다음 중 하나를 제출한다.

- gate를 통과한 후보 `SELECT`
- 명시적 `NO_OP`
- corrected Combined candidate 직접 입력
- compaction이면 rewrite 후보 `SELECT` 또는 corrected compacted memory 입력

request hash가 달라진 제출은 거부하며, Human 결과도 같은 rule gate를 재통과해야 한다.
통과하면 paused turn과 checkpoint를 atomic commit하고 controller가 해당 tmux runner를
재개한다. 실패한 제출은 `REJECTED` audit으로 보존하고 새 revision에서 다시 검토한다.

**30-add compaction 예외:** rewrite 후보 3개가 일치하지 않으면 Terra를 거치지 않고
SOL에 post-patch memory, A/B/C rewrite, gate와 consensus를 전달한다. SOL은 후보 선택,
직접 correction 또는 `UNCERTAIN`만 반환하며, `UNCERTAIN`은 Human Review로 중단한다.

### 3.7 Gold checkpoint와 controller

기존 checkpoint의 transaction·hash-chain 패턴을 참고해 별도
`GoldCheckpointStore`를 만든다.

turn commit 하나에 다음을 atomic하게 저장한다.

- 세 Luna candidates와 usage
- Combined patch/compaction record와 Summary export hash
- consensus 결과
- gate와 escalation 결과
- 확정 update record, reason, evidence
- before/after Gold Memory와 hash
- 다음 turn index와 scenario status

상태는 `RUNNING / PAUSED_REVIEW / COMPLETED / FAILED`로 제한한다. Controller는
scenario별 process를 관리하지만 scenario 내부에서는 항상 turn 하나만 처리한다.

운영 프로세스는 다음 두 개다.

```bash
python evaluation/experiment-scripts/serve_vehiclemembench_v2_reviews.py
python evaluation/experiment-scripts/run_vehiclemembench_v2_review_controller.py
```

Web은 기본 `127.0.0.1:8765`에만 bind한다. Controller는 5초마다 paused checkpoint를
queue에 등록하고 `SUBMITTED` review가 생기면 해당 `vehiclemem_v2_sN` tmux pane을
checkpoint에서 재개한다.

### 3.8 Export, audit, 외부 평가

canonical label에서 방식별 학습 형식을 만든다.

- Summary SFT: `M(t-1) + turn → complete M(t) 또는 NO_OP`
- Combined SFT: `M(t-1) + turn → temporal patch operations + reason`

생성 중 audit:

- turn gap과 memory hash chain
- evidence exact match, Summary schema/hash, Combined patch replay
- forbidden input leakage
- unresolved/review 상태
- consensus, escalation, usage와 비용

scenario가 완료되고 artifact가 고정된 뒤 공식 QA로 최종 memory를 별도 평가한다.
QA 실패를 개별 turn correction에 사용하지 않고, pipeline version 전체의 품질
지표로만 보고한다.

완료 직후 SOL coverage validator에 최종 memory와 10개 QA의 질문·Gold Memory·Gold
Tool Calls를 제공한다. 질문과 Gold 항목은 필요한 정답 사실을 정의할 뿐 evidence로
인정하지 않으며, `SUPPORTED`는 최종 memory의 exact evidence를 요구한다. 결과는
`SUPPORTED / PARTIAL / MISSING`으로 저장한다. `SUPPORTED + PARTIAL`을 usable
coverage로 집계하며, 10개 QA에서 `MISSING > 3`이면
`REGENERATION_RECOMMENDED`로 표시한다. 이 표시는 기존 label을 수정하거나 자동으로
재생성하지 않는다.

## 4. 결정 상태

### 확정

- 동일 prompt/config의 독립 Luna 호출 3회, seed와 prompt variant는 사용하지 않음
- Summary와 Combined만 지원
- 하나의 공통 Gold trajectory를 사용하고 Combined temporal patch를 canonical update로 사용
- 확정 `M(t)`를 Summary full-rewrite target으로 export
- 3/3 또는 비본질적 형식 차이만 auto path; 의미가 다른 2/3은 Terra
- V2 첫 version에서는 soft 2/3 auto-accept를 사용하지 않음
- Combined는 기존 contract와 같이 turn당 최대 32개 operations와 30-add compaction 사용
- deterministic normalization은 casing, whitespace, field order와 명백한 alias만 허용
- NO_OP reason도 current turn에 근거한 한 문장으로 저장
- reasoning effort는 Luna `medium`, Terra `high`, SOL `high`로 고정
- S1의 UPDATE·escalation은 전수 감사하고 NO_OP은 stratified 5% 감사; 오류 발견 시 확대
- Human Review는 중앙 SQLite queue + local Web + tmux 자동 재개로 운영

### 실행 결과를 본 뒤 결정

| 항목 | 권장안 | 결정 시점 |
| --- | --- | --- |
| Soft 2/3 도입 | 첫 version에서는 사용하지 않고 S1–S10 audit 후 새 version에서만 검토 | S1–S10 후 |
| NO_OP audit 확대 | 5% 표본 오류율을 보고 확대 또는 전수 감사 결정 | S1 후 |
| Train/Eval 30/20 split | 전체 분포를 본 뒤 scenario 단위로 고정 | S1–S50 완료 전 |

## 5. Phase A — Scenario 1 신규 Smoke Test

### A0. 로컬 fixture

Cloud 호출 없이 다음 케이스를 먼저 통과시킨다.

- background NO_OP
- 신규 preference add
- 기존 preference update/supersede
- correction과 delete
- user별 동일 attribute 분리
- conditional/temporary state 보존
- 세 후보 agree/disagree
- 30-add compaction 후보 agree/disagree
- Summary export hash와 Gold hash 일치
- Terra/SOL correction 재검증
- unresolved에서 scenario 중단 및 checkpoint resume
- generator/critic 입력에 QA, trace, future turn이 없음을 확인

### A1. 제한 canary

빈 `M0`에서 S1 prefix를 strict order로 실행한다. 특정 UPDATE 위치를 기존
annotation으로 고르지 않고, 고정된 turn limit까지 실제 pipeline을 실행한다.

완료 조건:

- 처리된 매 turn에 candidate 3개가 저장됨
- consensus, gate, escalation path가 재현 가능함
- restart 후 완료된 candidate와 Gold turn이 재호출되지 않음
- 실패나 review 발생 시 다음 turn이 실행되지 않음
- before/after memory hash chain이 연속적임

### A2. S1 전체

빈 `M0`에서 S1의 2,438 turns를 처음부터 strict order로 실행한다. turn candidate의
기본 Luna 호출은 retry 전 7,314회이며, 여기에 `3 × compaction 횟수`와 escalation
호출이 추가된다. 실행 전 dry-run으로 calls, token, 비용과 provider latency를
산출한다.

완료 조건:

- active Gold label 2,438개, turn gap 0
- before/after hash chain 100% 일치, Combined patch step replay 100% 일치
- Summary export memory hash가 동일 turn의 Gold `M(t)` hash와 100% 일치
- forbidden input leakage 0
- unresolved와 open review 0
- 모든 UPDATE, escalation, human correction을 사람이 감사
- NO_OP는 날짜 구간별 stratified 5%를 사람이 감사
- human audit의 unsupported label 0
- prompt/schema/model과 dataset hash가 manifest에 고정됨

S1 artifact가 고정된 뒤 공식 QA 10개로 최종 memory를 외부 평가하고 결과를 별도
저장한다. 이 평가 결과를 이용해 S1 label을 사후 수정하지 않는다. prompt나
schema가 바뀌면 새 version root에서 빈 `M0`부터 다시 실행한다.

## 6. Phase B — Scenario 1–10 확장

S1에서 schema를 freeze한 뒤 진행한다. S1–S10은 총 27,303 turns이므로 turn
candidate의 기본 Luna 호출은 retry 전 81,909회이며, compaction과 escalation 호출은
별도다.

### 실행 순서

1. S1–S10 history의 parse, ordering, source hash를 preflight
2. S2를 단일 scenario canary로 빈 `M0`부터 실행
3. 문제가 없으면 scenario 최대 2개 병렬로 S3–S10 실행
4. 각 scenario 내부는 strict turn order 유지
5. scenario 완료 직후 audit, replay와 artifact hash 검증
6. 완료 artifact를 고정한 뒤 공식 QA 외부 평가를 별도 실행

### 완료 조건

- S1–S10 모두 `COMPLETED`, turn gap과 unresolved 0
- 모든 memory hash chain과 Combined patch replay 일치
- evidence exact match와 forbidden input audit 통과
- UPDATE와 escalation 전수 감사
- scenario·기간별 NO_OP 표본 감사
- consensus, Terra, SOL, Human 비율과 실패 유형 집계
- S1 대비 operation/escalation drift 분석
- 실제 비용으로 S11–S50 concurrency와 예산 확정

도중 prompt/schema를 변경하면 같은 version에서 일부 scenario만 새 설정을 쓰지
않는다. 새 version root를 만들고 S1부터 비교 가능한 범위를 재실행한다.

## 7. Phase C — Scenario 11–50 확장

S1–S10 human audit와 비용 검토가 통과한 뒤 prompt/schema를 동결한다.

### 실행 순서

1. S11–S50 history parse, ordering, source hash preflight
2. S11–20을 첫 production batch로 실행
3. 안정적이면 S21–30, S31–40, S41–50 순서로 확장
4. provider limit 안에서 scenario concurrency를 조절
5. review/failure scenario만 멈추고 나머지는 계속 실행
6. 완료 scenario는 read-only artifact로 잠그고 재호출하지 않음
7. 각 완료 artifact에 대해 공식 QA 외부 평가를 별도 실행

### 품질 관리

- 모든 escalation과 human correction 전수 감사
- auto-accepted UPDATE 전수 또는 높은 비율 감사
- NO_OP는 scenario·기간·operation 분포별 표본 감사
- S1–S10 대비 consensus와 escalation rate drift 감시
- drift가 크면 해당 batch를 중단하고 새 version 여부 결정
- 50개 scenario 전체에서 leakage, evidence, Combined replay, hash-chain audit

### 최종 완료 조건

- S1–S50 모두 `COMPLETED`, turn gap과 unresolved 0
- canonical labels와 Summary/Combined SFT export 재현 가능
- dataset, prompt, schema, model, artifact hash manifest 완성
- fixed 30/20 train/eval split과 leakage audit 완료
- QA 외부 평가와 cost, latency, consensus, escalation, human correction 보고서 완성

## 8. Artifact 구조

새 pipeline은 기존 결과를 읽거나 덮어쓰지 않고 새 root를 사용한다.

```text
evaluation/vehiclemembench-v2/consensus-gold-v2/
  global_manifest.json
  split_manifest.json
  review_queue.sqlite
  scenario-XX/
    manifest.json
    checkpoint.sqlite
    candidates.jsonl
    consensus.jsonl
    labels.jsonl
    audit.json
    metrics.json
    COMPLETED | PAUSED_REVIEW | FAILED
  external-evaluation/
    scenario-XX-qa.json
  exports/
    summary_sft.jsonl
    combined_sft.jsonl
```

## 9. 구현 순서

1. ~~Versioned schema와 V1-equivalent input builder~~
2. ~~Luna 3-sample generator와 candidate cache~~
3. ~~Consensus normalizer~~
4. ~~Combined patch/30-add compaction validator와 consensus~~
5. ~~Gold checkpoint와 strict S1 runner~~
6. ~~Terra/SOL adapter와 gate 재검증~~
7. ~~Human Review queue/Web UI/rule 재검증/tmux 자동 resume~~
8. Metrics, audit, Summary/Combined SFT exporter
9. S1 one-turn → prefix cloud canary
10. S1 전체 및 외부 QA → S1–S10 → S11–S50
