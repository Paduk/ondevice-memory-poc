# Fact-First 단계별 Oracle 평가 계획

상태: `O0–O6 완료`

최종 업데이트: 2026-07-30

관련 결과:

- [R5 Scenario 1–5](tool-schema-on-device-memory-r5-results.md)
- [Holdout Scenario 6–10](tool-schema-on-device-memory-holdout-s6-s10-results.md)

## 1. 목표

Fact-first의 성능 손실을 다음 파이프라인 단계별로 분리한다.

```text
History
  → Extraction
  → Structure
  → Validation·Storage
  → Routing·Retrieval
  → Late Binding
  → Agent·Tool 실행
```

각 단계만 정답으로 교체한 **독립 Oracle**과 앞 단계부터 누적 교체한
**누적 Oracle**을 함께 측정한다. Oracle 결과는 실제 방법의 성능이 아니라
해당 단계가 완벽할 때 얻을 수 있는 성능 상한이다.

## 2. 평가 모드

| 모드 | 정답으로 교체하는 부분 | 확인할 병목 |
| --- | --- | --- |
| `baseline_fact` | 없음 | 현재 Fact-first 기준선 |
| `oracle_extraction` | Gold fact proposal | 사실 후보 누락 |
| `oracle_structure` | entity·predicate·condition | 잘못된 구조화 |
| `oracle_gate` | 정답 proposal의 저장 판정 | reject·review·잘못된 통합 |
| `oracle_route` | Gold Tool 이름 | Tool domain 선택 |
| `oracle_retrieval` | Gold 관련 Memory Top-k | Memory 검색·순위 |
| `oracle_binding` | Gold Tool argument hint | late binding |
| `oracle_full` | 위 모든 단계 | Agent 자체의 잔여 오차 |

`oracle_route`와 `oracle_retrieval`은 같은 상위 문제에 속하지만 원인 분리를
위해 별도 결과로 기록한다. `Gold Tool`은 Tool 이름만 제공하며 argument를
노출하지 않는다. `oracle_binding`에서만 정답 argument를 사용한다.

## 3. Gold contract

VehicleMemBench가 직접 제공하는 정답은 다음과 같다.

- `gold_memory`: task와 관련된 원본 History 근거
- `gold_calls[].name`: 정답 Tool route
- `gold_calls[].arguments`: 정답 runtime argument

벤치마크에는 Fact-first의 `entity·predicate·condition` 정답 label이 없다.
따라서 Extraction·Structure·Gate Oracle에는 별도의
`oracle_facts_v1.jsonl` annotation이 필요하다. 이 파일은 각 사실에 대해
scenario, task, 원본 evidence, canonical entity, predicate, value,
conditions, 관련 Gold Tool을 기록하고 원본 History에 실제 근거가 있는지
검증한다. 자동 생성 결과를 그대로 Gold로 간주하지 않는다.

## 4. 실행 순서

### O0 — Gold contract·기준선 고정

- Scenario 1–10의 benchmark hash와 기존 Fact-first 결과를 고정한다.
- Gold memory·Tool·argument의 완전성과 History 근거 연결 가능성을 검사한다.
- Oracle mode, 단독·누적 실행, artifact schema를 정의한다.

완료 조건:

- 정답 누락·중복·스키마 오류 보고서가 생성된다.
- Gold argument가 허용된 모드 외 Agent prompt에 들어가지 않는 테스트가 있다.

구현·실행 결과:

- `oracle.py`에 독립 Oracle mode와 누적 stage contract를 추가했다.
- Scenario 6–10의 50 task에서 Gold Memory 누락 0, Gold call 누락 0,
  미등록 Tool 0, 중복 Gold call 0을 확인했다.
- Gold call 57개와 argument leaf 80개를 확인했다.
- Gold Memory의 인용 근거 115개 중 91개는 History와 literal 연결됐으며,
  나머지 24개는 paraphrase·문장부호 차이 가능성이 있어 Fact annotation
  작성 시 검토 대상으로 남겼다.
- 기존 holdout run의 dataset·Tool schema·Scenario hash가 모두 일치했다.
- Binding 외 모드에서 Gold argument 노출을 금지하는 contract test를 추가했다.

Artifact:
`../../ubuntu/evaluation/vehiclemembench-oracle/8813cd56-8ad9-4502-a5c4-e45839d4ef79`

### O1 — Oracle Routing

- 기존 `oracle_tool_fact_patch`를 공통 Oracle mode로 편입한다.
- 정답 Tool schema만 제공하되 Memory와 argument는 Ours를 유지한다.

완료 조건:

- Baseline 대비 Tool F1·ESM 변화와 route miss 감소량을 보고한다.

구현·실행 결과:

- 기존 `oracle_tool_fact_patch`를 `oracle_route`의 실행 profile로 사용했다.
- Ours의 Fact DB·Retrieval·late binding은 유지하고 Gold Tool 이름만
  Agent의 Tool boundary와 Memory router에 제공했다.
- Scenario 6–10, 50 task에서 ESM `0.50→0.56` (`+0.06`), State F1
  `0.661→0.800` (`+0.139`), Tool F1 `0.507→0.573` (`+0.067`),
  Argument exact `0.42→0.46` (`+0.04`)을 기록했다.
- Tool selector miss는 15건에서 0건으로 감소했다.
- Retrieval Recall@k는 `0.49`로 동일했고 argument mapping rejection과
  Tool execution error가 남아 있어 Routing만으로 병목이 해소되지는 않았다.
- 실제 추가 Agent token 기준 추정비용은 약 `$0.3945`다.

Artifact:
`../../ubuntu/evaluation/vehiclemembench-oracle/4d693463-d834-438b-8e60-7d88b3e611cf`

### O2 — Oracle Retrieval

- Ours DB에 정답 사실이 존재하는 task만 대상으로 관련 record를 강제 Top-k에
  포함한다.
- `record_absent`와 `record_present_but_missed`를 분리한다.

완료 조건:

- 정답 record가 존재하는 subset에서 record-level Recall@k를 1로 만든
  성능 상한을 보고한다.

구현·실행 결과:

- Scenario 6–10의 50 task를 Gold evidence와 frozen active Fact에 대조한
  reviewed annotation을 추가했다.
- 정답 Fact가 active DB에 존재하는 task는 23건, 존재하지 않는 task는
  27건이었다. wrong value·wrong predicate·superseded-only record는
  `record_absent`로 분류했다.
- 23개 semantic key가 cache에서 각각 정확히 하나의 record와 일치하는지
  deterministic하게 검증했다.
- `oracle_retrieval_fact_patch`는 Ours routing·DB·late binding을 유지하고
  reviewed record만 Top-k 선두에 배치한다. Gold Tool·argument는 Agent에
  직접 노출하지 않는다.
- Baseline은 존재하는 정답 record 23건 중 21건을 이미 Top-k에 포함했다.
  실제 Top-k miss는 Scenario 7의 task 08·09 두 건뿐이었고, 두 건 모두
  Oracle 주입 후 ESM 실패에서 성공으로 바뀌었다.
- 정답 record의 순서 또는 포함 여부가 바뀐 task는 9건이었다. 이 구간의
  ESM 성공은 Baseline 6/9에서 Oracle 7/9로 1건 순증했다.
- 전체 ESM은 `0.50→0.46`이지만, 하락 4건 중 3건은 Memory context가
  동일했던 task에서 발생해 Agent 확률 변동으로 분류한다. 따라서 전체
  단일-run 차이를 Retrieval의 음의 효과로 해석하지 않는다.
- 기존 value-substring Recall@k는 숫자 `1` 등이 unrelated record의
  version·다른 숫자와 일치하는 false positive가 있어 `0.49`로 변하지
  않았다. O2부터는 record-level recall `21/23→23/23`을 주 판정으로 쓴다.
- 실제 추가 Agent token 기준 추정비용은 약 `$0.5659`다.

Annotation:
`../../ubuntu/src/palmclaw_ubuntu/evaluation_datasets/vehiclemembench_oracle_retrieval_s6_s10_v1.json`

Artifact:
`../../ubuntu/evaluation/vehiclemembench-oracle/727c09fb-1eef-4481-befc-bc3b3fcdaa53`

### O3 — Oracle Binding

- 검색된 사실은 Ours를 유지하고 정답 Tool argument를 실행 hint로 제공한다.
- Gold Tool만 제공한 결과와 Gold argument까지 제공한 결과를 분리한다.

완료 조건:

- hint rejection, Argument exact, Tool F1 개선량을 보고한다.

구현·실행 결과:

- `oracle_binding_fact_patch`는 기존 Ours Top-k에 reviewed 정답 Fact가 모두
  포함된 21 task에서만 Gold Tool 이름과 canonical argument를 실행 hint로
  제공한다. Fact가 없거나 검색에서 빠진 task에는 Gold 값을 노출하지 않는다.
- `oracle_retrieval_binding_fact_patch`는 O2 Retrieval까지 누적해 정답 Fact가
  DB에 존재하는 23 task에서만 같은 Binding Oracle을 적용한다.
- 전체 50 task 결과는 다음과 같다.

| 모드 | ESM | State F1 | Tool F1 | Argument exact | Binding 적용 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline Fact-first | 0.50 | 0.661 | 0.507 | 0.42 | 0 |
| O3 Binding | 0.54 | 0.707 | 0.540 | 0.46 | 21 |
| O2 + O3 | 0.54 | 0.673 | 0.570 | 0.48 | 23 |

- 누적 O2+O3의 Binding 적용 subset은 ESM `19/23→22/23`, Argument exact
  `15/23→19/23`으로 개선됐고 적용 subset의 퇴행은 없었다.
- 전체 Baseline과 비교하면 누적 모드는 ESM 실패 3건을 복구하고 1건이
  퇴행해 `25/50→27/50`으로 2건 순증했다. 서로 다른 Cloud run의 확률
  변동이 포함되므로 subset의 단계별 비교를 주 판정으로 사용한다.
- 실행 hint rejection은 Baseline 177건에서 O3 107건, 누적 O2+O3
  99건으로 감소했다. Gold hint가 적용되지 않은 27 task의 기존 rejection은
  그대로 남는다.
- 누적 모드의 유일한 Binding 적용 실패는 `vehicle-10-08`이다. 정답
  `centerInformationDisplay=English` hint가 있었지만 Agent가 overhead screen과
  instrument panel Tool까지 추가 호출했다. 이는 Binding 이후 Agent의
  과잉 실행 오차다.
- canonical enum을 저장 단계부터 유지하면 semantic value를 runtime enum으로
  다시 번역하는 부담을 줄일 수 있다. 다만 이번 Oracle은 Tool과 전체
  argument까지 제공하므로 enum 정규화만의 효과로 해석하지 않는다.
- 두 프로필의 Agent 사용량은 입력 314,264, 출력 15,189 token이며 기존과
  같은 표준 단가 환산 시 증분비용은 약 `$1.0135`다. 기존 Fact Memory
  cache를 재사용해 Memory 생성 비용은 다시 포함하지 않았다.

Artifact:
`../../ubuntu/evaluation/vehiclemembench-oracle/2d365ff2-7d50-4d3d-807c-93daedd1361f`

### O4 — Oracle Gate

- proposal이 Gold fact와 일치하면 통과시키고, 불일치하면 기존 정책을 유지한다.
- 단순 Validation disable은 별도 보조 실험으로만 사용한다.

완료 조건:

- reject·review 중 복구 가능한 정답 사실 수와 E2E 개선량을 보고한다.

구현·실행 결과:

- Scenario 6–10의 `record_absent` 27 task를 frozen 후보 이벤트와 대조했다.
  전체 review 8건·reject 2건 중 task 정답 Fact와 일치하는 후보는
  `vehicle-07-00`의 1건뿐이었다. 나머지 26 task는 Gate가 살릴 정답 후보
  자체가 없었다.
- 복구 후보는 `voice_guidance_preference={"enabled":false}`와
  `time_of_day=morning`을 가진 semantic `REVIEW`였다. 후보의 구조·값·근거는
  바꾸지 않고 disposition만 reviewed Gold `ACCEPT`로 교체했다.
- 원본 Memory cache를 변경하지 않도록 SQLite를 in-memory로 복제한 뒤
  annotation hash와 원래 status·validation code가 정확히 일치하는 후보만
  재적용한다. 구조 오류, PII, secret 후보는 Oracle도 우회하지 못한다.
- Scenario 7의 Fact record는 전체 `18→19`, active `15→16`으로 증가했다.
  원래 review event는 보존되고 Oracle 적용 event가 별도로 기록된다.
- Gate-only에서는 새 record가 일반 검색 순위 10위로 Top-5에서 빠졌다.
  질문의 `reading`이 조명 Tool로 lexical routing되어 Voice Guidance가
  검색되지 않았으므로 Scenario 7 ESM은 Baseline과 같은 `0.30`이었다.
- `Gate + Oracle Retrieval + Oracle Binding`에서는 새 record를 Top-k에
  포함하고 exact Tool argument로 binding했다. `vehicle-07-00`이 실패에서
  성공으로 바뀌었고, 다른 9 task의 ESM 변화는 없었다.

| Scenario 7 모드 | ESM | State F1 | Tool F1 | Argument exact |
| --- | ---: | ---: | ---: | ---: |
| Baseline Fact-first | 0.30 | 0.633 | 0.367 | 0.30 |
| O2 + O3 | 0.60 | 0.773 | 0.633 | 0.50 |
| O4 Gate-only | 0.30 | 0.567 | 0.367 | 0.30 |
| O4 + O2 + O3 | 0.70 | 0.800 | 0.667 | 0.60 |

- 다른 Scenario에는 Gate 복구 후보가 없어 Cloud Agent를 재호출하지 않았다.
  기존 50-task O2+O3 결과에 유일한 순개선 1건만 합성하면 ESM 상한은
  `0.54→0.56`이다. 이는 fresh 50-task run이 아닌 taskwise counterfactual이다.
- 20 Agent task의 입력은 67,619, 출력은 3,386 token이며 기존과 같은
  표준 단가 환산 증분비용은 약 `$0.2198`다. 새 record embedding 1회 외
  Memory 생성 호출은 재사용 cache에 포함돼 다시 과금한 것으로 세지 않았다.

Annotation:
`../../ubuntu/src/palmclaw_ubuntu/evaluation_datasets/vehiclemembench_oracle_gate_s6_s10_v1.json`

Artifact:
`../../ubuntu/evaluation/vehiclemembench-oracle/6e9bf47d-732d-47a3-9b58-1f1c7b37239f`

### O5 — Oracle Structure·Extraction

- 검토된 `oracle_facts_v1.jsonl`을 사용한다.
- Structure Oracle은 기존 후보의 canonical field만 교체한다.
- Extraction Oracle은 누락된 Gold proposal을 추가하되 이후 단계는 Ours를
  유지한다.

완료 조건:

- proposal recall, active Fact 수, Recall@k, E2E 개선량을 각각 보고한다.

구현·실행 결과:

- O2의 `record_absent` 27 task를 frozen proposal과 원본 History에 다시
  대조해 `Gate 1 + Structure 6 + Extraction 20`으로 완전 분류했다.
  Extraction 20 task는 19개의 고유 Fact로 구성되며, Scenario 7의 두 task는
  같은 hot-weather AC 18°C Fact를 공유한다.
- Structure 6개는 기존 applied candidate hash를 고정하고 entity, predicate,
  applicability만 reviewed canonical field로 교정했다. Extraction 19개는
  원본 History에서 정확히 한 번 발견되는 contiguous evidence를 가진 reviewed
  Gold proposal로 추가했다.
- 모든 annotation은 dataset·Tool schema hash, task coverage, candidate hash,
  evidence 위치를 load 시 검증한다. 원본 cache는 변경하지 않고 SQLite
  in-memory clone에만 overlay한다.
- Structure 6/6, Extraction 19/19 proposal이 현재 Validation·Storage 경로를
  통과했다. 기존 semantic validation을 통과한 Structure candidate는 같은
  stored decision을 재사용하며 임의로 Gate를 우회하지 않는다.
- 원래 Gold-relevant proposal이 존재한 것은 27건 중 7건
  (`Gate 1 + Structure 6`, 25.9%)이고, 20건(74.1%)은 proposal 자체가
  누락된 Extraction 실패였다.
- active Fact 수는 5개 Scenario 합계 `47→53`(Structure),
  `47→66`(Extraction)으로 증가했다. 추가 Fact는 강제 Top-k 없이도
  Structure task `6/6`, Extraction task `20/20`에서 일반 검색 Top-k에
  포함됐다.

전체 50 task:

| 모드 | ESM | State F1 | Tool F1 | Argument exact |
| --- | ---: | ---: | ---: | ---: |
| Baseline Fact-first | 0.50 | 0.661 | 0.507 | 0.42 |
| O2 + O3 | 0.54 | 0.673 | 0.570 | 0.48 |
| O5 Structure | 0.48 | 0.662 | 0.505 | 0.38 |
| O5 Structure + Retrieval + Binding | 0.66 | 0.791 | 0.687 | 0.60 |
| O5 Extraction | 0.76 | 0.837 | 0.721 | 0.60 |
| O5 Extraction + Retrieval + Binding | 0.90 | 0.933 | 0.900 | 0.86 |

O5 적용 task subset:

| subset | Baseline ESM | Stage ESM | Stage + R·B ESM |
| --- | ---: | ---: | ---: |
| Structure 6 task | 2/6 | 5/6 | 6/6 |
| Extraction 20 task | 3/20 | 18/20 | 20/20 |

- Structure의 일반 파이프라인 잔여 실패 1건은 fan speed 값은 맞았지만
  `zone=driver`로 binding되어 Gold `zone=all`과 달랐다.
- Extraction의 일반 파이프라인 잔여 실패 2건은 각각 video의
  switch·play 동작 누락과 seat massage Tool binding 실패였다. 둘 다 Fact
  검색 실패가 아니라 Memory 이후 실행 변환 실패이며 Binding Oracle에서
  복구됐다.
- 전체 평균에서 O5 Structure가 Baseline보다 낮아 보이는 것은 Structure
  비대상 44 task의 Cloud run 변동과 여전히 누락된 Extraction Fact가 섞였기
  때문이다. 원인 판정은 적용 subset의 `2/6→5/6`을 사용한다.
- Extraction proposal이 없던 경우 canonical structure도 함께 정의해야 하므로
  이 Oracle은 엄밀히는 `missing proposal + proposal structure`의 결합 상한이다.
  실제 Fact-first 성능이 아니라 Extraction이 완전할 때의 상한으로 해석한다.
- 두 run의 Agent 사용량은 입력 731,096, 출력 29,342 token이다. 기존과 같은
  표준 단가 환산 시 약 `$2.2679`이며, 추가 embedding 비용은 약 `$0.0001`다.
  기존 Fact Memory 생성 cache는 재사용했으므로 Memory 생성 비용은 포함하지
  않았다.
- Ruff와 전체 Ubuntu 회귀 테스트 `191 passed, 1 skipped`를 통과했다.

Annotation:
`../../ubuntu/src/palmclaw_ubuntu/evaluation_datasets/vehiclemembench_oracle_stage_facts_s6_s10_v1.json`

Artifacts:

- Structure:
  `../../ubuntu/evaluation/vehiclemembench-oracle/2be16bac-890d-4e85-a579-749813682212`
- Extraction:
  `../../ubuntu/evaluation/vehiclemembench-oracle/afe56080-880b-4ac0-bcfd-b4fd2213465e`

### O6 — 독립·누적 비교

- 독립: Baseline에서 한 단계만 Oracle로 교체
- 누적: Extraction부터 Binding까지 앞 단계 Oracle을 차례로 누적
- `oracle_full`로 Memory 이후 Agent 오차를 측정

완료 조건:

- 단계별 절대 개선폭과 누적 개선폭을 같은 표로 생성한다.
- Scenario별·reasoning type별 결과와 ESM의 초기 상태 우연 일치를 표시한다.

구현·실행 결과:

- 기존 active Gold-relevant Fact 23 task에 O5 Structure 6 task,
  Extraction 20 task, O4 Gate 1 task를 한 in-memory SQLite clone에서
  결합했다. 27 task는 공유 Fact 한 개를 포함해 26개 고유 overlay record로
  복구됐으며 원본 cache는 변경하지 않았다.
- `Full Memory`는 관련 record coverage만 `50/50`으로 만든 뒤 기존
  routing·Top-k·argument binding·동적 Tool discovery를 그대로 사용한다.
- `Full Pipeline`은 같은 DB에서 reviewed record를 Top-k에 보장하고,
  Gold Tool boundary와 Gold argument binding을 적용한다. 따라서 결과는
  실제 방법의 성능이 아니라 Memory 이후 단계까지 완벽할 때의 상한이다.

| 모드 | ESM | State F1 | Tool F1 | Argument exact |
| --- | ---: | ---: | ---: | ---: |
| Baseline Fact-first | 0.50 | 0.661 | 0.507 | 0.42 |
| Full Memory Oracle | 0.80 | 0.869 | 0.763 | 0.64 |
| Full Pipeline Oracle | 1.00 | 1.000 | 0.993 | 0.98 |

- Full Memory에서 reviewed record는 일반 Top-k에 `46/50` 포함됐고 ESM은
  `40/50`이었다. 검색에서 빠진 4건 중 2건은 우연히 성공했고, 2건은
  실패했다.
- 관련 record가 검색된 task에서도 8건이 ESM 실패했다. 대표 원인은
  `zone=driver` 대 `zone=all`, radio 대 music Tool 선택, 복합 action의
  switch·play 누락, 여러 display·language Tool의 과잉 호출이었다.
- Full Pipeline은 record-level Top-k `50/50`, Binding 적용 `50/50`,
  ESM `50/50`을 달성했다. execution hint rejection은 `124→0`,
  discovery call 평균은 `0.40→0`으로 감소했다.
- Full Pipeline의 유일한 Tool-call 불일치는 `vehicle-10-02`다. Gold는
  window close와 inside circulation 두 호출이지만 Agent가 이미 닫혀 있던
  window close를 생략했다. 최종 상태는 같아 ESM은 성공했지만 Tool F1과
  Argument exact는 각각 `0.993`, `0.98`로 남았다. 이는 초기 상태 우연
  일치 사례다.
- Baseline 대비 Full Memory의 ESM 상한 증가는 `+0.30`, Full Memory 이후
  routing·retrieval·binding 통제의 추가 증가는 `+0.20`이다. 따라서 현재
  병목은 Fact 생성 누락이 가장 크고, 그 다음은 Memory를 Tool 실행으로
  변환하는 downstream 단계다. Gate 단독 병목은 27건 중 1건으로 작았다.
- 두 O6 profile의 Agent 사용량은 입력 303,995, 출력 10,384 token이다.
  기존과 같은 표준 단가 환산 시 약 `$0.9157`, 추가 embedding은 약
  `$0.0001`다. 기존 Memory 생성 cache는 재사용했다.
- Ruff와 전체 Ubuntu 회귀 테스트 `191 passed, 1 skipped`를 통과했다.

Artifact:
`../../ubuntu/evaluation/vehiclemembench-oracle/ede6d720-0f0b-4a65-91c0-247a496d4f7c`

추가 표:

- `marginal-effects.tsv`: Structure·Extraction·전체 Fact·downstream의
  단계별 개선폭
- `cumulative-effects.tsv`: Baseline → Full Memory → Full Pipeline 누적 결과

## 5. 공통 지표와 산출물

주 지표:

- ESM, State F1, Tool F1, Argument exact
- proposal recall, accepted·review·rejected 수
- active Fact 수, Retrieval Recall@k
- route miss, hint rejection, Tool omission·extra call
- Agent·Memory 호출 수, token, latency, 비용

Artifact:

```text
evaluation/vehiclemembench-oracle/RUN_ID/
  manifest.json
  oracle-contract-audit.json
  cases.jsonl
  metrics.json
  marginal-effects.tsv
  cumulative-effects.tsv
  results.md
```

모든 case에는 사용한 Oracle 단계와 Gold 정보 노출 범위를 기록한다.

## 6. 평가 규칙

- 기본 비교 구간은 기존 holdout인 Scenario 6–10, 50 task다.
- 동일 모델·prompt·top-k·token budget과 기존 Memory cache를 사용한다.
- 독립 모드는 한 단계만 바꾸고 나머지는 Baseline과 동일하게 유지한다.
- Cloud Agent 변동을 줄이기 위해 주요 모드는 동일 seed 조건으로 반복한다.
- 기존 `gold_memory`가 완벽하다고 가정하지 않고 O0 audit 결과를 함께 공개한다.
- Gold argument는 `oracle_binding`과 `oracle_full` 외에는 Agent에 노출하지 않는다.
- ESM 성공 시에도 missing·extra·error Tool call을 별도 표시한다.
