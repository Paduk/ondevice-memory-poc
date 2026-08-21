# VehicleMemBench V1 재현 및 V2 확장 계획

## 결론

V1 생성 파이프라인을 먼저 재현하되, 신규 100개를 모두 만든 뒤 V2를 덧붙이지는
않는다. **소규모 V1 재현·검증 → V2 schema 결합 → 통합 pilot → 100개 생성** 순서로
진행한다. 그래야 V1 호환성을 먼저 확인하면서도 V2에 필요한 turn 원천 정보가 소실되지
않는다.

현재 상태: **S1 Hybrid 전체 파일럿과 시나리오당 30개 Turn Quiz 생성 완료** — Dialogue,
최종 Quiz, 공개 V1 serializer와 VehicleWorld 실행 검증에 더해 event-batched dialogue를
turn-wise memory label로 확정하는 Hybrid 경로와 UPDATE 직후 V1-style Quiz 경로를
구현했다. S1은 2,642 turn, 11 UPDATE와 2,631 NO_OP을 생성했고, Turn Quiz는
Immediate 11 + Delayed 11 + Composite 8개로 구성했다. V1의 마지막
`recorded/saved` event에만 있던 preference update는 원본
Stage 2를 변경하지 않고, Terra가 고른 최초 causal evidence event로 옮긴 V2 전용
anchored Stage 2 복사본으로 실행한다. 최종 V1 Quiz 10개와 Turn Quiz 30개는 모두
공식 Tool schema와 simulator 검증을 통과했다. 3-way 5-scenario parity pilot과 Human
Review는 아직 scale gate로 남아 있다.

## 고정 목표

- 논문의 V1 분포를 따르는 신규 scenario 100개를 생성한다.
- scenario마다 3인 persona, 비차량 chain 20개, 차량 chain 10개와 최종 executable
  Quiz 10개를 갖는다.
- 다섯 reasoning type과 23 module·111 Tool schema를 유지한다.
- 같은 structured event 원천에서 V1 history/최종 Quiz와 V2 turn-wise label/중간 Quiz를
  함께 파생한다.
- 기존 V1 50개는 학습에 사용하지 않고 공통 외부 평가 세트로 유지한다.

## 재현 판단 원칙

세부 설계는 다음 우선순위를 반드시 따른다.

1. **논문 명시 사항:** 논문 본문·부록의 단계, 수량, reasoning type, 모델 설정과
   품질검사 원칙을 그대로 적용한다.
2. **공개 V1 출력 역추론:** 논문에 생략된 출력 schema·표현·분포·대화 형식은 기존
   50개 history와 QA에서 공통 규칙을 추출해 복원한다.
3. **최소 임의 설계:** 두 근거로 결정할 수 없는 prompt 문구, retry, 중간 artifact와
   validator 세부만 보수적으로 설계한다.

각 설계 항목은 manifest/spec에 `paper_specified`, `v1_inferred`, `project_defined` 중
하나로 표시하고 근거 위치, version과 변경 이유를 저장한다. 서로 충돌하면 논문을
우선하며, `v1_inferred`와 `project_defined` 변경은 새 dataset version으로 분리한다.

기존 V1 50개는 schema·집계 분포·형식 추론에만 사용한다. 생성 prompt에 특정 V1
scenario의 정답이나 문장을 복사하거나, 기존 50개의 평가 점수에 맞춰 반복 튜닝하지
않는다.

실제 신규 데이터의 Persona·Event·Dialogue·Answer 생성 모델은
`gpt-5.6-terra`로 고정한다. 이는 논문의 Gemini-3-Pro-Preview/GPT-4.1 구성과 다른
`project_defined model substitution`이며, 논문의 원 model·temperature 설정도
manifest에 별도로 보존한다. Terra는 기본 `reasoning_effort=medium`으로 호출하고
지원이 불확실한 temperature는 강제하지 않는다.

## 단계

### 1. V1 생성 명세 복원

논문과 공개 데이터에서 다음 versioned schema와 prompt를 복원한다.

```text
Persona group
→ executable/background event chains
→ temporally interleaved events
→ natural dialogues
→ final query + gold Tool calls + target state
```

먼저 기존 50개 전체에서 history·QA schema, field 값, 길이와 reasoning/module 분포를
결정론적으로 추출하는 분석기를 만든다. 논문에 없는 전체 prompt와 중간 생성 코드는
새 구현으로 명시하고, 논문 그대로인 부분, V1 출력에서 역추론한 부분과 임의 설계한
부분을 manifest에서 구별한다. 모든 model, temperature, seed, prompt, schema와 source
hash를 저장한다.

### 2. V1 소규모 parity pilot

신규 persona group 3~5개만 생성해 다음을 확인한다.

- 각 group: background 20개 + executable 10개 chain
- reasoning type·target module 분포가 원본과 크게 어긋나지 않음
- event와 dialogue의 사용자·값·조건·시간 정합성
- Quiz 정답 Tool call이 simulator에서 실행되고 의도한 target state에 도달함
- 미래 정답을 보지 않고도 history가 생성됨

여기까지 통과해야 대규모 생성을 허용한다.

### 3. V2 turn-wise 출력 결합

V1의 structured event와 dialogue timestamp를 보존한 채 각 turn `t`에 대해 다음을
추가한다.

- `M(t-1) + 현재까지의 대화 → M(t)` gold memory update
- `UPDATE / NO_OP`, 한 문장 reason, reason code와 source evidence
- 해당 시점까지의 정보만으로 풀 수 있는 중간 Turn Quiz
- executable gold Tool calls와 simulator target state

Turn Quiz와 memory에는 미래 event, 최종 V1 Quiz 및 최종 상태를 노출하지 않는다.
최종 V1 Quiz는 독립적인 품질 평가에만 사용한다.

### 4. 3-way 통합 pilot

Native 세부 설계는
[VehicleMemBench V2 Native Turn-wise 설계](vehiclemembench-v2-native-turnwise-design.md)를
따른다.

동일한 5개 Persona·event chain·최종 Tool target을 고정하고, 다음 세 경로만 달리해
paired 비교한다.

1. `post_hoc`: V1 dialogue를 모두 만든 뒤 V2 turn label을 복원
2. `native_turnwise`: 대화 한 turn 생성과 memory 판정을 순차 반복
3. `hybrid`: event 단위로 대화를 배치 생성하되, 다음 event 전에 turn-wise memory를 확정

자동 검증 후 blind Human Review를 거쳐 다음을 통과해야 한다.

- V1 final Quiz executable validity 100%
- turn memory의 evidence·timestamp·hash-chain 재현 가능
- Turn Quiz가 생성 시점 정보만으로 answerable함
- 사용자·조건·최신 상태·교정 정보가 gold memory에 보존됨
- 중간 Quiz가 특정 reasoning type이나 module에 과도하게 편중되지 않음

5개 pilot은 통계적 결론이 아니라 명백히 불리한 경로를 제거하고 상위 1~2개를 고르는
scale gate로 사용한다. 선택된 경로만 10개 이상으로 확대한다.

#### Hybrid 구현 경로

각 event를 다음 순서로 처리한다.

```text
이전 승인 memory
  + 현재 structured event
  -> 기존 Stage 3 방식으로 event dialogue 26~40 turns 일괄 생성
  -> turn 순서별 NO_OP/UPDATE(Patch), reason code, evidence turn 생성
  -> evidence turn까지의 prefix만으로 근거가 존재하는지 결정론적 검증
  -> Patch 순차 적용 및 memory hash-chain 저장
  -> 확정 memory를 다음 event 입력으로 전달
```

- labeler는 event 전체를 볼 수 있지만, 각 UPDATE의 evidence가 해당 turn prefix 안에
  실제 존재하지 않으면 future leakage로 거부하고 재시도한다.
- Patch는 현재 승인 memory에 결정론적으로 적용하며 실패 시 부분 반영하지 않는다.
- Turn Quiz는 주요 UPDATE 직후의 memory snapshot과 그 시점까지의 history만 사용한다.
- event별 dialogue·label·memory snapshot을 checkpoint하여 중단 후 이어서 실행한다.
- 호출별 input/cached/output tokens, latency, NO_OP/UPDATE 수와 retry를 기록한다.

주요 신규 구현은 Hybrid orchestration/label schema, prefix-evidence validator,
memory hash-chain 및 Turn Quiz cutoff validator이다. Dialogue 생성기, Tool schema,
VehicleWorld 실행 검증과 공개 V1 serializer는 기존 구현을 재사용한다.

### 5. 신규 100개 생성 및 split

통합 pilot이 통과하면 원 논문의 100개 후보 중 50개 선별 비율을 확장해 persona 후보
200개를 만들고 최종 scenario 100개를 선별한다. 이 배율 적용은 논문 사실이 아닌
`project_defined` 결정으로 기록한다. 이후 scenario 단위로 병렬 생성한다. 예시는
다음과 같다.

- Train 80개
- Development 10개
- V2 internal test 10개
- 기존 V1 50개: 기존 Cloud 결과와 비교하는 외부 test

같은 persona 또는 event chain의 변형은 반드시 동일 split에 묶어 leakage를 막는다.

## 구현 회차

1. **완료** — V1 schema·provenance manifest와 공개 50개 분석기
2. **완료** — Persona/event-chain 생성과 temporal interleaving
3. **구현/1개 canary 완료** — Dialogue·최종 Quiz 생성, simulator 검증과 V1 pilot
4. **완료** — Hybrid turn-wise memory·reason/evidence와 prefix-evidence 검증
5. **완료** — Immediate/Delayed/Composite Turn Quiz 30개와 최종 V1 Quiz 생성
6. **완료** — causal memory anchor를 포함한 Hybrid S1 전체 80-event 파일럿
7. 품질 dashboard·resume/controller 후 신규 100개 생성

## 4회차 산출물

- 생성 코드: `vehicle_bench/v2_hybrid.py`
- resumable 실행기: `run_vehiclemembench_v2_hybrid_smoke.py`
- event dialogue는 기존 Stage 3 생성기를 재사용하고 event 종료 전에 memory label 확정
- LLM은 structured preference update별 최초 evidence turn과 exact quote만 정렬
- memory 문장과 ADD/REPLACE/DELETE Patch는 structured event에서 결정론적으로 생성
- 모든 turn에 `NO_OP/UPDATE`, reason code, 한 문장 reason과 memory hash-chain 저장
- UPDATE evidence가 지정된 turn에 실제 존재하지 않으면 checkpoint 생성 거부
- preference update가 없는 event는 LLM을 호출하지 않고 결정론적 NO_OP 처리
- 호출별 input/cached/output token, latency와 generation attempt 저장

기존 Dialogue checkpoint를 사용한 offline 1-event smoke에서 40개 NO_OP label,
prefix-evidence 검증과 memory hash-chain을 통과했으며 동일 checkpoint 재실행 결과의
artifact hash도 일치했다. 실제 UPDATE event의 Terra evidence alignment와 Turn Quiz는
다음 pilot에서 검증한다.

이후 첫 UPDATE event(`v01-e4`)를 새 Dialogue로 세 번 smoke했다. v1/v2 prompt는
주체의 직접 확인보다 1~2 turn 늦은 기록·재진술을 evidence boundary로 고르는 경우가
있었다. v3에서는 다음 두 안전장치를 적용했다.

- 주체의 첫 확인·교정을 후속 recap/recording보다 우선하도록 alignment prompt 명시
- 조건 없는 명시적 값에 한해 더 이른 `subject + value + setting/context` 확인 turn이
  존재하면 결정론적으로 boundary를 앞당기고 normalization provenance 저장

새 v3 Dialogue에서는 Mara가 brightness 3을 처음 확인한 turn 4를 Terra가 직접
선택했고 25 NO_OP + 1 UPDATE, 1 ADD Patch와 memory hash-chain을 통과했다. 과거 v2
출력에 guard를 재적용했을 때도 turn 5에서 turn 4로 보수적으로 교정됐다. 조건부 또는
어휘적으로 모호한 update는 이 guard로 이동하지 않고 Terra alignment를 유지한다.

## 5회차 Immediate Turn Quiz 산출물

- 생성 코드: `vehicle_bench/v2_turn_quiz.py`
- resumable 실행기: `run_vehiclemembench_v2_turn_quiz_smoke.py`
- 모든 Hybrid UPDATE label을 Quiz checkpoint로 선택
- 정답 Tool/Arguments는 Stage 2 preference update에서 결정론적으로 파생
- Terra는 현재 memory와 해당 turn까지의 causal prefix로 자연스러운 query만 생성
- 정답 value는 query에서 숨기고 위치·상황 selector만 허용해 memory retrieval 강제
- exact memory evidence, prompt/source hash와 future cutoff 검증
- 공식 Tool schema 및 VehicleWorld 실행·state change 검증

첫 Terra query는 `brightness 3`을 질문에 노출해 자동검사 보강의 필요성을 확인했다.
v2에서는 값을 숨긴 `Please set the rear-left reading lamp to Mara's remembered setting`
형식으로 다시 생성했다. Gold call
`carcontrol_light_set_reading_light_brightness(light=rear_left, brightness=3)`은
simulator에서 성공하고 state를 변경했다. 생성량은 input 1,975, output 146 tokens,
provider latency 3.54초였다.

S1 전체 정성검사에서는 단순히 `speed 2를 시험했다`는 발화를 선호 확정으로 잡은
사례를 발견했다. Alignment v4는 `tried/tested/selected`만으로 UPDATE하지 않고 최초
효과 확인·수용 turn까지 기다린다. 동일 대화를 재정렬한 R2에서 해당 경계는 turn 4에서
turn 6으로 이동했다. Query v3는 causal prefix에 없는 사고·reset·변경 상태를 새로
만들지 못하게 했다.

최종 S1 Turn Quiz 구성은 다음과 같다.

- Immediate 11개: 각 UPDATE 확정 직후
- Delayed 11개: 같은 chain의 마지막 event 시점에서 재질문
- Composite 8개: Terra가 양립 가능한 두 memory fact를 계획하고 하나의 다중 Tool
  요청으로 생성
- 총 30개 Quiz, 38개 Tool call 모두 schema·simulator·state-change 검증 통과
- 동일 history의 최종 V1 Quiz 10개, 11개 Tool call도 모두 통과

## 1회차 산출물

- 코드: `vehicle_bench/v1_reproduction.py`
- 실행기: `analyze_vehiclemembench_v1_reproduction.py`
- schema: `scenario.schema.json`
- reference: `reference_profile.json`, `manifest.json`
- 공식 source: commit `5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b`
  및 dataset hash `d745b359794da1540431d05bcf6ea6ff4d75308dfa4f672e51875deb731cf88c`

공개 V1 50개에서 확인한 핵심 수치는 논문과 일치한다.

- history 평균: 2,690.36 turns / 92,818.86 `o200k_base` tokens
- query 평균: 37.628 tokens
- 총 50 scenario, 500 query, 572 gold Tool calls
- reasoning type: 149/102/97/90/62로 논문 Table 4와 일치

원본 출력의 speaker alias 5건과 중복 timestamp prefix 1건도 기록한다. 이는 논문이
정한 3인 persona/시간 정합성 규칙보다 우선하지 않으며, 신규 데이터에서는 validator가
차단한다.

## 2회차 산출물

- 생성 코드: `vehicle_bench/v1_generation.py`
- PersonaHub source/sampling: `vehicle_bench/v1_persona_source.py`
- seed sampling 실행기: `sample_personahub_elite_for_vehiclemembench.py`
- preflight 실행기: `prepare_vehiclemembench_v1_stage2.py`
- Terra canary 실행기: `run_vehiclemembench_v1_stage2_smoke.py`
- pinned PersonaHub revision: `16777e34bf5cb758b925cae5d84e868ee6c2100c`
- 추출 결과: 19 shard의 각 1 MiB 구간, 24,372개 후보 중 중복 없는 seed 600개
- 생성 후보 계획: 논문의 100→50 선별 비율을 확장한 200개 Persona group 후보

실제 `gpt-5.6-terra` canary 한 건의 자동 검증 결과는 다음과 같다.

- 3 personas, background chain 20개, vehicle chain 10개
- background event 40개 + vehicle event 41개 = 총 81 events
- 10개 vehicle attribute update, 전체 시간 범위 167.991일
- reasoning plan: preference conflict 4, error correction 3, coreference 2,
  conditional constraint 1
- Persona: input 1,356 / output 1,315 tokens
- Event chain: input 16,003 / output 7,547 tokens
- strict schema, 공식 Tool argument schema와 temporal contract 모두 통과

PersonaHub의 정확한 원 seed/sample 절차, 최소 80 events와 최소 14일 범위는 논문에
완전히 명시되지 않았으므로 `project_defined`로 기록한다. 반면 3인 Persona,
20+10 chains, 시간 순 interleaving과 다섯 reasoning type은 `paper_specified`이다.

## 3회차 산출물

- 생성 코드: `vehicle_bench/v1_stage3.py`
- resumable 병렬 실행기: `run_vehiclemembench_v1_stage3_smoke.py`
- event별 Dialogue와 chain별 Quiz checkpoint
- 최종 `stage3.json`, `benchmark/history/history_1.txt`,
  `benchmark/qa_data/qa_1.json`

실제 `gpt-5.6-terra` canary 한 건의 결과는 다음과 같다.

- 80 structured events → 2,642 human-only dialogue turns
- background 1,602 turns, vehicle 1,040 turns; event당 26~41 turns
- 정확히 3명의 Persona만 사용, event마다 최소 2명, 중복 dialogue line 0건
- 최종 Quiz 10개, gold Tool call 11개
- 10/10 Quiz의 모든 Tool call이 fresh VehicleWorld 실행에 성공하고 state를 변경
- 공개 V1 loader가 history와 10개 QA를 strict mode로 파싱
- history 88,504 `o200k_base` tokens로 공개 V1 범위 81,622~118,267 안에 위치
- query 평균 32.6 tokens로 공개 V1 scenario 평균 범위 29.3~88 안에 위치

Dialogue는 논문의 Figure 6에 따라 event별로 `현재 event + 같은 chain의 과거 event +
최근 global event + 직전 preference state`를 입력한다. event별 API 호출, vehicle event의
26-turn target, 각 dialogue turn이 source event timestamp를 상속하는 방식은 원 코드가
공개되지 않아 `project_defined`이다. Background event 최소 40 turns와 human-only,
vehicle preference의 자연스러운 간접 언급은 `paper_specified`이다.

Stage 3 사전검사에서 JSON schema만으로는 simulator enum/range를 보장할 수 없음을
확인했다. 따라서 Stage 2에도 실제 VehicleWorld 실행 검증과 event당 최소 2인 조건을
추가했다. 실패 canary는 덮어쓰지 않고 별도 version으로 보존했으며, 최종 Stage 2
canary는 80 events·30 chains·11 updates·203.448일 범위를 통과했다.

## 주요 위험

- 공개 논문은 prompt를 축약해 제공하고 공식 저장소에는 원본 생성 코드와 중간
  persona/event-chain artifact가 없다. 따라서 **정확 복제**가 아니라 공개 명세에 따른
  **재현 구현**으로 표현해야 한다.
- V1 출력만으로는 생성 원인과 수동교정 과정을 유일하게 역추론할 수 없다. 추정 규칙을
  원 논문의 사실처럼 기술하지 않고 provenance와 uncertainty를 공개한다.
- V1 final Quiz나 정답을 turn memory 생성에 사용하면 학습 label leakage가 된다.
- 생성 모델의 자체 점검만으로는 충분하지 않다. Tool schema validation, simulator 실행,
  temporal cutoff와 일부 Human Review를 함께 사용한다.

## 완료 산출물

- 재현 가능한 V1 generation package와 versioned prompts
- 신규 V1-compatible scenario 100개
- 동일 scenario의 turn-wise gold trajectory와 중간 Quiz
- 생성 provenance, 자동 검증, Human Review 및 simulator 결과
- on-device 학습용 Summary/Patch view와 기존 V1 50개 평가 adapter
