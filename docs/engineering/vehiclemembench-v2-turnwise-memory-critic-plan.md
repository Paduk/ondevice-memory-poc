# VehicleMemBench V2 Turn-wise Gold Memory 계획

상태: `기존 Critic 결과를 gold로 사용하지 않고 처음부터 재설계`

## 1. 목표

VehicleMemBench V1의 대화와 공식 정답을 이용해 각 대화 turn에 대해 다음을 만든다.

- 차량 메모리와 무관하면 `NO_OP`
- 새로운 차량 선호·설정·조건·교정이 있으면 `UPDATE`
- `UPDATE`라면 그 근거와 반영 후 canonical memory
- Turn-wise Summary용 전체 memory와 Combined Patch용 최소 operation

기존 R2 Summary/Combined trace는 정답이 아니라 **모델이 만든 후보**로만 사용한다.

## 2. 사용할 정답 단서

VehicleMemBench는 시나리오마다 20개의 비차량 background chain과 10개의
실행 가능한 vehicle-preference chain을 섞어 만들었다. 공식 QA에는 사람이 검수한
다음 정보가 남아 있다.

- `gold_memory`: 차량 선호가 생기거나 바뀐 사건, 사용자, 날짜와 발화
- `new_answer`: 실행해야 하는 차량 tool과 argument
- `reasoning_type`: preference conflict, conditional constraint, state shift 등
- 전체 history: 원래 turn 순서와 message ID

시나리오의 10개 QA에서 위 정보를 합쳐 **Vehicle Oracle Ledger**를 만든다.

```text
event_id
person
date/time
module / attribute / value
condition
reasoning_type
gold evidence phrase
source QA
```

이 ledger는 gold label 생성과 평가에만 사용한다. 학습·평가 대상 모델의 입력에는
`gold_memory`, query, gold tool call을 노출하지 않는다.

## 3. Turn-wise gold 생성 방법

### 3.1 Gold event와 history 정렬

각 ledger event를 history turn에 연결한다.

1. 날짜, 사용자, exact quote를 우선 사용한다.
2. exact match가 없으면 gold 표현과 현재·직전 turn을 의미적으로 대조한다.
3. 하나의 event가 여러 발화로 전개되면 실제 차량 사실을 처음 명시하거나 변경한
   turn만 `UPDATE`한다. 단순 동의·반복·설명은 `NO_OP`이다.
4. 차량 event에 연결되지 않은 background conversation은 `NO_OP`이다.
5. 정렬이 확실하지 않은 turn은 추측하지 않고 `REVIEW_REQUIRED`로 보낸다.

직전 문맥은 지시 대상과 대화 주제를 확인하는 용도로만 사용한다. 미래 turn이나
최종 query를 현재 turn의 근거로 사용하지 않는다.

### 3.2 차량 범위 gate

`UPDATE`는 아래 두 조건을 모두 만족해야 한다.

- 현재 turn 또는 causal context가 oracle ledger의 vehicle event와 연결됨
- 기억할 내용이 차량 module/attribute 또는 해당 설정의 사용자·조건·시간 상태로
  설명 가능함

일반 생활, 직업, 취미, 인간관계는 차량 설정에 직접 영향을 주지 않으면 제외한다.
예를 들어 망원경, 반려견, 노트북, 건강보험, 배관, 산업용 보일러와 같은 내용은
durable하더라도 차량 memory가 아니다.

### 3.3 Canonical trajectory 생성

turn을 처음부터 순서대로 처리한다.

```text
이전 canonical memory
+ 현재 turn과 필요한 과거 문맥
+ 정렬된 oracle event 또는 background 판정
+ 기존 Trace 후보(참고용)
-> gold NO_OP/UPDATE 결정
-> canonical memory 갱신
-> 다음 turn으로 전달
```

- Summary `NO_OP`: 이전 memory를 그대로 전달한다.
- Summary `UPDATE`: 이전의 유효한 내용을 보존한 전체 memory를 만든다.
- Combined `NO_OP`: `operations=[]`로 전달한다.
- Combined `UPDATE`: 현재 canonical memory에 exact·atomic apply 가능한 최소
  `add/replace/delete`만 만든다.
- 두 방식의 표현은 달라도 최종 사실 집합은 같아야 한다.

기존 후보와 비교한 결과는 별도 metadata로만 기록한다.

- `ACCEPT`: 후보의 결정과 의미가 gold와 동일
- `CORRECT`: 후보의 결정 또는 사실을 실질적으로 수정
- `REBASE_ONLY`: 사실은 맞지만 앞선 canonical 교정을 보존하기 위해 재적용
- `FORMAT_ONLY`: 표현·순서만 다름

이 구분을 통해 단순 rebase를 실제 오류인 `CORRECT`로 세지 않는다.

## 4. Label 형식

각 turn label에는 최소 다음을 저장한다.

```json
{
  "scenario": 1,
  "turn_index": 96,
  "message_id": 97,
  "scope": "VEHICLE",
  "decision": "UPDATE",
  "reason_code": "NEW_VEHICLE_PREFERENCE",
  "reason": "Gary explicitly prefers a green instrument panel.",
  "evidence": {"message_id": 97, "quote": "I really love this green instrument panel."},
  "oracle_event_id": "s01-instrument-panel-color-01",
  "module": "InstrumentPanel",
  "attribute": "color",
  "condition": null,
  "candidate_assessment": "CORRECT"
}
```

`NO_OP`도 한 문장 reason과 다음 중 하나의 code를 갖는다.

```text
BACKGROUND_NON_VEHICLE
NO_NEW_VEHICLE_FACT
DUPLICATE_ALREADY_STORED
TRANSIENT_WITHOUT_MEMORY_VALUE
INSUFFICIENT_EVIDENCE
```

Summary label에는 `next_memory`, Combined label에는 `operations`와 deterministic
적용 결과를 추가한다.

## 5. 검수와 품질 기준

모델은 정렬과 memory 표현을 도울 수 있지만 oracle의 차량 범위를 임의로 넓힐 수
없다. 모호한 경우에만 강한 모델 또는 사람이 검수한다.

완료 전에 다음을 확인한다.

- 모든 `UPDATE`가 oracle event와 연결됨
- 모든 evidence가 history 원문에 exact match함
- query나 미래 turn을 근거로 사용하지 않음
- background chain이 canonical memory에 들어가지 않음
- state shift와 error correction에서 최신 유효 상태가 유지됨
- Summary와 Combined의 최종 사실 집합이 동일함
- Patch가 exact·atomic apply되고 replay 결과가 동일함
- `REVIEW_REQUIRED`가 모두 해결됨

Scenario 1에서는 전체 UPDATE와 REVIEW_REQUIRED를 사람이 확인한다. 이후
S1–S50 확장 시에는 scenario별 전수 자동 검증과 위험 사례 표본 검수를 병행한다.

## 6. Data leakage 방지

Oracle-assisted label은 학습용 teacher label 또는 평가 정답이며, autonomous memory
방법론의 성능 결과가 아니다. 같은 시나리오에서 oracle memory를 모델 입력으로
사용한 Quiz 결과는 상한선·label 품질 확인 용도로만 보고한다.

On-device 실험은 scenario 단위로 train/eval을 분리한다.

- Train 30 scenarios: gold turn labels로 SFT
- Eval 20 scenarios: 원본 과거 대화만 모델에 제공
- Eval oracle label: 채점과 오류 분석에만 사용

split은 reasoning type과 tool module 분포를 확인한 고정 manifest로 관리하고,
평가 시나리오의 gold memory·query·tool answer가 학습 prompt에 들어가지 않았는지
검사한다.

## 7. 산출물

기존 R2 결과는 보존하고 새 root를 사용한다.

```text
evaluation/vehiclemembench-v2/turnwise-memory-gold-v1/
  split_manifest.json
  scenario-XX/
    oracle_ledger.json
    turn_alignment.jsonl
    summary_labels.jsonl
    combined_labels.jsonl
    review_queue.jsonl
    audit.json
    COMPLETED
```

모든 파일에는 dataset hash, source history/QA hash, generator와 prompt/schema
version을 기록한다. 실행은 checkpoint/resume 가능해야 하며 원본 dataset과 기존
trace는 수정하지 않는다.

## 8. 구현 순서

1. **Oracle 추출:** QA의 gold memory, tool call, reasoning type을 ledger로 변환
2. **Turn 정렬:** ledger event를 history message ID에 연결하고 불확실 항목 분리
3. **S1 gold 생성:** Summary와 Combined canonical trajectory를 빈 memory부터 생성
4. **S1 전수 검수:** 기존 Trace·기존 Critic과 비교하고 Quiz로 label 유용성 확인
5. **S1–S50 확장:** 고정 split과 동일 schema로 전체 label 생성
6. **On-device 데이터화:** Train split만 Summary/Patch SFT example로 변환

첫 구현의 성공 기준은 Scenario 1에서 background 오기억이 0건이고, 모든 UPDATE가
공식 vehicle event에 연결되며, Summary와 Combined가 동일한 canonical 사실을
재현하는 것이다.
