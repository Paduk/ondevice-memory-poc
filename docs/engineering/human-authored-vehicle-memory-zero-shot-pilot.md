# Human-authored 차량 메모리 Zero-shot 평가 파일럿

## 1. 목적과 지위

이 파일럿은 VehicleMemBench V2의 합성 데이터 한계를 보완할 별도
`Human-authored Vehicle Memory Transfer Set`의 작성·변환·검수 절차를 검증한다.

- 기존 Summary, Patch, Delta-v3 체크포인트를 수정하지 않고 zero-shot으로 평가한다.
- 동일한 인간 작성 대화를 모든 방법과 모델에 replay한다.
- 평가용 Gold memory와 Gold Tool call은 인간 annotator가 작성하고 독립 검수한다.
- `HVP01–HVP20`은 외부 원문을 모델이 조립·변환한 파일럿이므로 **최종 human test에
  포함하지 않는다.** 기존 모델 작성 HVP01·02는 각 디렉터리의 `model-authored-v0/`에
  보존했다.
- 최종 데이터의 올바른 명칭은 `human-authored controlled transfer set`이다. 실제 차량
  운행 로그를 수집한 것이 아니므로 `real-world driving log`라고 부르지 않는다.

## 2. 외부 자료의 사용 범위

외부 자료의 문장을 복사하거나 번역해 최종 대화로 사용하지 않는다. 자료별로 다음 설계
요소만 참고했다.

| 자료 | 참고한 요소 | 최종 평가 데이터로 직접 사용하지 않는 이유 |
|---|---|---|
| [CAR-Bench](https://huggingface.co/datasets/johanneskirmayr/car-bench-dataset) | persona, initial context, gold action을 분리한 scenario card | LLM-simulated user를 사용하는 합성 benchmark |
| [Audio2Tool](https://huggingface.co/datasets/RVtech/Audio2Tool) | implicit, correction, distraction, multi-turn 유형과 smart-car Tool taxonomy | 발화·음성이 합성되어 human-authored 검증셋이 아님 |
| [KVRET](https://arxiv.org/abs/1705.05414) | in-car assistant를 위한 인간–인간 Wizard-of-Oz 수집 방식 | calendar, weather, POI navigation 중심이라 차량 설정 갱신 범위가 부족함 |
| [AI-Hub 차량 내 대화 및 명령어 음성](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=112) | 차량 안에서 명령과 일상 발화가 섞이는 수집 환경 | 한국어 read-speech 중심이며 현재 영문 V2와 언어·과제가 다름 |

외부 자료는 `무슨 상황과 기능을 포함할지`를 정하는 seed다. 최종 문장과 대화 흐름은
참가자가 직접 작성해야 한다. 외부 합성 대화를 사람이 조금 고치는 방식은
`human-refined synthetic`일 뿐 `human-authored`라고 주장하지 않는다.

## 3. 외부 대화 차용 파일럿 HVP01

HVP01은 HVP03–HVP20과 같은 external-adapted 규격으로 다시 만들었다. 서로 겹치지 않는
Envisioned Voice Assistant Dialogues 25 Turn과 Audio2Tool 25 Turn을 조립하고, 차량 설정
발화 10개에만 최소 지속성 문장을 붙였다. Devon의 좌석 환기, 음악 볼륨, 프런트·리어
트렁크, 공조 모드, 창문, 순환 모드를 다룬다.

### 확정 통계

| 항목 | 값 |
|---|---:|
| 화자 / 세션 | 2명 / 16개 |
| 전체 Turn | 50 |
| 인간 작성 / 합성 원문 | 25 / 25 |
| UPDATE / NO_OP | 10 / 40 |
| ADD / REPLACE | 7 / 3 |
| 최종 활성 memory slot | 7 |
| Memory Tool | 7종 |
| Turn / Final Quiz | 4 / 6 |
| Gold Tool call | 12개 |

3개의 REPLACE는 같은 owner·Tool attribute·selector·condition을 유지한다.

- 일반 음악 볼륨 `11 → 58`
- 일반 공조 모드 `auto → defrost`
- 식당 배기 냄새 조건의 순환 모드 `inside → outside`

## 4. 산출물

기준 디렉터리:

```text
evaluation/human-authored-vehicle-memory/pilot-hv01/
  scenario-source.json
  derived/
    dialogue.jsonl
    patch.jsonl
    memory_snapshots.jsonl
    turn_quiz.jsonl
    final_quiz.jsonl
    manifest.json
```

- `scenario-source.json`: 사람이 읽고 작성할 수 있는 canonical source다. persona,
  session context, dialogue, update annotation, Quiz를 한곳에 둔다.
- `dialogue.jsonl`: 모델 입력으로 replay할 label-free 원문이다.
- `patch.jsonl`: 현재 V2 grouped Patch SFT row와 같은 핵심 계약을 사용한다. 파일럿은
  `split=pilot_excluded`, `train_eligible=false`로 강제한다.
- `memory_snapshots.jsonl`: 매 Turn 이후의 전체 Gold memory와 SHA-256을 보존한다.
- `turn_quiz.jsonl`: 이전 값과 갱신된 값을 각각 해당 시점에서 물을 수 있다.
- `final_quiz.jsonl`: 마지막 활성 값만 참조하는 zero-shot 최종 평가 문제다.
- `manifest.json`: 통계, 파일 hash, simulator 검증 결과를 기록한다.

재생성 명령:

```bash
cd /home/hj153lee/PalmClaw
PYTHONPATH=/home/hj153lee/PalmClaw/ubuntu/src \
  /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python \
  -m memory_training.scripts.build_human_authored_pilot \
  --source evaluation/human-authored-vehicle-memory/pilot-hv01/scenario-source.json \
  --output-dir evaluation/human-authored-vehicle-memory/pilot-hv01/derived
```

## 5. Gold 작성 규칙

### UPDATE

다음 조건을 모두 만족할 때만 UPDATE로 지정한다.

1. 발화자가 자기 선호를 말했거나 차량 공통 규칙을 명시적으로 합의했다.
2. 값, 적용 대상과 조건이 현재 또는 바로 앞 문맥에서 하나로 확정된다.
3. 일회성 명령이 아니라 이후에도 재사용하려는 의도가 있다.
4. REPLACE라면 같은 `owner + Tool attribute + selector + condition`의 기존 활성 값이 있다.

`좋네`, `지금은 따뜻하다`, `한번 두고 보자`처럼 지속성이 확정되지 않은 평가는 저장하지
않는다. 다른 사람의 설정을 추측해서 그 사람의 memory로 저장해서도 안 된다.

### Quiz

- 질문에 정답 수치나 enum을 직접 노출하지 않는다.
- cutoff 시점에 활성인 memory만 정답 근거로 사용한다.
- REPLACE 이후 Final Quiz는 폐기된 과거 값을 정답 후보로 포함하지 않는다.
- owner, seat/zone, condition을 질문에서 식별할 수 있어야 한다.
- 하나의 사람이 여러 설정을 요청하거나 서로 다른 사람의 zone을 함께 적용하는 composite
  Quiz를 일부 포함한다.
- Gold Tool call은 공식 schema와 VehicleWorld simulator를 모두 통과해야 한다.

## 6. 실제 20개 시나리오 수집 절차

### 역할 분리

1. **Scenario designer**: 차량 기능과 값 범위로 자연어 goal card를 만든다. Tool 함수명과
   내부 memory 형식은 작성자에게 보여주지 않는다.
2. **Dialogue author**: goal card만 보고 3–5개 세션의 대화를 새로 쓴다. 기존 V2 예문을
   보거나 고쳐 쓰지 않는다.
3. **Naturalness reviewer**: Gold를 보지 않고 말투, 문맥, 사용자 구분과 과도한 정형성을
   검수한다.
4. **Annotator A/B**: 서로 독립적으로 UPDATE 시점, owner, ADD/REPLACE, 활성 값과 Gold
   Tool call을 작성한다.
5. **Adjudicator**: 두 annotation의 불일치만 원문 근거로 확정한다.

한 사람이 dialogue와 최종 Gold를 모두 확정하지 않는다. `기억해 둬` 같은 문구도 모든
UPDATE에 반복하지 말고, 명시적 합의·반복 사용·이전 경험 비교 등 여러 자연스러운
표현으로 지속성을 드러낸다.

### 목표 규모

20개 시나리오 각각에 다음 최소치를 적용한다.

- 사람 2–3명, session 3–5개
- 최종 활성 memory 6–10개
- REPLACE 2–4개
- 동일 slot의 이전 evidence와 새 evidence 사이에 최소 한 session 간격
- Turn/Final Quiz 합계 8–10개
- 공조, 좌석, 조명, 내비게이션, 창문·미러, 미디어 중 최소 3개 domain

NO_OP filler는 UPDATE core가 승인된 뒤 길이·방해 강도별로 추가한다. filler를 추가해도
UPDATE evidence와 cutoff는 변경하지 않는다.

## 7. 승인 기준

각 시나리오는 다음 gate를 모두 통과해야 한다.

| Gate | 판정 방법 |
|---|---|
| 인간 원문성 | 작성 이력과 동의서 확인; LLM 생성·번역·paraphrase 금지 |
| 자연스러움 | 독립 reviewer PASS |
| UPDATE grounding | 두 annotator가 evidence Turn을 지정하고 adjudication 완료 |
| 상태 전이 | ADD/REPLACE를 처음부터 replay하여 모든 snapshot hash 일치 |
| Quiz answerability | 질문마다 활성 Gold line이 cutoff memory에 존재 |
| Tool 유효성 | 공식 JSON Schema 및 VehicleWorld simulator PASS |
| 데이터 격리 | 기존 학습·검증·checkpoint 선택에 미사용 |

## 8. 파일럿에서 드러난 한계

HVP01–HVP20은 파이프라인을 검증하기에는 충분하지만 실제 human-authored 성능 근거는
아니다.

- 인간 참가자가 작성한 원문은 각 시나리오의 25 Turn뿐이며, 나머지 원문과 지속성 문장,
  Gold label, Quiz 및 시나리오 조립은 합성·모델 작성이다.
- 각 시나리오는 50 Turn이지만 여러 독립 source record를 이어 붙인 구성이다.
- 영어 text-only이며 실제 차량 음성, ASR 오류, 소음은 다루지 않는다.
- delete, temporary override, 모호한 owner는 포함하지 않았다.
- 실제 인간 데이터에서는 참가자가 대화 전체와 지속성 표현을 직접 작성하고 별도
  annotator가 Gold를 확정해야 한다.

따라서 이 파일럿은 **사람이 만들 20개 시나리오의 형식과 검수 기준**으로만 사용한다.

## 9. 외부 대화 차용 파일럿 HVP02

HVP02도 HVP03–HVP20과 같은 external-adapted 규격으로 다시 만들었다. Robin의 음악
볼륨, rear-right 온도, 조수석 환기, 프런트·리어 트렁크, 도어, 내비게이션 음성을
다루며 음악 볼륨의 연속 갱신 `52 → 25 → 70 → 5`를 포함한다.

| 항목 | 값 |
|---|---:|
| 화자 / 세션 | 2명 / 16개 |
| 전체 Turn | 50 |
| 인간 작성 / 합성 원문 | 25 / 25 |
| UPDATE / NO_OP | 10 / 40 |
| ADD / REPLACE | 7 / 3 |
| 최종 활성 memory slot | 7 |
| Turn / Final Quiz | 4 / 6 |
| Gold Tool call | 12개 |

네 개의 Turn Quiz는 초기 활성값을 해당 cutoff에서 묻고, 여섯 개의 Final Quiz는 최종
활성값만 참조한다. 원래 모델 작성 HVP02와 산출물은 `model-authored-v0/`에 보존했다.

```text
evaluation/human-authored-vehicle-memory/pilot-hv02/
  scenario-source.json
  derived/
    dialogue.jsonl
    patch.jsonl
    memory_snapshots.jsonl
    turn_quiz.jsonl
    final_quiz.jsonl
    manifest.json
```

## 10. 외부 대화 차용 파일럿 HVP03

HVP03은 실제 human test를 대신하지 않는다. 라이선스가 확인된 외부 대화를 가능한 한
그대로 V2 형식에 이식할 수 있는지 보는 **external-adapted pilot**이다.

| 항목 | 값 |
|---|---:|
| 전체 Turn | 50 |
| 외부 source trace 보유 | 50 (100%) |
| 인간 작성 원문 / 합성 원문 | 25 / 25 |
| 원문 그대로 / 최소 지속성 변형 | 40 / 10 |
| UPDATE / NO_OP | 10 / 40 |
| ADD / REPLACE | 7 / 3 |
| Turn / Final Quiz | 4 / 6 |

인간 작성 부분은 Envisioned Voice Assistant Dialogues의 세 레코드에서 가져왔다. 차량
Tool과 직접 연결되는 부분은 Audio2Tool의 smart-car 레코드를 차용했지만, 해당 query
text는 LLM 생성이므로 인간 작성으로 세지 않는다. 일회성 명령을 지속 memory로 오인하지
않도록 UPDATE가 되는 10개 발화만 지속성 조건을 최소 추가했으며, 원문도 각 turn의
`source_trace.original_text`에 함께 보존했다.

출처, 라이선스, 사용 record ID와 변형 규칙은
`evaluation/human-authored-vehicle-memory/pilot-hv03/SOURCE_ATTRIBUTION.md`에 기록한다.
특히 Audio2Tool은 CC BY-NC 4.0이므로 이 파일럿의 이용 범위도 비상업적 연구로 제한한다.

## 11. 독립 외부 대화 차용 파일럿 HVP04

HVP04는 HVP03과 동일한 규칙으로 만들되, 사용한 외부 record를 완전히 분리한 두 번째
external-adapted pilot이다.

| 항목 | 값 |
|---|---:|
| 전체 Turn | 50 |
| 외부 source trace 보유 | 50 (100%) |
| 인간 작성 원문 / 합성 원문 | 25 / 25 |
| 원문 그대로 / 최소 지속성 변형 | 40 / 10 |
| UPDATE / NO_OP | 10 / 40 |
| ADD / REPLACE | 7 / 3 |
| Turn / Final Quiz | 4 / 6 |

길찾기, 주유, 영화 예약과 여정 알림의 인간 작성 대화를 NO_OP 문맥으로 보존하고,
Audio2Tool의 온도, 열선, 외기, 창문, 트렁크와 correction 발화를 VehicleMemBench Tool에
매핑했다. HVP03과 HVP04 사이의 source record 교집합은 0개다. 출처별 세부 record와
라이선스는 `evaluation/human-authored-vehicle-memory/pilot-hv04/SOURCE_ATTRIBUTION.md`에
기록한다.

## 12. 추가 독립 외부 대화 파일럿 HVP05–HVP07

HVP05–HVP07은 동일한 external-adapted 규칙으로 만든 세 개의 추가 시나리오다. 세
시나리오와 HVP03–HVP04를 합친 250 Turn 전체가 서로 다른 외부 record를 사용한다.

| 시나리오 | Turn | 인간/합성 | 원문/최소 변형 | UPDATE/NO_OP | ADD/REPLACE | 활성 slot |
|---|---:|---:|---:|---:|---:|---:|
| HVP05 | 50 | 25/25 | 43/7 | 7/43 | 5/2 | 5 |
| HVP06 | 50 | 25/25 | 43/7 | 7/43 | 4/3 | 4 |
| HVP07 | 50 | 25/25 | 43/7 | 7/43 | 5/2 | 5 |

각 시나리오는 Turn Quiz 4개와 Final Quiz 6개를 포함한다. 전체 150 Turn의 기록된
`original_text`를 다운로드 원문과 다시 대조한 결과 불일치가 없었으며, HVP03–HVP07
간 source record 교집합도 없다. 이 파일들은 외부 언어 전이 파일럿일 뿐, 최종 인간
작성 평가셋으로 간주하지 않는다.

HVP03–HVP07 전체 집계는 UPDATE 41 / NO_OP 209로, 목표로 삼은 약 1:5 비율을
유지한다.

재현 가능한 생성 코드는
`memory_training/scripts/build_external_adapted_pilots.py`에 있으며 각 디렉터리의
`SOURCE_ATTRIBUTION.md`에 사용 record와 라이선스를 기록한다.

## 13. 결정론적 확장 HVP08–HVP12

HVP08–HVP12는 동일한 원칙을 고정한 추가 다섯 시나리오다. 각 시나리오는 외부 원문
50 Turn(인간 작성 25, 합성 25), UPDATE 10 / NO_OP 40, ADD 7 / REPLACE 3,
Turn Quiz 4 / Final Quiz 6으로 구성한다.

| 시나리오 | Turn | 인간/합성 | 원문/최소 변형 | UPDATE/NO_OP | ADD/REPLACE | 활성 slot |
|---|---:|---:|---:|---:|---:|---:|
| HVP08 | 50 | 25/25 | 40/10 | 10/40 | 7/3 | 7 |
| HVP09 | 50 | 25/25 | 40/10 | 10/40 | 7/3 | 7 |
| HVP10 | 50 | 25/25 | 40/10 | 10/40 | 7/3 | 7 |
| HVP11 | 50 | 25/25 | 40/10 | 10/40 | 7/3 | 7 |
| HVP12 | 50 | 25/25 | 40/10 | 10/40 | 7/3 | 7 |

UPDATE 값은 Audio2Tool의 원래 정답 호출에서 결정론적으로 가져오며, 원문에는 지속성·조건
문장만 덧붙인다. 색상처럼 Tool ontology가 다른 경우 의미가 같은 VehicleMemBench enum으로
정규화한다. 각 source record는 HVP03–HVP12의 다른 시나리오와 중복되지 않는다.

HVP03–HVP12 전체 500 Turn을 원본 파일과 대조한 결과 `original_text` 불일치 0건,
source record 교집합 0건이다. 전체 UPDATE / NO_OP는 91 / 409(약 1:4.49)이다.
모든 시나리오는 공식 Tool JSON schema, VehicleWorld 실행, Patch replay, NO_OP identity,
Quiz cutoff 검증을 통과했다.

## 14. 결정론적 확장 HVP13–HVP16

HVP13–HVP16도 HVP08–HVP12와 같은 고정 규칙으로 생성했다. 각 시나리오는 서로 겹치지
않는 외부 원문 50 Turn, 인간 작성/합성 원문 25/25, UPDATE/NO_OP 10/40,
ADD/REPLACE 7/3, Turn/Final Quiz 4/6으로 구성한다.

| 시나리오 | Turn | 인간/합성 | 원문/최소 변형 | UPDATE/NO_OP | ADD/REPLACE | 활성 slot |
|---|---:|---:|---:|---:|---:|---:|
| HVP13 | 50 | 25/25 | 40/10 | 10/40 | 7/3 | 7 |
| HVP14 | 50 | 25/25 | 40/10 | 10/40 | 7/3 | 7 |
| HVP15 | 50 | 25/25 | 40/10 | 10/40 | 7/3 | 7 |
| HVP16 | 50 | 25/25 | 40/10 | 10/40 | 7/3 | 7 |

HVP13은 팬·창문·온도·defrost·안내 음성·재생·핸들 열선, HVP14는 선루프·목적지·
트렁크·안내 음성·defrost·핸들 열선, HVP15는 순환 모드·팬·온도·안내 음성·재생·
핸들 열선·트렁크, HVP16은 순환 모드·선루프·안내 음성·재생·defrost·창문·핸들
열선을 다룬다. 모든 REPLACE는 같은 slot, Tool, context arguments, activation condition을
유지하고 값만 시간순으로 변경한다.

HVP03–HVP16 전체 700 Turn을 다운로드 원문과 대조한 결과 `original_text` 불일치 0건,
212개 source record의 시나리오 간 교집합 0건이다. 전체 UPDATE/NO_OP는
131/569(약 1:4.34), ADD/REPLACE는 91/40이다. HVP13–HVP16은 공식 Tool JSON schema,
VehicleWorld 실행, Patch replay, NO_OP identity, Quiz evidence cutoff 등 8개 자동 검증을
모두 통과했다. 이 네 시나리오 역시 external-adapted pilot이며 실제 human-authored test로
간주하지 않는다.

## 15. 결정론적 확장 HVP17–HVP20

HVP17–HVP20도 같은 외부 원문 차용 및 결정론적 변환 규칙을 적용했다. 각 시나리오는
서로 겹치지 않는 외부 원문 50 Turn, 인간 작성/합성 원문 25/25, UPDATE/NO_OP 10/40,
ADD/REPLACE 7/3, Turn/Final Quiz 4/6으로 구성한다.

| 시나리오 | Turn | 인간/합성 | 원문/최소 변형 | UPDATE/NO_OP | ADD/REPLACE | 활성 slot |
|---|---:|---:|---:|---:|---:|---:|
| HVP17 | 50 | 25/25 | 40/10 | 10/40 | 7/3 | 7 |
| HVP18 | 50 | 25/25 | 40/10 | 10/40 | 7/3 | 7 |
| HVP19 | 50 | 25/25 | 40/10 | 10/40 | 7/3 | 7 |
| HVP20 | 50 | 25/25 | 40/10 | 10/40 | 7/3 | 7 |

HVP17은 조명·온도·창문·목적지·defrost·순환 모드·트렁크, HVP18은 조명·좌석
환기·온도·순환 모드·프런트 트렁크·선루프·defrost, HVP19는 조명·좌석 환기·순환
모드·defrost·프런트 트렁크·도어 잠금·라디오, HVP20은 온도·조명·목적지·안내
음성·defrost·순환 모드·도어 잠금을 다룬다. 기본 VehicleWorld 상태와 같은 Gold
호출은 단독 Quiz로 두지 않고 다른 활성 설정과 결합하여, 모든 Quiz가 실행 결과로도
검증되도록 했다.

HVP01–HVP20 전체는 1,000 Turn, 308개 source record이며 시나리오 간 source record
교집합은 0개다. 전체 UPDATE/NO_OP는 191/809(약 1:4.24), ADD/REPLACE는 133/58이다.
각 Turn의 `original_text`는 다운로드 원문과 대조하고, 생성 결과는 공식 Tool JSON
schema, VehicleWorld 실행, Patch replay, NO_OP identity, Quiz evidence cutoff 등 8개
자동 검증을 통과해야 한다. 이 네 시나리오도 외부 차용 파일럿이므로 실제 인간 작성
평가셋으로 간주하지 않는다.
