# Recursive Summarization 병목 분석

상태: `Scenario 6–10, 50 task semantic audit 완료`

분석일: 2026-07-30

## 1. 범위와 판정 기준

분석 대상은 다음 Recursive Summarization 평가 artifact다.

- [Scenario 6–10 결과](../../ubuntu/evaluation/vehiclemembench/29518f79-654d-4da0-865a-17ca633b6e70/results.md)
- [50-task case](../../ubuntu/evaluation/vehiclemembench/29518f79-654d-4da0-865a-17ca633b6e70/cases.jsonl)

VehicleMemBench의 `gold_memory`, Gold call, 최종 Recursive Summary, 모든
중간 summary version을 task 단위로 대조했다. Ours의 `record_absent`와
비교할 수 있도록 다음 기준을 사용했다.

- `sufficient`: 사용자·조건·설정·값이 최종 요약에 있어 정답 실행을
  결정할 수 있음
- `never_stored`: 원본 History에는 있으나 올바른 canonical fact가 어떤
  summary version에도 만들어지지 않음
- `lost`: 올바른 fact가 중간 summary에는 있었으나 이후 rewrite에서 사라짐
- `distorted`: 값, 사용자 identity, 장치·predicate가 잘못 귀속되거나
  다른 fact와 합쳐짐

Gold Tool의 부가 동작까지 summary에 문장으로 있어야 한다고 요구하지는
않았다. 예를 들어 child-lock 선호가 요약에 있으면 door/window의 정확한
Tool scope는 Memory 이후 실행 변환 문제로 판정했다.

이 분석은 reviewed semantic audit이며 아직 Gold fact를 Recursive Summary에
주입해 재평가한 causal Oracle 실험은 아니다.

## 2. 핵심 결과

| 최종 summary 상태 | Task | 비율 | ESM 성공 |
| --- | ---: | ---: | ---: |
| 정답 근거 충분 | 34 | 68% | 28/34, 82.4% |
| 최초 올바른 저장 실패 | 9 | 18% | 1/9 |
| 중간 저장 후 소실 | 2 | 4% | 0/2 |
| 오귀속·왜곡 | 5 | 10% | 2/5 |
| 합계 | 50 | 100% | 31/50, 62% |

최종 요약에서 정답 근거가 불충분한 task는 총 `16/50`이며, 서로 공유하는
Fact를 합치면 14개 고유 Gold Fact다. 이 16 task 중 ESM 성공은 3건뿐이고,
정답 근거가 충분한 34 task에서는 28건이 성공했다.

전체 ESM 실패 19건 중 13건은 최종 summary의 누락·소실·왜곡과 함께
발생했다. 나머지 6건은 필요한 Memory가 최종 요약에 있었지만 Agent의
Tool·argument·복합 호출 변환에서 실패했다.

따라서 Recursive Summary에서도 Memory coverage가 가장 큰 병목이다.
다만 모든 13건이 summary만 보완하면 바로 복구된다는 인과관계는 아직
검증하지 않았다.

## 3. 최초 저장 실패

다음 9 task는 원본 History에 정답 근거가 있지만 올바른 canonical fact가
어떤 summary version에도 나타나지 않았다.

| Task | 누락된 정답 근거 | 평가 결과 |
| --- | --- | --- |
| `vehicle-06-01` | Samuel, 밝은 낮 center display brightness 90 | 실패 |
| `vehicle-06-05` | Michael, field video quality 480p | 실패 |
| `vehicle-06-07` | Samuel, standard AC temperature 21 | 실패 |
| `vehicle-07-03` | Brian, installation ambient color cyan | 실패 |
| `vehicle-07-05` | David, center display language English | 실패 |
| `vehicle-08-03` | Jeffrey, back pain 시 driver massage enabled | 성공 |
| `vehicle-09-00` | Christopher, focus ambient color green | 실패 |
| `vehicle-09-09` | Susan, local news radio station | 실패 |
| `vehicle-10-07` | Jack, driving massage level 1 | 실패 |

`vehicle-08-03`은 최종 요약에 Jeffrey가 massage를 껐고 이전 설정이 너무
강했다는 흔적만 있다. `back_pain → enabled`라는 정답 Fact는 없지만,
Agent가 질의의 “back pain is back”과 “restore”를 이용해 성공했다.

`vehicle-10-07`의 massage level 1은 올바른 장치에 저장되지 않았고,
기존 steering-wheel heating Fact의 값을 덮어쓰는 데 사용됐다. 이는
단순 누락과 cross-fact 오염이 동시에 발생한 사례다.

## 4. Recursive rewrite 소실

Scenario 7에서 Jacob의 야간 overhead-screen brightness 1은 summary
version 5에 정확히 존재했지만 version 6부터 사라졌다. 이후 History에
동일 Fact가 다시 등장하지 않아 최종 요약에서 복구되지 않았다.

이 한 개의 소실 Fact가 다음 2 task에 영향을 줬다.

- `vehicle-07-08`: level 1 복원 실패
- `vehicle-07-09`: 저장값 1에 1을 더한 level 2 계산 불가

최종 summary 길이는 모든 Scenario에서 8,192-character 제한보다 훨씬
작았고 truncation은 0회였다. 따라서 이 소실은 hard limit에 의한 절단이
아니라 LLM recursive rewrite 중의 망각이다.

## 5. 오귀속·왜곡

| Task | 중간·최종 summary 문제 | 평가 결과 |
| --- | --- | --- |
| `vehicle-07-06`, `vehicle-07-07` | Jacob의 hot-weather AC 18을 Brian에게 귀속 | 두 task 성공 |
| `vehicle-09-01` | Eric의 `instrumentPanel.theme=map`을 일반적인 topographic map 화면으로 압축하고 `scene` theme은 유지 | 실패 |
| `vehicle-09-06` | Susan mirror position 84가 version 7–10에 있었으나 version 11에서 Christopher의 60을 Susan의 최신값으로 잘못 적용 | 실패 |
| `vehicle-10-04` | Jack steering-wheel heat level 2가 version 11–12에 있었으나 version 13에서 massage level 1 발화를 steering-wheel 갱신으로 오해 | 실패 |

Scenario 7의 두 AC task는 identity가 틀렸지만 값 18과 hot-weather 조건이
요약에 남아 있어 Agent가 우연히 올바른 값을 사용했다. 따라서 ESM 성공만
보면 summary의 identity corruption을 놓치게 된다.

Scenario 9와 10은 새 대화가 기존 Fact의 같은 숫자 필드처럼 보일 때
다른 사용자나 장치의 Fact를 덮어쓰는 위험을 보여준다. Recursive Summary는
개별 record key와 evidence provenance가 없어서 이런 오염을 갱신 시점에
기계적으로 차단하기 어렵다.

## 6. Memory 이후 실행 변환 병목

정답 근거가 최종 summary에 충분했지만 ESM이 실패한 task는 6건이다.

| Task | 실행 변환 문제 |
| --- | --- |
| `vehicle-07-00` | voice mode mute는 맞았지만 불필요한 navigation 호출 추가 |
| `vehicle-09-04` | door lock은 맞았으나 window child-lock scope를 `all` 대신 `rear`로 변환 |
| `vehicle-09-08` | Eric 78에서 Susan이 40을 낮춘다는 Memory를 38로 계산하지 않고 78 사용 |
| `vehicle-10-02` | Brenda의 location 조건부 air/window 선호를 질의의 coreference에 적용하지 못함 |
| `vehicle-10-05` | `feet_window`를 먼저 잘못된 enum으로 호출하고 power·defrost 등 불필요한 상태까지 변경 |
| `vehicle-10-08` | Christine의 English map-language 선호를 center display language Tool로 연결하지 못함 |

이 구간은 Ours에서 확인한 late binding 병목과 같은 종류다. Recursive
Summary는 Retrieval을 제거했지만 자연어 요약을 Tool·argument·복합 호출로
변환하는 문제까지 제거하지는 못한다.

Tool execution error는 총 4 task에서 발생했다. 이 중 3건은 Agent가 올바른
enum으로 재시도해 최종 ESM에는 성공했고, `vehicle-10-05`만 불필요한 상태
변경이 남아 실패했다.

## 7. Ours와 비교

| 항목 | Ours Fact-first | Recursive Summary |
| --- | ---: | ---: |
| 정답 Memory 충분 | 23/50 | 34/50 |
| 정답 Memory 불충분 | 27/50 | 16/50 |
| ESM | 0.50 | 0.62 |

두 audit의 교차 결과는 다음과 같다.

| Ours | Recursive | Task |
| --- | --- | ---: |
| 충분 | 충분 | 18 |
| 충분 | 불충분 | 5 |
| 불충분 | 충분 | 16 |
| 불충분 | 불충분 | 11 |

Recursive는 Ours에서 빠진 27 task 중 16개의 정답 근거를 최종 요약에
보존했다. 반대로 Ours DB에는 있었지만 Recursive 최종 요약에서 불충분한
task도 5건 있었다. 즉 두 방법의 Memory coverage는 완전히 겹치지 않으며,
Recursive의 전체 ESM 우위는 더 높은 semantic coverage와 연결된다.

비교 시 주의할 점은 Ours 수치는 frozen active record에 대한 deterministic
reviewed annotation이고, Recursive 수치는 자연어 summary에 대한 reviewed
semantic audit라는 점이다.

## 8. 병목 우선순위와 다음 검증

현재 근거로 본 Recursive Summary의 개선 우선순위는 다음과 같다.

1. 최초 extraction coverage
   - 정답 Fact 9 task가 올바른 형태로 한 번도 저장되지 않음
2. rewrite 보존성
   - 기존 Fact 삭제 금지와 사용자·장치별 identity 보존
3. cross-fact update 방지
   - 장치·predicate가 불명확한 발화가 기존의 다른 Fact를 덮어쓰지 못하게 함
4. 실행 변환
   - 사용자·조건 적용, arithmetic, Tool route, enum, 복합 호출

인과적 기여도를 확인하려면 다음 두 Oracle을 분리해 실행해야 한다.

- `Recursive Memory Oracle`: 16 task의 누락·왜곡 Gold Fact만 최종 요약에
  추가하고 기존 Agent와 Tool loop는 유지
- `Recursive Execution Oracle`: 충분한 summary를 유지하고 Gold Tool
  route·argument만 제공

두 결과를 분리해야 “요약 coverage를 고치면 실제 ESM이 얼마나 회복되는가”와
“요약 이후 실행 변환이 남기는 상한”을 Ours Oracle 분석과 같은 방식으로
비교할 수 있다.
