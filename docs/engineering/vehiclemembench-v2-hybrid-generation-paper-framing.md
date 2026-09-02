# VehicleMemBench V2 Hybrid Turn-wise Generation: 논문 프레이밍 초안

> **가정 기반 문서:** 아래 결과 해석은 현재 S1–S5에서 관찰된 평균값이 독립적인
> 50개 시나리오에서도 동일하게 재현됐다고 가정한 예시다. 실제 50개 실험 결과나
> 통계적 유의성을 보고하는 문서가 아니며, 논문 제출 전 실제 수치와 신뢰구간으로
> 교체해야 한다.

## 한 줄 요약

VehicleMemBench의 event-driven 생성 구조를 turn-wise supervision으로 확장하고,
`post_hoc`, `native_turnwise`, `hybrid` 생성 순서를 동일한 scenario에서 비교한 결과,
**Hybrid가 Native에 가까운 시간적 정합성을 유지하면서 event batching의 생성 효율과
가장 높은 downstream 품질을 함께 제공하는 Pareto 선택지**로 나타났다.

## 문제의 기원

원 VehicleMemBench는 persona → event chain → temporal interleaving → dialogue → final
Quiz의 다단계 합성 파이프라인과 executable vehicle simulator를 결합한다. 그러나
모든 대화가 끝난 뒤 final Quiz만 평가하므로, 실제 assistant처럼 대화 도중 수시로
질문받는 상황과 각 turn에서 memory가 어떻게 변해야 하는지는 직접 다루지 않는다.

V2는 동일한 structured event 원천을 유지하면서 매 turn 다음 항목을 추가한다.

```text
M(t-1) + turn t
  → UPDATE 또는 NO_OP
  → evidence와 reason
  → deterministic Patch
  → M(t)
  → 해당 cutoff에서 답할 수 있는 Turn Quiz
```

핵심 연구 질문은 “turn-wise label을 만들 것인가”가 아니라, **어떤 생성 순서가
causal quality와 생성 비용의 균형을 가장 잘 달성하는가**이다.

## 비교한 세 생성 방식

### Post-hoc

V1 방식으로 scenario의 dialogue를 모두 생성한 뒤, 완성된 event dialogue에서 turn별
UPDATE/NO_OP와 evidence boundary를 복원한다. 기존 V1 artifact를 V2로 변환하기 쉽고
비용이 낮지만, label이 실제 dialogue 생성 과정에 관여하지 않으므로 시간적 경계가
사후적으로 정렬된다.

### Native turn-wise

대화를 정확히 한 turn씩 생성하고 즉시 memory label과 `M(t)`를 확정한 뒤 다음 turn을
생성한다. 가장 엄격한 online causal cadence지만 수천 번의 순차 호출과 반복 prefill이
필요하다. Memory를 dialogue generator에는 노출하지 않아 self-confirming loop는
차단한다.

### Hybrid — 제안 방식

한 event의 dialogue는 한 번에 batch 생성하되, 다음 event로 넘어가기 전에 그 event의
turn들을 순서대로 검사해 memory trajectory를 확정한다.

```text
frozen structured event + 이전 상태
  → event dialogue batch 생성
  → 최초 causal evidence turn 정렬
  → turn별 NO_OP/UPDATE 생성
  → deterministic ADD/REPLACE/DELETE 적용
  → evidence·prefix·memory hash-chain 검증
  → 승인된 M(t)를 다음 event로 전달
```

LLM은 자연어 dialogue와 evidence alignment에 사용하고, memory operation과 Patch 적용은
가능한 범위에서 structured event로부터 결정론적으로 수행한다. 이 설계는 Native의
turn-level supervision과 Post-hoc의 batch 효율 사이를 절충한다.

중요하게도 Hybrid는 26–40개 turn의 memory를 한 번에 갱신하지 않는다. **한 번에
생성하는 것은 event dialogue뿐이며**, 생성된 dialogue 안에서는 turn 순서대로
`M(t-1) → NO_OP/UPDATE → M(t)`를 계산한다. 따라서 모든 cutoff memory와 turn-wise
학습 label은 Native와 동일한 시간 해상도를 가진다.

### Hybrid batching의 잠재적 단점

- Dialogue generator가 structured event의 전체 목표를 보므로 앞선 발화가 후속 결론을
  지나치게 자연스럽게 예고하는 `future-conditioned wording`이 생길 수 있다.
- Event 중간에 확정된 memory가 이후 발화 생성에 영향을 주는 실제 closed-loop 상호작용은
  재현하지 못한다.
- 전체 event dialogue를 본 evidence aligner가 최초 근거보다 후반의 명확한 재진술을
  UPDATE 경계로 선택할 수 있다.
- 일부 turn이 잘못 생성되면 해당 turn만이 아니라 event batch 전체를 재생성해야 할 수
  있다.

현재 구현은 각 UPDATE의 evidence가 해당 turn까지의 prefix에 실제 존재하는지 검사하고,
조건 없는 명시적 선호에는 earliest-evidence guard를 적용해 늦은 UPDATE를 줄인다.
다만 prefix 검사는 미래 정보의 직접 참조를 막을 뿐, 앞선 발화의 표현이 전체 event
결말에 의해 간접적으로 형성되는 문제까지 증명적으로 제거하지는 않는다.

Native가 이론적으로 더 적합한 경우는 승인된 `M(t)` 또는 system action이 다음 사용자
발화와 event 전개를 실제로 바꾸는 closed-loop benchmark다. 현재 Native 구현은
self-confirming loop를 막기 위해 memory를 다음 dialogue generator에 제공하지 않으므로,
이론적인 online advantage 중 상당 부분을 의도적으로 사용하지 않는다. 이 설정에서는
Hybrid의 batching disadvantage가 제한적인 반면, event 내부 dialogue coherence와 호출
효율은 오히려 좋아질 수 있다.

## 공정한 평가 설정

- 세 방법은 동일한 frozen persona, event chain, temporal order와 Stage 2 hash를 사용한다.
- scenario마다 Turn Quiz 30개와 Final Quiz 10개를 생성한다.
- Turn Quiz는 해당 cutoff의 memory만 사용하며 미래 dialogue와 final state를 보지 않는다.
- Agent 평가는 동일 모델·prompt·Tool schema로 두 번 fresh 실행한다.
- Quiz 품질은 ESM, Tool F1, Argument Exact로 평가한다.
- Memory만으로 gold call을 복원 가능한지는 별도 Answerability Judge로 평가한다.
- 모든 turn에 대해 Patch replay, NO_OP 불변성, evidence grounding, causal cutoff와 memory
  hash-chain을 결정론적으로 검사한다.
- UPDATE 전수, expected-update event 전수와 NO_OP 10% 층화 표본은 독립적인
  Luna/Terra Judge가 검사하고, 불일치는 SOL과 필요 시 Human Review로 보낸다.

50개 scenario라면 method당 unique Quiz 2,000개, 두 번 fresh 실행 기준 Agent run
4,000개가 된다. 통계 검정의 독립 단위는 Quiz가 아니라 **scenario**로 두고 paired
bootstrap 신뢰구간과 scenario-level win rate를 함께 보고한다.

## 가정된 50-scenario 결과

현재 S1–S5 평균이 50개에서도 그대로 유지됐다고 가정한 품질 표다.

| 생성 방식 | ESM | Tool F1 | Arg Exact | Answerability Full Support |
| --- | ---: | ---: | ---: | ---: |
| Post-hoc | 0.8925 | 0.9063 | 0.8450 | 0.940 |
| **Hybrid** | **0.9175** | **0.9361** | **0.8575** | **0.965** |
| Native turn-wise | 0.9025 | 0.9253 | 0.8525 | 0.955 |

생성 비용은 S1–S5의 dialogue 및 evidence-alignment usage를 10배 선형 외삽한 값이다.
Stage 1/2, 공통 Quiz 생성 및 평가 비용은 제외했다. 비용은 Terra의 input $2/M,
cached input $0.2/M, output $12/M 가정이다.

| 생성 방식 | Input | Output | 총 tokens | 예상 비용 | 누적 provider latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Post-hoc | 12.71M | 6.44M | 19.15M | $102.55 | 19.1시간 |
| **Hybrid** | **12.71M** | **6.42M** | **19.13M** | **$102.25** | **18.7시간** |
| Native turn-wise | 450.28M | 12.56M | 462.84M | $1,051.27 | 86.4시간 |

누적 latency는 호출 latency의 합이며 병렬 실행의 wall-clock time이 아니다. Native는
Hybrid보다 약 24.2배 많은 dialogue/alignment token과 약 10.3배 높은 비용을 사용한다.

## 결과 해석

가정된 결과에서는 Hybrid가 Post-hoc보다 ESM +2.5%p, Native보다 +1.5%p 높고,
Answerability도 각각 +2.5%p와 +1.0%p 높다. 동시에 생성 비용은 Post-hoc과 사실상
같고 Native보다 크게 낮다. 따라서 평균값만 보면 Hybrid가 세 방법 중 Pareto 우위다.

가능한 설명은 다음과 같다.

- Post-hoc은 dialogue가 완성된 뒤 label을 붙이므로 earliest evidence와 state transition의
  연결이 약해질 수 있다.
- Native는 causal order는 가장 엄격하지만, turn별 생성이 문맥을 잘게 나누고 호출
  편차와 반복 prefill을 크게 늘린다. 이 비용이 관찰된 품질 이득으로 이어지지 않았다.
- Hybrid는 event 내부의 자연스러운 dialogue coherence를 batch로 보존하면서, event
  사이에서는 승인 memory를 전달해 장기 상태 일관성을 유지한다.

따라서 관찰된 결과는 “batching에는 단점이 없다”는 뜻이 아니다. 현재 benchmark에서는
그 단점이 causal-prefix validator와 deterministic turn-wise replay로 대부분 통제되고,
Native가 memory-conditioned closed loop를 사용하지 않아 추가 비용이 새로운 정보나
감독 신호로 이어지지 않았다는 해석이 더 정확하다. 향후 closed-loop V2 variant에서는
Hybrid와 Native의 차이가 지금보다 커질 수 있으므로 별도 ablation이 필요하다.

이는 원인 규명이 아니라 결과에 대한 가설이다. 논문에서는 paired scenario delta의
95% 신뢰구간이 0을 벗어나는지 확인한 뒤 우월성을 주장해야 한다. 유의하지 않더라도
Native 대비 큰 비용 절감과 비열등 품질은 Hybrid 선택의 독립적인 근거가 된다.

현재 UPDATE/NO_OP 감사는 S1만 완료되어 방법별 표본 수가 다르므로 위 50-scenario
품질 표에는 포함하지 않았다. 실제 논문 표에는 50개 전체의 expected-update recall,
spurious-update rate, evidence-boundary offset와 memory-faithfulness를 추가해야 한다.

## 관련 연구와 차별점

- **VehicleMemBench**는 multi-user preference evolution을 executable Tool state로
  평가하고, Recursive Summarization·Key Value Store·Mem0·MemOS·LightMem 등 runtime
  memory system을 비교한다. 본 연구는 그 benchmark construction을 turn-wise causal
  supervision으로 확장한다. [로컬 논문](vehicleMembench.pdf)
- **LoCoMo, PerLTQA와 장기 대화 memory benchmark**는 장기 보존과 개인화를 평가하지만,
  주로 QA 중심이다. V2는 각 시점의 executable Tool call과 환경 상태까지 검증한다.
- **Mem0, MemOS, LightMem 등의 agent memory**는 실제 서비스에서 정보를 저장·검색하는
  runtime 방법론이다. Post-hoc/Hybrid/Native는 runtime memory baseline이 아니라,
  이를 학습·평가할 turn-wise gold trajectory의 **생성 cadence**다.
- **Retrieval-as-Reasoning과 LLM-Wiki**는 flat retrieval 대신 구조화된 knowledge와
  self-correction을 강조한다. V2도 evidence grounding, deterministic validation과
  escalation을 사용하지만, query-time traversal이 아니라 causal memory label 생성과
  데이터 품질 보증이 목적이다. [로컬 논문](../assets/2605.25480v2.pdf)

## 논문에서 주장할 수 있는 범위

실제 50개 paired 실험과 신뢰구간이 확보되면 다음을 주장할 수 있다.

1. 기존 final-only executable benchmark를 turn-wise memory/Quiz로 확장하는 생성·평가
   프로토콜을 제안했다.
2. 생성 cadence 자체를 통제한 최초의 paired ablation을 통해 causal strictness와
   비용 사이의 trade-off를 분리했다.
3. Hybrid event-batched turn-wise construction이 Native의 순차 호출 비용 없이 더 높은
   또는 비열등한 downstream quality를 달성했다.
4. Quiz 정확도만으로는 데이터 품질을 판별하기 어려워 Answerability, UPDATE/NO_OP,
   evidence boundary와 replay validity를 함께 평가해야 함을 보였다.

반대로 현재 5개 결과를 50개로 단순 복제한 값, 동일 generator/Judge의 공통 편향,
synthetic persona의 현실 대표성, Quiz 간 상관을 숨기면 안 된다. 최종 논문에는 실제
50개 독립 scenario, scenario-level confidence interval, blind Human Review와 별도
held-out 검증을 포함해야 한다.

## 추천 논문 포지셔닝

이 연구의 중심은 새로운 inference-time memory algorithm이 아니다. 가장 정확한
포지셔닝은 다음과 같다.

> **A causal, executable data-construction framework for turn-wise personalized vehicle
> memory, with a controlled study of post-hoc, fully online, and event-batched hybrid
> supervision.**

현재 가정 아래에서는 Hybrid를 production V2 생성 방식으로 선택하고, Post-hoc을
저비용 변환 baseline, Native turn-wise를 causal upper-bound 성격의 고비용 baseline으로
유지하는 구성이 가장 설득력 있다.
