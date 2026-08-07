# 현재 최고 Ours 요약

상태: `S1–10 단일-run 최고 Ours, 동일-cache 반복 평가 전`

## 방법

현재 방식은 **Post-normalized Schema-informed Recursive-assisted Fact Memory**에
**Joint Memory–Tool Planning**을 결합한다.

```text
History → 검증·정규화된 versioned Fact
Query → 관련 Fact/Bundle 검색
Query + Fact + 제한된 Tool Schema → 공동 Tool 선택
관련 Fact + 현재 상태 → Desired State → schema-valid Tool Plan
→ Agent 실행 (불확실하면 기존 경로로 fallback)
```

핵심은 질문만으로 Tool을 고르지 않고 검색된 Fact와 Tool Schema를 함께 보며,
기존 query route를 보존한 채 Fact가 직접 지지하는 복구 후보를 최대 1개만 추가하는
보수적 reranker다. 후보 근거도 Agent에게 전달한다. Desired-state planner는 여러
Fact와 현재 상태를 하나의 실행 목표 및 Tool 묶음으로 변환하도록 설계됐다.

## S1–10 결과

| 지표 | 기존 동일-cache Ours | 현재 Ours |
| --- | ---: | ---: |
| ESM | 0.52 | **0.60** |
| State F1 | 0.686 | **0.755** |
| Tool F1 | 0.558 | **0.595** |
| Argument Exact | 0.46 | **0.52** |
| Selector miss | 29 | **20** |

현재 Ours는 기존 Ours 계열 중 가장 높고 Recursive Summary와 ESM 0.60으로
동률이다. 다만 Recursive Summary의 State/Tool F1에는 아직 못 미친다. 현재 성능
향상은 주로 **공동 Tool 선택(개선 1)**에서 확인됐으며, 실제 multi-Fact call이 없어
**Desired-state 복합 계획(개선 2)**의 효과는 아직 검증되지 않았다.

