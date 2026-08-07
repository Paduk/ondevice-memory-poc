# Vehicle Memory 방법론 진행상황

마지막 갱신: `2026-08-04`

## 현재 진행상황

| 항목 | 내용 |
| --- | --- |
| 현재 후보 | Post-normalized Ontology + Recursive-assisted Ours |
| 완료 | 구현, Scenario 6 smoke, Scenario 6–10 평가, P0 고정-cache 3회 비교 |
| 개선 | 3회 평균 Recall `0.627→0.667`, State F1 `0.702→0.746`, Tool F1 `0.584→0.620`, argument exact `0.440→0.507` |
| 미확정 | 평균 ESM `0.593→0.613`: `+0.020`이지만 run 변동 범위 안 |
| 현재 위치 | [Scenario 1–10 구조 비교](memory-methodology-s1-s10-structural-bottleneck-comparison.md) 완료, 단편 수정 보류 |
| 다음 개선 대상 | 기존 cache 기반 100-task Memory Bundle·desired-state stage audit |

## 방법론 현황

| 방법 | 상태 | ESM | 개선점 | 남은 문제 |
| --- | --- | ---: | --- | --- |
| Recursive Summary | Baseline | 0.58 (S1–5), **0.62** (S6–10) | 구조 없이 넓은 context를 직접 사용 | Summary 누락·왜곡, 전체 Summary 주입 비용 |
| Structured Hybrid | Baseline | 0.56 | 안정적인 구조화 검색 기준선 | Ours와 저장 방식이 다름 |
| Ours Fact-first | 완료 | 0.50 | Fact, evidence, versioning, query별 검색 | Fact 누락 27건, selector·binding 오류 |
| Ours + Recursive Hybrid | 완료 | 0.60 | State F1 0.798, Tool F1 0.646 | Summary 직접 주입으로 context 증가 |
| Recursive-assisted Ours | 완료 | 0.593 (3회 평균) | active Fact 증가, P0 Recall 0.627 | extraction 호출 약 2배, selector·binding 오류 |
| 앞단 Ontology-assisted | 중단 | 0.56 | 잘못된 UPDATE 감소 | Candidate `115→71`로 중요 Fact 누락 |
| **Post-normalized Ontology-assisted** | **현재 후보** | 0.52 (S1–5), **0.613** (S6–10 3회 평균) | 구조화 검색과 낮은 Agent context | ESM 우위 없음, 높은 생성비용, 실행 후단 오류 |

## 남은 작업

| 우선순위 | 작업 | 상태 |
| ---: | --- | --- |
| P0 | 기존 Recursive-assisted와 Post-normalized를 고정 cache로 각각 3회 비교 | **완료: 평균 ESM 0.593 / 0.613** |
| P1 | 두 방법이 모두 0/3 실패한 7개 task에서 Tool selector의 범용 원인 분리 | **완료: 6/7 selector 누락, 전체도 27/50** |
| P1 | 같은 안정 실패에서 argument mapping의 범용 원인 분리 | **완료: route bonus·시간 조건·불충분 근거 분리** |
| P1 | Recursive Summary와 최신 Ours의 Scenario 1–5 교차 평가 | **완료: ESM 0.58 / 0.52, selector miss 22/50** |
| P1 | Scenario 1–10 전체 실패 taxonomy와 구조적 수정 선정 | **완료: Memory Bundle + desired-state planner 방향** |
| P1 | 기존 cache로 bundle·capability·required-slot offline stage audit | 다음 작업 |
| P1 | 선정한 구조 PoC 후 matcher-only 고정-cache replay | 대기 |
| P2 | 핵심 Fact 값이 없는 안정 실패 4건과 speaker reject 분석 | 대기 |
| P2 | on-device 비용 비교 | 대기 |

## 주요 실험

| 방법 | Run ID | 결과 |
| --- | --- | --- |
| Recursive Summary | `29518f79-654d-4da0-865a-17ca633b6e70` | ESM 0.62 |
| Ours + Recursive Hybrid | `0500a4ff-48e1-4335-b775-142198afc45c` | ESM 0.60 |
| Recursive-assisted Ours | `24dcf1f2-40af-48da-b893-420e4404525c` | ESM 0.60, Recall 0.603 |
| 앞단 Ontology-assisted | `a92d4bdb-d7e5-49f8-860f-07a77131e0e6` | ESM 0.56 |
| Post-normalized Ontology-assisted | `15f5a789-8dd5-4d6c-9190-887290bd5bff` | ESM 0.60, Recall 0.667 |
| P0 고정-cache 반복 | [상세 결과](memory-methodology-p0-agent-repeat-results.md) | 평균 ESM: Recursive-assisted 0.593, Post-normalized 0.613 |
| P1 안정 실패 분석 | [상세 결과](memory-methodology-p1-stable-failure-analysis.md) | 4건 Fact 누락, 2건 근거 불충분, 1건 명확한 조건 판정 오류 |
| Scenario 1–5 교차 평가 | [상세 결과](memory-methodology-s1-s5-comparison-results.md) | Recursive Summary 0.58, 최신 Ours 0.52 |
| Scenario 1–10 구조 비교 | [상세 결과](memory-methodology-s1-s10-structural-bottleneck-comparison.md) | Recursive 0.60, 최신 Ours 0.56; 공통 action-plan 계층 부재 |

> 단일 run 수치는 보존된 artifact 기준이다. P0의 3회 평균 차이 `+0.020`도
> 실행 변동 범위 안이므로 확정적인 ESM 개선으로 보지 않는다.
