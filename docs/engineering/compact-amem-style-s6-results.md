# Compact A-MEM-style Scenario 6 검증 결과

상태: `100-line smoke · 전체 cache · 10-task E2E 완료`

## 설정

- profile: `cloud_compact_amem_style`
- Memory / Agent: `gpt-5.6-luna` / `gpt-5.6-terra`
- embedding: `text-embedding-3-small`, 256 dimensions
- episode: 최대 16 entries, 8,000 characters, gap 21,600 seconds
- link threshold / candidates: `0.75` / `5`
- retrieval: top-k `10`, token budget `2,000`
- Ours Tool ontology, Gold Memory와 Gold call은 Memory 생성에 사용하지 않음

## 100-line smoke

| 항목 | 결과 |
| --- | ---: |
| Raw lines | 100 / 2,698 (partial) |
| Episodes / compact notes | 7 / 29 |
| Note compression ratio | 29.0% |
| Compaction calls | 7 |
| Generation tokens | 11,583 |
| Memory provider latency | 37.8초 |
| Retrieval recall@k, 1 task | 1.00 |
| ESM, 1 task | 0.00 |

첫 task의 정답 preference는 앞 100줄 밖에 있어 partial smoke의 ESM은 성능
판정에 사용하지 않는다. 이 smoke는 build, linked retrieval, Agent 연결과 privacy
audit를 확인하는 구조 검증이다.

## 전체 Scenario 6 결과

| 항목 | 결과 |
| --- | ---: |
| Raw lines | 2,698 |
| Episodes | 190 |
| Compact notes / links | 522 / 2,458 |
| Note compression ratio | 19.35% |
| 정상 compaction calls | 190 |
| Generation tokens | 281,576 (input 215,617 / output 65,959) |
| 전체 Memory provider tokens | 336,905 |
| 전체 Memory provider latency | 975.7초 |
| ESM | 0.60 |
| State / Value F1 | 0.776 / 0.776 |
| Tool F1 / Argument exact | 0.633 / 0.50 |
| Retrieval recall@k / hit rate | 0.75 / 0.70 |
| Artifact privacy audit | 통과, residual 0 |

10 task 중 ESM 성공은 6건이다. 실패 4건 중 두 건은 retrieval recall 0, 한 건은
정답 근거를 검색했지만 Agent가 호출을 생략했고, 한 건은 검색 후 zone argument를
잘못 선택했다. 따라서 다음 개선 지점은 compact 생성량보다는 preference coverage,
conflict ranking과 downstream argument binding이다.

전체 실행 중 모델이 유효한 source ID를 비순차 배열로 한 번 반환했다. source
evidence 집합은 맞았으므로 episode 순서로 canonicalize하는 graph policy v2를
추가했고 cache resume으로 완료했다. 결과 trace에는 감사용 실패 1회와 retry가
남아 있다. 위 generation token 합계는 저장된 정상 episode usage이며 그 실패
응답의 실제 token은 provider 예외 이전에 회수되지 않아 포함하지 않는다.

## `cloud_amem_style` 호출량 비교

동일 Scenario 6의 기존 full E2E artifact가 없어 정확도 수치는 직접 비교하지
않았다. 네트워크 호출 없는 dry-run 비교는 다음과 같다.

| 방식 | 생성 호출 |
| --- | ---: |
| Compact A-MEM-style | 190 정상 + 1 validation retry |
| A-MEM-style 최소 | 2,698 |
| A-MEM-style 보수적 최대 | 5,395 |

Compact의 정상 생성 호출은 최소 기준 대비 `92.96%`, 최대 기준 대비 `96.48%`
적다. 즉 약 `14.2–28.4배` 적은 생성 호출로 전체 cache를 구축했다. 비용 rate가
설정되지 않아 USD 비용은 보고하지 않는다. 수천 회 호출이 필요한 비교군을 이번
검증에서 새로 실행하는 대신 호출량 경계를 명시한다.

## 산출물

- [전체 10-task 결과](../../ubuntu/evaluation/vehiclemembench/compact-amem-style-s6-full/9226ee59-1d19-4651-9a8e-e62442e80774/results.md)
- [전체 cache manifest](../../ubuntu/evaluation/vehiclemembench-memory/compact-amem-style-v1/d745b359794da154/scenario-06/8a4bea509e8a14f1/manifest.json)
- [100-line smoke 결과](../../ubuntu/evaluation/vehiclemembench/compact-amem-style-s6-smoke/ae77bc67-0936-4547-8aa5-493f100d96bc/results.md)
