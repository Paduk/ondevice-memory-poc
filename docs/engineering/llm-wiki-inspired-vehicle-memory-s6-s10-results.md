# LLM-Wiki-inspired Vehicle Memory Scenario 6–10 결과

상태: `단일 paired E2E 완료 · 두 profile 모두 현재 형태 채택 보류`

평가일: 2026-08-10

> 후속 fresh-cache Scenario 1–10 2회 반복에서는 Recursive Summary ESM
> `0.655`, Fact-Wiki `0.610`이었다. 일반 성능 판단에는 단일 run인 이 문서보다
> [fresh-cache 반복 결과](./recursive-summary-vs-fact-wiki-fresh-s1-s10-2x-results.md)를
> 우선한다.

## 조건

- VehicleMemBench Scenario 6–10, profile별 50 task
- Agent `gpt-5.6-terra`, Memory `gpt-5.6-luna`
- Embedding `text-embedding-3-small`, 256 dimensions
- 기존 Recursive Summary 및 Post-normalized Fact cache 고정
- Gold Memory, Gold Tool/argument 미사용
- 현재 비교 단가는 기존 보고서와 같은 Agent input/output `$2/$8`, Memory
  input/output `$1/$6`, embedding input `$0.02` per 1M token이다. 실제 billing이
  아니라 uncached 비교 추정치다.

Artifact:

- [Summary paired run](../../ubuntu/evaluation/vehiclemembench/llm-wiki-s6-s10-summary-r1/841de86b-7ce5-4870-8086-1e940ba0eefd/results.md)
- [Fact baseline](../../ubuntu/evaluation/vehiclemembench/llm-wiki-s6-s10-fact-baseline-r1/f49afd61-c363-47fd-bb55-bac372c356f0/results.md)
- [Fact-Wiki](../../ubuntu/evaluation/vehiclemembench/llm-wiki-s6-s10-fact-r1/e63f541a-b527-4977-8b0f-ca199a516371/results.md)

## 결과

| 방법 | ESM | State F1 | Tool F1 | Arg exact | Recall@k | Agent tokens | Agent latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recursive Summary | **0.68** | **0.844** | **0.701** | **0.58** | N/A | 203,537 | 329.6초 |
| Summary-gated Wiki | 0.66 | 0.827 | 0.679 | 0.56 | N/A | 274,405 | 329.3초 |
| Post-normalized Ours | 0.64 | 0.755 | 0.607 | 0.52 | 0.667 | 196,765 | 267.5초 |
| Fact-Wiki | **0.70** | **0.756** | **0.679** | **0.56** | 0.667 | 374,444 | 405.0초 |

Summary-gated Wiki는 ESM `-0.02`, Agent token `+34.8%`로 채택 조건인 ESM
`+0.02`와 token 증가 `30% 이하`를 모두 실패했다. Fact-Wiki는 ESM `+0.06`,
Tool F1 `+0.073`, Arg exact `+0.04`였지만 Agent token `+90.3%`, Agent latency
`+51.4%`로 token 상한 `50%`를 크게 넘었다.

Fact-Wiki의 개선은 Scenario 7 `0.30→0.60`, Scenario 9 `0.20→0.30`에 집중됐고,
Scenario 8은 `1.00→0.90`으로 회귀했다. Coreference ESM은 `0.462→0.615`, error
correction은 `0.667→0.833`으로 개선됐지만 조건·conflict·state-shift에서는 ESM
이득이 없었다.

## Traversal과 단계별 비용

| 항목 | Summary-gated Wiki | Fact-Wiki |
| --- | ---: | ---: |
| gate/available task | 13 | 50 |
| Wiki Tool 사용 task | 9 | 17 |
| Wiki Tool calls | 22 | 48 |
| Wiki read context | 2,839 tokens | 8,358 tokens |
| evidence sufficient | 7 | 13 |
| traversal 새 성공 / 회귀 | 1 / 1 | 3 / 1 |

Summary traversal task는 순개선이 없었다. Fact traversal task는 `+2` 순개선이
있었지만 17개 중 14개는 baseline ESM을 바꾸지 못했다. Tool을 부르지 않은
Fact-Wiki task의 새 성공 1건은 linked planner expansion 또는 Agent 변동을 분리할
수 없어 traversal 효과로 주장하지 않는다.

| Workflow | Recursive | Summary-Wiki | Post-normalized | Fact-Wiki |
| --- | ---: | ---: | ---: | ---: |
| Idle Memory LLM tokens | 643,739 | 643,739 | 3,937,932 | 3,937,932 |
| Idle Memory latency | 590.7초 | 590.7초 | 2,338.9초 | 2,338.9초 |
| Online embedding tokens | 0 | 0 | 949 | 949 |
| Online retrieval latency | 0초 | 0.04초 | 22.2초 | 23.5초 |
| Quiz Agent tokens | 203,537 | 274,405 | 196,765 | 374,444 |
| Quiz Agent latency | 329.6초 | 329.3초 | 267.5초 | 405.0초 |
| 추정 전체 비용 | `$1.210` | `$1.362` | `$4.853` | `$5.224` |

가상 Wiki이므로 두 확장 profile의 Memory 생성 token은 baseline과 동일하다. 비용
증가는 전부 online Agent 경로에서 발생한다. 특히 Fact-Wiki의 로컬 Tool 자체는
75ms에 불과하고, 실제 병목은 search/read 뒤 Agent를 다시 호출하며 늘어난 LLM
input과 model latency다.

## Cloud smoke에서 수정한 회귀

- OpenAI strict function schema가 허용하지 않는 `uniqueItems`를 Wiki read Tool에서
  제거하고 회귀 테스트를 추가했다.
- 서로 다른 predicate가 같은 condition을 공유할 때 page title이 충돌하던 projection
  규칙을 generic title 병합으로 고쳤다. projection policy를 v2로 올렸다.
- 수정 후 Scenario 6–10에서 projection fallback과 schema 400 오류는 0건이며, 모든
  artifact privacy audit가 통과했다.

## 판정

1. `cloud_recursive_summary_gated_wiki`는 성능 이득이 없으므로 보류한다.
2. `cloud_post_normalized_fact_wiki`는 정확도 신호는 유망하지만 현재 progressive
   traversal 비용이 과도해 on-device baseline으로 채택하지 않는다.
3. 사전 중단 기준에 따라 추가 2회 반복, controlled ablation 및 Error Book 구현은
   수행하지 않는다.
4. 후속 후보는 Fact page/link와 planner expansion은 유지하되, deterministic
   query-complexity gate로 Wiki Tool을 노출할 task를 더 줄이는 Compact Fact-Wiki다.
