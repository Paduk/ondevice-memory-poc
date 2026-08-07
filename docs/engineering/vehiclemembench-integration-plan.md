# VehicleMemBench 통합 구현 계획

상태: `Phase V4 완료 — Phase V5 착수 가능`

최종 검토: 2026-07-25

## 1. 목표와 고정 결정

PalmClaw Ubuntu Runtime의 논문용 주 평가를
[VehicleMemBench](https://github.com/MINE-USTC/VehicleMemBench)로 확장한다.
기존 합성 평가는 제거하지 않고 빠른 회귀·보안·단위 평가에 사용한다.

- 평가 단위: 50개 사용자 시나리오, 시나리오당 10개 task, 총 500개 task
- 실행 환경: 실제 차량이 아닌 공식 Python `VehicleWorld` simulator
- 정답 판정: 공식 state scorer의 Exact State Match와 field/value F1
- Agent/Memory 모델: OpenAI Cloud LLM
- 격리 원칙: history는 시나리오당 한 번만 처리하고 각 task의 차량 상태는 초기화
- 누출 방지: task 실행 시 원본 history를 context에 직접 넣지 않고 검색된 memory만 사용
- 재현성: upstream commit, dataset hash, 모델·prompt·schema·policy version 고정
- 기존 Local SLM adapter는 유지하되 이번 주 평가 범위에서는 제외

## 2. Phase 요약

| Phase | 목표 | 완료 결과 |
| --- | --- | --- |
| V1 | Benchmark fixture와 scorer 연결 | 완료: 한 시나리오를 offline으로 재현 |
| V2 | PalmClaw Agent E2E 연결 | 완료: OpenAI Agent와 공식 scorer 연결 |
| V3 | Memory ingestion·retrieval 연결 | 완료: 4개 memory profile로 scenario 1 평가 |
| V4 | 평가 지표·진단·개인정보 | 완료: 5 scenario, 200 profile-task 진단 |
| V5 | 전체 평가·Ablation | 500 task 결과표와 재현 가능한 산출물 생성 |

## 3. Phase V1 — Fixture와 Offline Smoke

구현:

- 외부 benchmark 경로와 고정 commit을 받는 dataset loader
- `history_N.txt`와 `qa_N.json`의 50:50 pairing 및 schema 검증
- 111개 차량 API의 PalmClaw Tool schema 변환
- 공식 `VehicleWorld`와 state scorer wrapper
- gold tool call을 실행하는 1 scenario offline smoke test

완료 조건:

- dataset 원본을 수정하거나 저장소에 복제하지 않는다.
- task마다 simulator state가 초기화된다.
- gold call의 ESM, state F1과 tool F1이 공식 scorer 결과와 일치한다.

검증 결과:

- 고정 commit `5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b`
- 50 scenario, 500 task, 111 Tool schema 및 전체 gold argument 검증
- scenario 1의 10/10 task에서 ESM, state/value F1과 Tool F1 모두 1.0

## 4. Phase V2 — Agent E2E 통합

구현:

- `ToolRegistry` 주입과 `list_module_tools` 기반 동적 Tool discovery
- Vehicle Tool 호출을 simulator에만 전달하는 adapter
- query → OpenAI Agent → Tool → 결과 → 최종 응답의 평가 전용 loop
- query별 Tool trace, 최종 차량 state와 scorer 입력 저장
- No Memory와 Gold Memory profile

완료 조건:

- OpenAI live mode로 1 scenario의 10개 task를 끝까지 실행한다.
- 실제 OS·차량 side effect가 발생하지 않는다.
- 원본 history가 Agent prompt에 직접 포함되지 않는다.
- PalmClaw 결과와 공식 scorer 결과가 일치한다.

검증 결과:

- `gpt-5.6-terra`, scenario 1, No Memory와 Gold Memory 각 10 task 실행
- 20/20 runtime 완료, No Memory ESM 0.30, Gold Memory ESM 0.90
- query별 model/Tool trace와 initial/reference/predicted state 저장
- 원본 history, 파일·웹 Tool과 API key가 prompt·artifact에 포함되지 않음

## 5. Phase V3 — Memory 통합

구현:

- history의 날짜·화자·timestamp를 보존하는 parser
- 시나리오별 1회 ingestion과 일자·token 단위 Cloud consolidation
- 10개 task가 공유하는 read-only memory snapshot
- Cloud Summary, Structured BM25, Embedding, Hybrid retrieval profile
- evidence, memory version, retrieval score와 선택 결과 trace

완료 조건:

- 같은 history를 task마다 재처리하지 않는다.
- 평가 query와 결과가 다음 task의 memory를 오염시키지 않는다.
- `history → memory → retrieval → vehicle action`을 한 trace에서 추적한다.
- 중단 후 memory cache와 완료 task를 재사용해 재개한다.

검증 결과:

- 공식 50개 scenario의 history 134,518행을 파싱하고 날짜·화자·timestamp를 보존
- scenario 1의 2,438행을 10개 batch로 1회 처리해 Summary/Structured
  consolidation 각각 10/10 완료
- Structured memory 21개, version 2 이상 6개, evidence 24건 저장
- 4개 profile × 10 task의 40/40 runtime 완료:
  Summary ESM 0.40, BM25/Embedding/Hybrid ESM 각 0.70
- 전체 task에서 같은 memory fingerprint를 사용했고, 평가 query는 message/memory를
  추가하지 않음
- 완료 run 재개 시 40개 task를 재호출하지 않고 기존 checkpoint를 복원

위 수치는 scenario 1 smoke 결과이며 전체 benchmark 성능 결론은 아니다.

## 6. Phase V4 — 지표·진단·개인정보

구현:

- ESM, state precision/recall/F1, tool F1과 불필요한 Tool 호출 수
- reasoning type별 성능 집계
- memory 오류, retrieval 누락, extra call, execution 오류 분류
- latency, token, 비용과 Cloud 전송량 측정
- Cloud 전송 전 secret·PII 탐지/redaction 및 정보 손실 보고
- JSONL, TSV, Markdown, SVG와 SQLite 결과 저장

완료 조건:

- 5 scenario, 50 task에서 핵심 profile을 비교한다.
- 실패 task를 memory·retrieval·Tool·state 단계까지 역추적한다.
- 원본 개인정보와 secret이 trace·결과 파일에 저장되지 않는다.
- redaction으로 정답 정보가 손실된 사례를 별도로 표시한다.

검증 결과:

- scenario 1–5의 50개 고유 task를 4개 핵심 profile로 평가해 200/200
  runtime 완료
- ESM: No Memory 0.20, Gold 0.90, Cloud Summary 0.30,
  Cloud Structured Hybrid 0.58
- state precision/recall/F1, value/tool F1, reasoning type, 불필요한 호출,
  latency p95, token과 실패 taxonomy 집계
- memory generation 140회와 retrieval embedding 50회의 Cloud 전송량을
  분리하고 address 2건을 전송 전에 redaction; 민감문자 노출률 0.0
- 전체 dataset에서 redaction 정답 손실 위험 task 1건(`vehicle-39-03`) 식별
- 42개 산출물 재귀 privacy audit에서 잔존 민감정보 0건
- JSONL/TSV/Markdown/SVG와 SQLite 200 case 저장 및 완료 suite refresh 검증
- 가격 환경변수가 설정되지 않아 token은 실측했지만 비용은
  `unconfigured`/0으로 기록

## 7. Phase V5 — 전체 평가와 Ablation

비교 profile:

1. No Memory
2. Gold Memory
3. Cloud Summary Memory
4. Cloud Structured Memory + BM25
5. Cloud Structured Memory + Embedding
6. Cloud Structured Memory + Hybrid
7. Hybrid에서 Retrieval, Gate, Redaction을 각각 제거한 ablation

구현:

- 50 scenario, 500 task 전체 실행
- scenario 단위 checkpoint, retry와 실패 격리
- profile 및 reasoning type별 비교표·그래프
- model, prompt, schema, policy, dataset와 upstream version manifest

완료 조건:

- 500개 task가 완료되거나 실패 원인이 명시적으로 기록된다.
- 동일 입력과 scorer로 모든 profile을 비교한다.
- 전체 결과와 개별 trace를 다시 생성할 수 있는 명령과 manifest를 제공한다.
- 논문용 표·그래프와 대표 failure case를 생성한다.

## 8. 목표 CLI와 산출물

```bash
palmclaw eval vehicle \
  --benchmark-root /path/to/VehicleMemBench \
  --mode live \
  --profiles cloud_summary,cloud_structured_bm25,cloud_structured_embedding,cloud_structured_hybrid \
  --scenario 1 \
  --task-limit 10
```

예정 코드 경계:

```text
ubuntu/src/palmclaw_ubuntu/vehicle_bench/
  dataset.py   # history/QA loader
  tools.py     # VehicleWorld Tool adapter
  runner.py    # scenario/task lifecycle
  scoring.py   # official scorer wrapper
  agent.py     # 평가 전용 Agent loop
  memory.py    # history batching, cache, retrieval snapshot
```

평가 산출물은 `ubuntu/evaluation/vehiclemembench/` 아래에 저장하고, 대용량 원본
dataset과 API 응답 원문은 Git에 포함하지 않는다.

## 9. 수행 원칙

- V1 → V2 → V3 → V4 → V5 순서로 진행하며 각 Phase 완료 후 다음 단계로 이동한다.
- 전체 500 task는 1 scenario 및 5 scenario gate를 통과한 뒤 실행한다.
- 공식 simulator와 scorer는 수정하지 않고 adapter 밖의 평가 oracle로 유지한다.
- upstream 코드를 vendoring할 경우 라이선스와 attribution을 먼저 확인한다.
- benchmark에 맞추기 위한 memory schema 변경은 baseline 측정 후 별도 ablation으로 검증한다.
