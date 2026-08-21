# Turn-wise VehicleMemBench 실험 아카이브 안내

## 1. 목적과 범위

이 아카이브는 2026-08-13부터 2026-08-16까지 수행한 Turn-wise VehicleMemBench 실험 산출물을 보존한다. 메모리를 매 대화 턴마다 갱신하는 환경에서 다음 계열을 비교하기 위한 결과다.

| 이름/디렉터리 표기 | 의미 |
| --- | --- |
| `turnwise-summary-*` | 매 턴 전체 메모리를 다시 작성하는 Turn-wise Recursive Summary |
| `turnwise-patch-*` | 매 턴 필요한 부분만 `add`/`replace`/`delete`하는 기본 Patch |
| `turnwise-patch-soft30-*` | Patch 누적 메모리가 임계치를 넘으면 Recursive Summary 계열 압축을 시도하는 soft-30 변형 |
| `turnwise-temporal-*` | `durable/current/temporary/conditional` 상태를 구분하는 Temporal-aware Patch |
| `turnwise-combined-*`, `*-temporal-compact-*` | Temporal-aware Patch와 soft-30 압축을 결합한 Combined |
| `*-fresh-*` | 메모리 생성부터 Quiz까지 새로 실행한 실험 |
| `*-frozen-replay-*` | 저장된 동일 메모리를 고정하고 Quiz Agent만 다시 실행한 실험 |
| `*-r1-*`, `*-r2-*` | 동일 설정의 반복 실행 번호 |
| `*-sN-*`, `*-sN-sM-*` | 평가한 시나리오 번호 또는 범위 |
| `*-controller-*` | 여러 시나리오 실행을 순차 배치로 시작하고 완료 여부를 모은 제어 작업 |

아카이브에는 성공한 실행뿐 아니라 실패·중단 실험과 smoke test도 포함한다. 실패 기록은 방법론 오류, API/실행 오류, 벤치마크 정답 오류를 나중에 구분해 감사하기 위해 보존한다.

## 2. 디렉터리 구조

각 `turnwise-*` 최상위 디렉터리는 대체로 다음 구조를 갖는다.

```text
turnwise-<method>-<run>-<scenario>-<date>/
├── COMPLETED | FAILED       # 실행의 최종 상태 marker
├── experiment.log           # 실행 명령, 진행 상황, 오류 로그
├── experiment.pid           # 실행 당시 PID(현재 재사용 불가)
├── cache/
│   └── <config>/<scenario>/<snapshot>/
│       ├── manifest.json    # 캐시 설정, signature 및 snapshot 메타데이터
│       └── memory.db        # 생성된 메모리와 trace를 보관한 SQLite DB
└── results/
    └── <run-id>/
        ├── manifest.json
        ├── metrics.json
        ├── cases.jsonl
        ├── diagnostics.jsonl
        ├── results.md
        ├── results.tsv
        ├── reasoning_types.tsv
        ├── privacy_audit.json
        └── *.svg
```

Controller 디렉터리에는 `controller.log`, `controller.pid`, `BATCH*_STARTED`, `BATCH*_TERMINAL`, `ALL_TERMINAL` 같은 배치 상태 파일이 들어갈 수 있다. 실행 도중 남은 `*.tmp`, `memory.db-wal`, `memory.db-shm`도 원상 보존한다.

## 3. 파일별 설명

| 파일 | 설명 | 보존 중요도 |
| --- | --- | --- |
| `cases.jsonl` | 태스크별 질문, 정답/예측 tool call, ESM 및 세부 score, reasoning type, memory/model/tool trace, 입력 문맥과 usage가 담긴 핵심 원자료 | 필수 |
| 결과 `manifest.json` | run ID, 시나리오·태스크 목록, 모델/에이전트 설정, memory snapshot, 상태와 시각 | 필수 |
| 캐시 `manifest.json` | 메모리 profile, prompt/schema version 및 캐시 signature를 포함한 재사용 조건 | 필수 |
| `memory.db` | 비싼 메모리 생성 결과를 보존하는 SQLite DB. 동일 메모리의 frozen Quiz replay에 필요 | 필수 |
| `metrics.json` | profile별 집계 성능, provider usage, memory/agent token·latency, 비용 추정치, 실패 태스크 수 | 필수 |
| `results.tsv` | ESM, State F1, Tool F1, Arg Exact 등 표 형태의 결과 | 권장 |
| `reasoning_types.tsv` | reasoning type별 집계 결과 | 권장 |
| `diagnostics.jsonl` | 태스크별 진단 및 실패 원인 | 권장 |
| `results.md` | 사람이 빠르게 읽을 수 있는 실행 요약 | 권장 |
| `privacy_audit.json` | cloud 노출 및 privacy 검사 결과 | 권장 |
| `*.svg` | metric별 시각화 | 재생성 가능 |
| `experiment.log` | 실행 과정, 예외, 실제 결과 위치 확인용 로그 | 필수 |
| `COMPLETED` / `FAILED` | 실행 종결 상태 marker | 필수 |
| `experiment.pid`, `controller.pid` | 실행 당시 PID이며 재현에는 직접 사용하지 않음 | 참고 |
| `*.tmp`, `*-wal`, `*-shm` | 중단 시점의 임시/SQLite 보조 파일. 실패 감사 목적으로 포함 | 참고 |

`cases.jsonl`은 가장 크지만 결과 재채점, ESM flip 분석, reasoning-type 분석의 근거이므로 제거하지 않는다. `memory.db` 역시 메모리 생성 비용 없이 Quiz Agent 편차를 재측정하는 데 필요하다.

## 4. 핵심 비교 결과

S1–S50의 첫 fresh run에서 잘못된 reference 4개를 제외한 496개 태스크 기준 결과다.

| 방법 | ESM | 정답 수 |
| --- | ---: | ---: |
| Turn-wise Recursive Summary | 0.6492 | 322 / 496 |
| Combined | 0.6411 | 318 / 496 |

두 방법의 성능 차이는 4개 태스크, ESM 0.0081로 작았다. 같은 구간의 비용/효율 집계는 다음과 같다.

| 항목 | Turn-wise Summary | Combined | Combined 변화 |
| --- | ---: | ---: | ---: |
| Memory input tokens | 119.409M | 136.472M | +14.3% |
| Memory output tokens | 7.127M | 2.655M | -62.7% |
| Agent input tokens | 2.799M | 2.273M | -18.8% |
| E2E tokens | 129.409M | 141.482M | +9.3% |
| 누적 memory-call latency | 50.35시간 | 39.35시간 | -21.8% |
| 누적 전체 provider latency | 51.01시간 | 40.00시간 | -21.6% |
| 추정 API 비용 | $168.37 | $157.60 | -6.4% |
| 최종 memory 평균 길이 | 831.9 tokens | 487.6 tokens | -41.4% |

비용은 Memory=Luna 입력/출력 $1/$6, Agent=Terra 입력/출력 $2/$8 per 1M tokens 가정이며 cached-input 할인은 반영하지 않았다. Latency는 병렬 실행의 실제 경과시간이 아니라 provider 호출 latency의 합이다.

## 5. 평가 시 제외할 benchmark reference 오류

다음 태스크는 방법론 실패가 아니라 gold/reference가 허용된 tool schema를 위반해 모든 공정 비교에서 제외한다.

| 태스크 | 문제 |
| --- | --- |
| `vehicle-19-04` | 허용 범위를 벗어난 brightness |
| `vehicle-42-08` | 유효하지 않은 `map_view` |
| `vehicle-45-03` | 유효하지 않은 `air_direction` |
| `vehicle-47-05` | 유효하지 않은 ambient color |

원본 파일은 수정하지 않는다. 향후 재집계 시 exclusion 목록을 별도 파일 또는 분석 코드로 적용한다.

## 6. 해석 주의사항

- ESM은 strict 0/1이므로 인자 하나의 추가·누락도 전체 태스크 실패가 된다.
- 시나리오 하나는 보통 Quiz 10개라 태스크 하나가 시나리오 ESM 0.1에 해당한다.
- 동일 memory를 고정한 Quiz replay에서도 Agent 출력 편차가 관측됐다. 단일 실행의 작은 차이를 방법론 우위로 단정하지 않는다.
- Fresh run은 memory 생성 편차와 Quiz Agent 편차를 모두 포함한다. Frozen replay는 후자만 측정한다.
- `cases.jsonl`에는 원문 대화와 모델 trace가 포함될 수 있으므로 외부 보관 시 암호화 및 접근 통제를 적용한다.

## 7. 무결성 확인과 복원

아카이브와 같은 위치의 `.sha256` 파일을 사용한다.

```bash
sha256sum -c PalmClaw-turnwise-vehiclemembench-20260816.tar.gz.sha256
tar -tzf PalmClaw-turnwise-vehiclemembench-20260816.tar.gz >/dev/null
```

복원은 충분한 공간이 있는 새 디렉터리에서 수행한다.

```bash
mkdir -p restored-turnwise-results
tar -xzf PalmClaw-turnwise-vehiclemembench-20260816.tar.gz \
  -C restored-turnwise-results
```

아카이브를 유일한 사본으로 간주하지 말고, 원본 NVMe와 별도의 물리 장치 또는 암호화된 object storage에 최소 한 사본을 추가 보관한다.
