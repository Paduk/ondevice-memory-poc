# On-device Memory 4-Method 학습·평가 설계

## 0. 현재 상태 (2026-08-23)

학습·Validation·Ollama Test에 필요한 코드는 모두 구현됐다. 현재 남은 핵심 작업은
**primary run 실행과 결과 비교**이며, 첫 실행 대상은 `Qwen3.5 4B × Patch`다.

| 영역 | 상태 | 현재 근거 |
|---|---|---|
| 데이터·split·sampler | 완료 | 276,938 Turn 정렬, S1–80/81–90/91–100 분리, replay 검증 |
| 4개 방법 schema/executor | 완료 | Summary/Patch/Delta/Summary+Reason 전체 replay 통과 |
| HF/PEFT 학습·resume | 완료 | Granite 3B 1-step 및 resume canary 통과 |
| 학습 중 Validation | 완료 | teacher-forced, one-step, S81–S90 closed-loop 구현 |
| 모니터링 UI | 완료·실행 중 | read-only dashboard `:5060`, 다중 run 비교 지원 |
| GGUF/Ollama Test | 준비 | BF16 무양자화 exporter/Test runner 구현, fine-tuned Qwen 실모델 smoke 대기 |
| Qwen3.5 4B primary | 학습 중 | Summary/Delta/Patch/Patch-NO_OP5 네 run 실행 중 |
| 최종 비교 리포트 | 대기 | primary run과 S91–S100 Test 완료 후 작성 |

현재 구현 검사는 Ruff와 단위 테스트 **17건**을 통과했다. 시스템 Ollama `0.6.5`
(`:11434`)는 보존하고, Qwen3.5용 Ollama `0.32.15`를 `:11435`에 격리 운영한다.

## 1. 비교 방법

동일 모델·데이터·sampler를 사용하고 출력 schema와 memory state transition만 바꾼다.

| 방법 | 입력 state | Decode 출력 | 적용 방식 |
|---|---|---|---|
| Summary | 전체 memory + Turn | `decision`, UPDATE 시 `next_memory` | 전체 교체 |
| Patch | 전체 memory + Turn | `decision`, UPDATE 시 `operations[]` | 즉시 deterministic 적용 |
| Delta | base Summary + pending Patch + Turn | `decision`, UPDATE 시 `operations[]` | 누적 후 UPDATE 5회마다 deterministic 병합 |
| Summary+Reason | 전체 memory + Turn | `reason_code`, 한 문장 `reason`, `decision`, UPDATE 시 `next_memory` | 전체 교체 |

- 앞의 세 방법은 `reason_code/reason`을 학습 target과 decode에서 모두 제외한다.
- Summary 계열의 `NO_OP`은 전체 memory를 다시 생성하지 않고 runtime이 기존 state를 유지한다.
- Summary+Reason은 autoregressive 효과를 보기 위해 **reason → decision → memory** 순서로 직렬화한다.
- Delta compaction은 모델 생성이 아니라 executor가 수행하며 `compaction.jsonl`은 검증에 사용한다.

## 2. 코드 구조

```text
memory_training/
  environment.yml
  train.py
  models.py               # HF model loader, LoRA target/freeze
  training_data.py        # chat template, label masking, collator
  dataset.py              # aligned JSONL offset catalog, lazy read
  sampling.py             # NO_OP sampling, trajectory window
  tracking.py             # MLflow + local metrics.jsonl
  validation.py           # teacher-forced, one-step, closed-loop
  dashboard/
    server.py             # 통합 학습 현황 API·웹 서버
    static/               # loss/validation/run 비교 화면
  export_ollama.py        # LoRA merge, GGUF/Modelfile, Ollama 등록
  ollama_client.py        # structured generation, Tool Calling adapter
  ollama_isolated.py      # Qwen3.5용 격리 Ollama start/status/stop
  ollama_test.py          # S91–S100 Memory + Turn/Final Quiz Test
  methods/
    summary.py
    patch.py
    delta.py
    summary_reason.py
```

공통 `MemoryMethod` interface는 `format_input`, `format_target`, `parse_output`,
`apply_output`, `materialize_memory`를 제공한다. 기존 JSONL을 복제하지 않고 dataset
adapter가 필요한 target 필드만 투영한다.

### HF 모델 입력

학습기는 등록된 model key를 `--model`로 받고 `--method`, `--run-id`와 함께 사용한다.
Hugging Face ID는 `config.py`의 고정 matrix에서 해석한다.

```bash
python -m memory_training.train \
  --model qwen3.5-4b|qwen3.5-9b|granite4.1-3b|granite4.1-8b \
  --method summary|patch|delta|summary_reason \
  --run-id <name>
```

Transformers + PEFT/LoRA 기반으로 구현하고 tokenizer/chat template도 해당 HF repository
설정을 사용한다. 학습 전 architecture가 GGUF/Ollama 변환을 지원하는지 canary 검사를
통과시켜, 학습 후 Test 단계에서 변환 불가능한 모델을 선택하는 일을 막는다.

### 고정 대상 모델

| Family | 크기 | Hugging Face ID | Ollama 호환 확인용 tag |
|---|---:|---|---|
| Qwen 3.5 | 9B | `Qwen/Qwen3.5-9B` | `qwen3.5:9b` |
| Qwen 3.5 | 4B | `Qwen/Qwen3.5-4B` | `qwen3.5:4b` |
| Granite 4.1 | 8B | `ibm-granite/granite-4.1-8b` | `granite4.1:8b` |
| Granite 4.1 | 3B | `ibm-granite/granite-4.1-3b` | `granite4.1:3b` |

위 ID는 `-Base`가 없는 post-trained/instruct checkpoint다. 구조화 instruction과 Tool
task를 빠르게 검증하는 현재 목적에는 Base보다 이를 우선한다. Qwen은 text-only SFT로
vision tower를 freeze하고 language backbone에만 LoRA를 적용한다. Family별 module 이름이
다르므로 LoRA target을 고정 문자열로 가정하지 않고 canary에서 trainable module과 parameter
수를 검증한다.

Qwen3.5의 기본 thinking은 모든 학습·Validation·Test에서 끈다. HF chat template에는
`enable_thinking=false`, Ollama 요청에는 `think=false`를 명시한다. 따라서
Summary+Reason의 한 문장 reason만 실험에서 측정하는 reasoning output이다.

4개 모델 × 4개 방법으로 16개 primary run을 만든다. 배관 canary는 이미 완료했으므로,
먼저 Granite 3B와 Qwen 4B의 8개 primary run을 실행한 뒤 Granite 8B와 Qwen 9B로
확장한다. 전체 16개를 1 seed로 비교한 후 유망한 설정만 추가 seed로 반복한다.

## 3. 학습 설정

- Split: Train S1–S80 / Validation S81–S90 / Test S91–S100
- Train: UPDATE 전부 + epoch별 NO_OP sampling, 기본 `UPDATE:NO_OP = 1:10`
  - NO_OP의 30%는 UPDATE 인접 hard negative, 70%는 전체 Train 랜덤 표본
  - 인접 표본은 ±1 Turn을 우선하고 목표량이 부족한 경우에만 ±2 Turn에서 보충
- Batch 구성: 독립 Turn 80% + 연속 32–128 Turn trajectory 20%
- 네 방법에 동일 sample ID, seed, optimizer step과 checkpoint 기준을 적용한다.
- 네 모델은 tokenizer별 실제 길이를 audit한 뒤 동일한 task context 상한을 사용하며,
  모델의 최대 128K/256K context를 학습에 그대로 사용하지 않는다.
- Validation/Test는 sampling 없이 원래 분포와 시나리오 순서를 유지한다.
- 학습 example 수를 우선 동일하게 하고 실제 input/output/training token은 별도 보고한다.

## 4. 학습 환경

하나의 prefix Conda 환경을 NVMe 경로에 만들고 모든 방법이 공유한다.

```bash
source /mnt/data/miniconda3/bin/activate
conda create -p /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft python=3.11
conda activate /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft
```

모델 cache, checkpoint, run, report, MLflow와 Ollama export는 아래 전용 루트에 저장한다.
canonical JSONL은 기존 위치를 참조하며 복사하지 않는다.

```text
/mnt/data/hj153lee/PalmClaw/on-device-memory-training/
```

`environment.yml`에는 CUDA 호환 PyTorch, Transformers, Datasets, Accelerate, PEFT,
TRL, sentencepiece, MLflow와 Ollama HTTP client를 고정한다. 첫 구현 단계에서 GPU/CUDA,
HF 다운로드, 1-batch forward/backward와 Ollama 연결 smoke test를 수행한다. Token과
credential은 환경변수로만 전달한다.

## 5. 학습 모니터링 웹 UI

MLflow local tracking server를 원본 metric/artifact 저장소로 사용하고, 연구용 통합
대시보드를 별도 웹 서버로 제공한다. 학습 코드는 UI에 직접 의존하지 않고 MLflow와 각
run의 `metrics.jsonl`, `status.json`만 기록한다. 따라서 웹 서버가 꺼져도 학습은 계속된다.

```bash
mlflow server \
  --backend-store-uri sqlite:////mnt/data/hj153lee/PalmClaw/on-device-memory-training/mlflow/mlflow.db \
  --default-artifact-root /mnt/data/hj153lee/PalmClaw/on-device-memory-training/mlflow/artifacts \
  --host 0.0.0.0 --port 5050
```

별도 UI는 FastAPI 기반 로컬 관리 서버로 구현하고 기본 포트는 `5060`으로 한다. MLflow
API와 로컬 상태 파일을 읽어 다음 화면을 제공한다.

- **Overview:** 4개 모델 × 4개 방법의 16개 run을 `queued/running/completed/failed`로
  표시하고 epoch/step 진행률, 경과시간과 ETA를 보여준다.
- **Learning curves:** run별 train/eval loss, learning rate, token/s와 GPU memory를
  시계열로 표시하고 여러 모델·방법을 한 그래프에 겹쳐 비교한다.
- **Validation:** UPDATE F1, NO_OP specificity, schema/apply 성공률, State F1,
  Turn/Final Quiz ESM·Tool F1·Arg Exact를 checkpoint별 표와 그래프로 보여준다.
- **Run detail:** 모델, 방법, seed, sampler, LoRA 설정, 데이터 manifest hash, Git commit,
  최근 로그, best checkpoint와 실패 원인을 보여준다.
- **Comparison:** 동일 model 내 4개 방법 및 동일 method 내 4개 모델을 선택해 best
  validation 성능과 token/latency를 나란히 비교한다.

학습 실행·중단은 tmux/controller가 담당한다. UI에서는 `RUNNING/QUEUED` 삭제를 차단하고
`COMPLETED/FAILED`만 확인 후 `trash/runs`로 이동한다. 이는 화면에서는 즉시 제거되지만
복구 가능한 archive이며 영구 삭제가 아니다. 화면은 10초마다 자동 갱신하고 완료된 run은
정적 요약 JSON으로도 내보낸다.

기록 원칙은 다음과 같다.

- 모든 run을 `model_family/model_size/method/seed/run_id`로 식별한다.
- step별 train/eval loss, learning rate, token/s, GPU memory와 ETA를 기록한다.
- Validation 지표와 checkpoint 선택 사유를 동일 run에 기록한다.
- dataset/quiz manifest hash, Git commit, LoRA·sampling 설정과 checkpoint를 artifact로
  남긴다.
- MLflow 장애가 학습을 중단하지 않도록 각 run에도 `metrics.jsonl`과 원자적으로 갱신하는
  `status.json`을 동시에 기록한다.

```bash
python -m memory_training.dashboard.server \
  --mlflow-uri http://127.0.0.1:5050 \
  --runs-root /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs \
  --host 0.0.0.0 --port 5060
```

## 6. Validation 전략

학습 중에는 checkpoint를 Ollama로 변환하지 않고 현재 HF/PEFT 모델을 직접 greedy
decode한다. Validation은 비용에 따라 세 단계로 나눈다.

1. 500 optimizer step마다 고정 512건으로 teacher-forced loss 측정
   - S81–S90 UPDATE 136건 전량
   - UPDATE 인접 hard NO_OP 188건과 seed 고정 random NO_OP 188건
2. epoch마다 UPDATE 전량과 동일 수의 NO_OP, 총 272건 one-step 생성 평가
3. epoch 종료마다 S81–S90 전체 closed-loop 평가
4. S91–S100 Test는 best checkpoint 확정 후 정확히 한 번만 실행

모든 생성은 고정 seed, `do_sample=false`로 실행한다. Best checkpoint는 closed-loop Quiz
ESM을 우선하고 trajectory State F1을 tie-break로 사용하되, False UPDATE가 사전 기준을
넘으면 제외한다. Test split은 checkpoint 선택에 사용하지 않는다.

## 7. 평가

1. **One-step memory:** Gold 이전 state를 입력해 UPDATE F1, NO_OP specificity,
   schema/apply 성공률과 Fact/State F1을 측정한다.
2. **Closed-loop memory:** Turn 1부터 예측 state를 전달해 trajectory/final State F1,
   최초 오류 Turn, 오류 지속·복구율을 측정한다.
3. **Quiz:** 동일 Agent로 Turn/Final Quiz의 ESM, Tool F1, Arg Exact, State F1을 측정한다.
4. **효율:** prefill/decode token, output token, peak memory, per-turn 및 Quiz latency를 측정한다.
5. **Reason ablation:** Summary+Reason의 reason-code 정확도와 추가 token/latency 대비
   Summary 및 Quiz 성능 변화를 비교한다.

주 결과는 **closed-loop Quiz 성능과 runtime 효율**, memory 평가는 원인 분석 지표로
사용한다. Summary의 ADD/REPLACE/DELETE는 이전/생성 fact map diff로만 사후 계산한다.

### Ollama Test

Validation으로 고른 checkpoint만 LoRA merge 후 **BF16 GGUF(무양자화)** 로 만들어 고유한
Ollama model tag로 등록한다. Test runner는 Ollama HTTP API에 연결해 S91–S100을 정확히
한 번 closed-loop로 실행하며 Ollama version, model digest, precision, context length,
temperature와 latency를 저장한다. 네 방법은 동일 base model·BF16·Quiz Agent 조건으로
비교한다. Q4_K_M 등 양자화 영향은 BF16 결과 확정 후 별도 ablation으로 측정한다.

## 8. 구현 순서

1. **완료:** Conda 환경·dependency lock과 GPU/HF/Ollama smoke test
2. **완료:** 공통 dataset/sampler와 네 target serializer
3. **완료:** Method별 parser/executor 및 단위 테스트
4. **완료:** HF/PEFT SFT runner, checkpoint/resume와 MLflow 기록
5. **완료:** 별도 통합 웹 UI와 16-run 진행 현황·loss·validation 시각화
6. **완료:** HF 기반 단계별 Validation과 best-checkpoint 선택
7. **완료:** Ollama export 및 one-step/closed-loop/Quiz Test
8. 네 방법의 성능·token·latency 비교 리포트

## 9. 구현 현황

### 1회차 완료

- 전용 Conda 환경: `/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft`
- 전용 산출물·cache 루트: `/mnt/data/hj153lee/PalmClaw/on-device-memory-training`
- 전체 276,938 Turn의 Summary/Patch/Delta 정렬, split과 SHA-256 검증 통과
- Turn Quiz 3,000개, Final Quiz 1,000개 연결·schema 검증 통과
- 8×A100 80GB 및 네 Hugging Face model repository 접근 확인
- Granite 4.1 3B와 Qwen3.5 4B tokenizer 및 LoRA 1-batch backward canary 통과
- 시스템 Ollama `0.6.5` 연결 확인 후 기존 서비스를 보존하고, 이후 Qwen3.5용 격리
  Ollama `0.32.15`와 별도 model store를 추가함

### 2회차 완료

- Summary/Patch/Delta의 동일 Turn을 묶는 SQLite byte-offset catalog 구현
- canonical JSONL을 복사하거나 메모리에 전부 적재하지 않는 worker-safe lazy dataset 구현
- scenario split 강제 및 source manifest 변경 시 stale catalog 자동 감지
- Train UPDATE 전량 보존 + epoch별 deterministic NO_OP `1:10` sampling 구현
- NO_OP 10,350개를 UPDATE 인접 3,105개(30%) + 일반 랜덤 7,245개(70%)로 계층화
  - ±1 Turn 후보 2,068개 전부 + 부족분 1,037개를 ±2 Turn에서 표본화
- 독립 sample에 추가해 전체 turn exposure의 약 20%가 되도록 32–128 Turn 연속 trajectory
  window 생성
- 공통 `MemoryMethod`의 format/parse/apply/materialize contract 정의
- 실제 catalog 276,938행 검증: Train `NO_OP 220,784 / UPDATE 1,035`, Validation
  `27,500 / 136`, Test `27,346 / 137`
- epoch 0 기준 독립 11,385 Turn + trajectory 2,909 Turn(38 windows) 생성 확인

### 3회차 완료

- 네 방법의 고정 system prompt, 입력 serializer와 strict JSON target schema 구현
- Summary/Patch/Delta target에서 `reason_code/reason` 제거, NO_OP은 `decision`만 decode
- Summary+Reason은 `reason_code → reason → decision → next_memory(UPDATE만)` 순서 보장
- Summary 전체 교체와 Patch의 exact-block `add/replace/delete` executor 구현
- Delta의 pending Patch 누적, materialized memory와 UPDATE 5회 deterministic compaction 구현
- markdown fence, extra field, 잘못된 operation/target과 state 변화 없는 UPDATE 거부
- 전체 276,938 Turn을 네 방법으로 replay하여 매 Turn canonical memory hash 일치 확인
- Delta compaction 291건(`UPDATE_INTERVAL`/`FINAL_FLUSH`) 별도 replay 검증
- 전체 평균 target 문자 수: Patch/Delta `21.19`, Summary `25.00`, Summary+Reason `126.24`

### 4회차 완료

- Qwen3.5는 `AutoModelForImageTextToText`, Granite 4.1은 causal LM loader를 사용하되
  vision 계층을 동결하고 language projection 7종에만 LoRA를 적용
- 각 모델의 native chat template를 사용하고 system/user prompt token은 `-100`으로 masking해
  assistant JSON 출력에만 SFT loss 적용
- epoch별 UPDATE 전량·계층화 NO_OP·연속 trajectory plan을 실제 DataLoader batch로 연결
- base model 전체가 아닌 LoRA adapter와 tokenizer만 checkpoint하고 optimizer, scheduler,
  Python/Torch/CUDA RNG 및 정확한 다음 batch 위치를 함께 저장하여 resume 구현
- 학습 loss, LR, target token/s, peak GPU memory, ETA를 `metrics.jsonl`과 원자적
  `status.json`에 저장하고 선택적으로 MLflow에 비차단 mirror
- teacher-forced loss, stratified one-step UPDATE/NO_OP·schema/apply·State F1,
  S81–S90 closed-loop trajectory/final State F1·최초 오류·복구 지표 구현
- Validation Quiz hook이 제공되면 ESM 우선·final State F1 tie-break, 없으면 closed-loop final
  State F1을 사용하며 False UPDATE 상한을 넘는 checkpoint는 best에서 제외
- Granite 4.1 3B Patch 1-step backward → adapter checkpoint(약 119MB) → optimizer/RNG resume
  → greedy one-step/closed-loop 생성 canary 통과
- 전체 단위 테스트 12건 및 Ruff 검사 통과

### 5회차 완료

- FastAPI 기반 read-only dashboard와 10초 자동 갱신 구현; 학습 실행·중단·삭제 API는 제공하지 않음
- **통합 Overview:** RUNNING/COMPLETED/FAILED/QUEUED 집계, stale heartbeat 감지,
  4개 모델 × 4개 방법 matrix 및 반복 run 개수 표시
- **다중 Run 비교:** 선택한 run들의 train/validation loss, closed-loop/final State F1,
  target token/s와 peak GPU memory를 동일 SVG 그래프와 요약 카드로 비교
- **개별 Run 상세:** 설정·sampler·manifest, 진행률/ETA, validation artifact,
  checkpoint 목록·용량, best 선정 근거와 실패 traceback 제공
- 동시 기록 중인 원자적 `status.json`과 append-only `metrics.jsonl`을 안전하게 읽고,
  마지막의 불완전한 JSONL line은 다음 refresh까지 무시
- 완료 run을 포함한 현재 전체 상태를 `/api/export` 또는 `--export-summary`로 JSON export
- 실제 run workspace 연결 API 검증, 경로 traversal 및 mutation 요청 차단 테스트 포함
- 전체 단위 테스트 14건 및 Ruff 검사 통과

### 6회차 완료

- 학습 중 HF/PEFT checkpoint를 직접 사용하는 teacher-forced, one-step, closed-loop
  Validation 경로를 공통 `validation.py`로 통합
- one-step은 Validation UPDATE 전량과 동일 수의 deterministic NO_OP 표본으로 decision,
  schema/apply, State F1과 reason ablation을 평가
- closed-loop는 S81–S90을 Turn 1부터 예측 state로 이어 최초 오류, 지속·복구율과 final
  State F1을 계산하며 Test S91–S100에는 접근하지 않음
- False UPDATE threshold를 넘는 checkpoint를 best 후보에서 제외하고, Quiz hook이 있으면
  ESM 우선·final State F1 tie-break, 없으면 final State F1로 선택
- epoch/step Validation 결과를 run artifact와 dashboard에 연결하고 greedy decode·고정 seed를 적용

### 7회차 완료

- Validation이 선택한 best LoRA adapter를 base model에 병합하고 BF16 GGUF를 양자화 없이
  고유 Ollama tag로 등록하는 export CLI 구현; Q4_K_M은 명시 옵션으로 유지
- 실행 전 converter/architecture/Ollama version/free disk를 검사하며, 성공 후 merged HF
  중간 파일만 안전하게 정리한다. BF16은 source GGUF와 Ollama blob이 함께 생기므로 한
  모델씩 export/test한다.
- S91–S100을 Turn 1부터 예측 memory로 이어가는 exactly-once closed-loop Test 구현;
  25 Turn마다 state를 checkpoint하고 중단 시 완료 호출을 반복하지 않고 재개
- Turn/Final Quiz 시점의 예측 memory snapshot을 고정하고, 모든 방법에 동일한 별도 Ollama
  Quiz Agent를 사용해 기존 VehicleMemBench simulator/scorer로 ESM·State F1·Tool F1·Arg
  Exact를 계산
- Memory/Quiz 각각 prefill·decode token, model latency와 Ollama version/model digest/context
  설정을 저장하며 reasoning type별 Quiz 결과도 함께 집계
- Test signature가 model/data/config 변화와 일치하지 않으면 과거 산출물 재사용을 거부하고,
  `--force`는 진단용으로만 허용
- Dashboard run 상세에 Ollama Memory final F1, Quiz ESM/Tool F1/Arg Exact와 두 단계 latency 연결
- Granite 4.1 3B 1-step canary를 실제로 병합·Q4_K_M 2.1GB 변환·Ollama 등록하고 HTTP 추론까지
  검증; canary는 배관 검증용이므로 성능 수치에는 사용하지 않음
- 시스템 llama.cpp/Ollama는 보존하고 `/mnt/data`에 최신 llama.cpp와 Ollama `0.32.15`를
  격리 설치; 별도 port `11435`, model store, GPU 6과 tmux session으로 운영
- 최신 converter에서 `Qwen3_5ForConditionalGeneration`/`Qwen3_5ForCausalLM` 지원을 확인하고,
  custom merge 변환에는 `--no-mtp`를 자동 적용해 NextN extra-block 불일치를 회피
- 공식 `qwen3.5:4b` Q4_K_M(약 3.4GB)을 격리 서버에 등록하고 GPU load 및 structured JSON
  추론 검증; 기본 thinking이 decode budget을 소비하지 않도록 평가 API는 `think=false` 고정
- 전체 단위 테스트 17건 및 Ruff 검사 통과

## 10. 다음 실행: Qwen3.5 4B Patch R1

### 실행 전 확인

- 이 run은 첫 **primary training**이며 기존 `phase4-canary-*`와 구분한다.
- 학습 GPU로 6번을 사용할 경우 Qwen smoke용 Ollama를 먼저 중지해 VRAM을 반환한다.
- canonical JSONL은 복사하지 않으며 run/checkpoint/cache는 모두 `/mnt/data`에 기록한다.
- Test S91–S100은 학습·Validation 중 보지 않고 best checkpoint 확정 뒤 정확히 한 번만 실행한다.
- 현재 free disk는 약 64GiB로 한 run에는 충분하지만 filesystem이 99% 사용 중이므로 학습 전
  preflight와 export 전 19.2GB peak disk 검사를 반드시 통과시킨다.

```bash
source /mnt/data/miniconda3/bin/activate
conda activate /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft

# GPU 6을 학습에 쓸 때만 실행한다.
python -m memory_training.ollama_isolated stop

python -m memory_training.preflight --mode full
python -m memory_training.model_canary --model qwen3.5-4b --mode metadata

CUDA_VISIBLE_DEVICES=<free-gpu> python -m memory_training.train \
  --model qwen3.5-4b \
  --method patch \
  --run-id qwen35-4b-patch-r1 \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --learning-rate 2e-4 \
  --max-length 4096 \
  --seed 42
```

기본값으로 UPDATE 전량, UPDATE당 NO_OP 10개, 인접 hard negative 30%, trajectory
window 20%가 적용된다. 매 500 optimizer step마다 고정된 512건의 teacher-forced
Validation과 checkpoint를 저장한다. 이 subset은 UPDATE 136건 전량, UPDATE 인접 hard
NO_OP 188건, seed 고정 random NO_OP 188건으로 구성되어 checkpoint 간 비교 편차와
JSONL 앞부분 편향을 막는다. epoch 종료마다 272건 one-step 및 S81–S90 전체
closed-loop를 수행해 best checkpoint를
선정한다. 중단 시 마지막 checkpoint의 `adapter/optimizer/scheduler/RNG/progress`를 함께
불러와 정확한 다음 batch부터 재개한다.

### 학습 후 순서

1. dashboard에서 loss, false UPDATE, final State F1과 best checkpoint 확인
2. 격리 Ollama 재시작: `python -m memory_training.ollama_isolated start`
3. best adapter를 `--ollama-url http://127.0.0.1:11435 --gguf-outtype bf16
   --quantization NONE`으로 plan → merge → BF16 등록
4. 모든 방법에서 동일한 고정 Quiz Agent tag를 사용해 S91–S100 Memory/Quiz Test 1회 실행
5. Qwen3.5 4B의 나머지 세 방법으로 확장한 뒤 방법 간 성능·token·latency 비교

### BF16 Test 실행 체크리스트

- 학습 중에는 GPU 6을 사용하므로 격리 Ollama `:11435`는 중지 상태가 정상이다. 학습 종료
  후 서버를 시작한다.
- epoch Validation이 만든 `best-checkpoint.json`을 확인하고, 아래 `plan`을 먼저 실행해
  architecture·converter·Ollama·disk 검사를 통과시킨다.
- primary Memory 모델은 `bf16 + NONE`을 고정한다. 기존 `qwen3.5:4b` Q4_K_M은 배관 smoke
  및 고정 Quiz Agent 후보일 뿐 Memory 모델의 BF16 결과와 섞지 않는다.
- BF16 source GGUF와 Ollama blob이 중복 저장되므로 **한 모델씩 export → Test → 보존/정리**한다.
  현재 약 59GiB 여유는 한 Qwen3.5 4B export에는 충분하지만 네 모델 동시 보존에는 빠듯하다.
- Test 결과는 run 아래 `ollama-test-summary.json`과 scenario별 checkpoint/JSONL에 저장되고,
  dashboard가 이를 읽는다. Test S91–S100은 checkpoint 선택에 사용하지 않는다.

```bash
source /mnt/data/miniconda3/bin/activate \
  /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft
cd /home/hj153lee/PalmClaw

python -m memory_training.ollama_isolated start

python -m memory_training.export_ollama \
  --run-dir /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/<run-id> \
  --ollama-url http://127.0.0.1:11435 \
  --gguf-outtype bf16 --quantization NONE --stage plan

python -m memory_training.export_ollama \
  --run-dir /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/<run-id> \
  --ollama-url http://127.0.0.1:11435 \
  --gguf-outtype bf16 --quantization NONE \
  --stage all --cleanup-intermediate --threads 16

python -m memory_training.ollama_test \
  --run-dir /mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/<run-id> \
  --ollama-url http://127.0.0.1:11435 \
  --memory-model <export-manifest의 ollama_tag> \
  --quiz-model <모든 방법에 고정할 Quiz Agent tag>
```
