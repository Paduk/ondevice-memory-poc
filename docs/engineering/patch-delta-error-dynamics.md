# Patch–Delta Error Dynamics

이 분석은 고정 Test의 Patch와 Delta-v3 decision trace를 재생하여 메모리 오류가
UPDATE 시점에 발생하는지, 이후 NO_OP 또는 다른 UPDATE에서 발생하는지를 구분한다.
분석 단위는 전체 원본 발화가 아니라 UPDATE:NO_OP=1:5로 고정된 평가 trajectory이다.

## Coverage

| Method | Model-scenarios | Turns | Quizzes |
|---|---:|---:|---:|
| Patch | 141 | 13,842 | 5,640 |
| Delta-v3 | 141 | 13,842 | 5,640 |

전체 144개 model–scenario 조합 중 141개가 방법별로 포함된다. 제외된 세 조합은
Patch와 Delta-v3의 Quiz 정오가 모두 일치하여 기존 불일치 snapshot 요청에 포함되지 않은
시나리오이다.

## Update Acquisition and Retention

| Metric | Patch (%) | Delta-v3 (%) |
|---|---:|---:|
| Gold UPDATE transition success | 35.76 | 36.28 |
| Introduced fact acquired immediately | 34.07 | 34.74 |
| Acquisition failures: missed UPDATE | 32.15 | 40.55 |
| Acquisition failures: incorrect UPDATE content | 66.14 | 58.28 |
| Acquisition failures: invalid output | 1.71 | 1.17 |
| Retired fact removed immediately | 92.75 | 91.60 |
| Acquired fact retained until retirement/end | 96.43 | 97.02 |
| First loss on a later NO_OP | 0.26 | 0.00 |
| First loss on a later UPDATE | 3.30 | 2.98 |

Fact acquisition은 Gold UPDATE에서 새로 도입된 정답 fact version이 같은 턴의 예측
메모리에 존재하는지로 정의한다. Retention은 즉시 획득된 fact가 Gold에서 유효한 동안
처음 누락되는 시점을 추적한다.

## NO_OP Dynamics

| Metric | Patch (%) | Delta-v3 (%) |
|---|---:|---:|
| False UPDATE on Gold NO_OP | 2.23 | 1.21 |
| Memory mutation on Gold NO_OP | 2.22 | 1.20 |
| New loss of at least one Gold fact | 0.02 | 0.00 |
| Introduction of at least one extra fact | 2.19 | 1.19 |

## Key Finding

주된 오류는 정답 UPDATE가 처음 등장할 때 발생했으며, 이후 NO_OP 구간에서의
정보 소실은 거의 없었다. 정확히 획득된 fact의 96--97%는 교체·삭제 또는 시나리오
종료까지 유지되었다. Gold NO_OP에서 발생한 false UPDATE도 대체로 기존 정답 fact의
삭제보다 불필요한 fact의 추가로 이어졌다.

## Memory State and Quiz Outcome

| Method | Covered quizzes | Exact fact state & correct | Exact fact state & wrong | Inexact fact state & correct | Inexact fact state & wrong |
|---|---:|---:|---:|---:|---:|
| Patch | 5,640 | 244 (4.33%) | 9 (0.16%) | 3815 (67.64%) | 1572 (27.87%) |
| Delta-v3 | 5,640 | 318 (5.64%) | 15 (0.27%) | 3524 (62.48%) | 1783 (31.61%) |

`Exact fact state & wrong`은 엄격한 reader-only 오류의 하한이다. 반대로 전체 fact state가
부정확하더라도 정답 호출에 필요한 fact는 존재할 수 있으므로, `Inexact fact state & wrong`을
모두 memory writer 오류로 해석하지 않는다.

## Interpretation Boundaries

- Fact 비교는 owner, timestamp, 구조화된 payload의 정확 일치를 사용한다.
- NO_OP 중 기존 오류가 그대로 남은 경우는 신규 소실로 세지 않는다.
- Later-UPDATE loss는 다른 선호 갱신이 기존 fact를 훼손한 경우를 포함한다.
- 결과는 현재 확보된 141개 paired model–scenario에 대한 기술 통계다.
