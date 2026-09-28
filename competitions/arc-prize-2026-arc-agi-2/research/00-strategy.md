# ARC-AGI-2 은메달 전략 (2026-09-28 기준)

근거 문서: [01 과거 솔루션](01-past-solutions.md), [02 대회 규칙·디스커션](02-competition-and-discussions.md), [03 공개 노트북 분석](03-notebook-analysis.md)

## 1. Public ↔ Private 판단

| 사실 | 의미 |
|---|---|
| Public 120 task, Private 는 **다른** 120 task. 제출 1회에 240개를 함께 채점 | Public 은 private 의 "샘플"조차 아님. 작은 차이는 전달되지 않음 |
| 1 task ≈ 0.83점. 한 task 안에서 부분점수 가능 (0.28 = 3-output task 의 1/3) | 금 컷(33.47)과 은 컷(32.22)의 차이가 task 1.5개 |
| 같은 코드를 다시 제출해도 28~33.5점 (std ≈ 2.2). 원인: `hash(str)` seed, bf16 TTT 비결정성 | 공개된 LB 점수는 여러 번 중 **최고치**. 평균은 더 낮음 |
| 2025: 우승팀 local 30.5 → public 27.6 → private 24.0. 상위권은 public 의 76~87% 유지, 중위권은 55~60% | Private 은 2~4점 낮게 나옴. 공개 노트북을 fork 한 약 1000팀은 private 에서 무작위로 섞임 |

**결론:** 방법 차이가 크면(예: +5점 이상) private 까지 전달된다. 1~3점 차이는 노이즈다. 은메달(private 약 111등)을 노리려면 fork 무리의 **평균**보다 확실히 3~5점 높아야 한다.

## 2. 핵심 제약
- 제출 1회/일 → 마감(11/02)까지 약 34회. 최종 선택 2개
- 4×L4, 12h, 인터넷 X. 베이스라인이 약 10h 사용 (여유 약 2h)
- **로컬 CV 의 문제:** 공개 NVARC 모델은 evaluation 120개로도 학습됐음 → eval 점수는 부풀려짐. 추론 쪽 변경(선택 로직, 예산 배분)의 **상대 비교**에만 쓰고, 과제별로 짝지어 비교(paired)한다
- Kaggle GPU quota: L4 는 2배로 소모 → 4×L4 로 eval 1회에 quota 5~10h. 로컬 eval 은 주 2~3회가 한계. 필요하면 외부 L4/A100 을 시간 단위로 빌림

## 3. 실행 계획

### Phase 0 — 재현 가능한 베이스라인 (9/29 ~ 10/2)
1. NVARC 베이스라인 fork → `hash()` seed 를 `zlib.crc32` 로 교체, `PYTHONHASHSEED` 고정, task 별 seed 고정
2. 시간 가드: 남은 시간 기반 예산 배분, 모든 task 에 최소 decode 보장, 빈 attempt 는 fallback 으로 채움
3. 같은 설정을 2회 제출 → 재현성 확인 (점수 차이가 줄었는지)
4. eval 셋 로컬 평가 파이프라인 (과제별 결과 저장 → paired 비교)

### Phase 1 — 기대값을 올리는 저비용 개선 (10/3 ~ 10/20)
우선순위 순, 한 번에 하나씩 로컬 eval 로 검증한 뒤 제출:
1. **NVARC 가 대회 후 공개한 re-scoring**: DFS 발견 빈도 + augmentation log-prob 기하평균
2. **남는 2h 활용**: TTT step / DFS 후보 / scoring augmentation 수 증가
3. **Self-ensemble**: seed 나 checkpoint 2개의 후보 합집합을 함께 scoring (MindsAI: 같은 compute 에서 샘플 2배보다 나았음)
4. 8-view 배치 decoding 으로 속도 개선 → 늘어난 시간을 1~3에 재투자
5. (선택) CPU 에서 싼 DSL/규칙 solver 병렬 실행 → 검증된 프로그램이 나오면 attempt 로 사용

### Phase 2 — (예산이 허용될 때만) 합성 데이터 continual FT
- NVARC 의 공개 합성 퍼즐 + 생성기로 LoRA / 짧은 continual FT. 가장 확실한 개선 방향이지만 비용이 듦 → Phase 1 결과를 보고 결정

### Phase 3 — 최종 선택 (10/21 ~ 10/28 확정)
- 최고 public 이 아니라 **(local eval + 여러 번 제출한 public 의 평균)** 기준으로 고른다
- 최종 2개: ① 기대값이 가장 높고 안정적인 설정 ② 성격이 다른 강한 설정 (다양화)
- 마감 주에는 큐 지연과 timeout 사례가 있으니 10/28 까지 확정

## 4. 하지 않을 것
- 공개 노트북 seed 만 바꿔 돌리며 public 최고점 쫓기 (private 에서는 추첨)
- 순수 zero-shot LLM/에이전트 (약 14% 이하), CompressARC 단독, 큰 모델 새로 사전학습

## 5. 현실적 기대치
은메달은 **목표치이지 예상치가 아니다.** Phase 1 에서 로컬 eval 과 public 평균 모두 +3점 이상 나오면 은메달 가능성이 의미 있게 생긴다. 그렇지 못하면 경험(TTT·LLM 추론 최적화)을 얻는 것으로 만족하고 enveda 로 무게를 옮긴다.
