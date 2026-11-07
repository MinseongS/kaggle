# name-that-llm

https://www.kaggle.com/competitions/name-that-llm

| 항목 | 값 |
|---|---|
| 유형 | nlp-llm (짧은 텍스트 18-클래스 저자 판별) |
| 평가지표 | Balanced accuracy (클래스별 recall 평균). public/private의 클래스 구성이 서로 다름 |
| 마감 | 2027-01-13 23:59 UTC (팀 병합 마감일은 미공개) |
| 제출 형식 | 코드 대회: CPU/GPU 노트북 9시간 이하, 인터넷 X, `submission.parquet` (`id`, `model`). 외부 공개 데이터·사전학습 모델 허용 |
| 메달 지급 | 예 (Featured, $50k) |
| 팀 수 / 메달 컷 | 4,942 (2026-11-07) / 금 0.706 (19위), 은 0.669 (247위), 동 0.668 (494위) |
| Best CV / LB | 0.5583 (e011, 자체 lite 계열) / **0.669** (s009·s010, 기준 s008 ozer672 0.668), 약 153위 / 4,942팀 (2026-11-07). 채점 중: s013 areeb670, s014 ozer672+DeBERTa 멤버. 금 컷 0.706 (+0.037), 은 컷 0.669 |

## 데이터
- `train.csv`: 2,699개 지문, 150개 프롬프트 (`id`, `prompt_id`, `text`, `model`)
- `test.csv`: 숨은 재실행에서 약 4,000개. 학습셋에 **없는** 프롬프트에 대한 답변
- 지문 길이 40~250단어 (무작위)
- 라벨 18개:
  - OpenAI: gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra
  - Google: gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash
  - Anthropic: claude-fable-5, claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5
  - 기타: qwen3.8-max, glm-5.3, glm-5.3-flash, kimi-k3

## 검증
- `prompt_id` 기준 GroupKFold (테스트 프롬프트가 학습셋에 없으므로)

## EDA 핵심 (2026-11-04)
- 150개 프롬프트 × 18개 모델이 **정확히 1개씩** (claude-fable-5만 149). 클래스 완전 균형
- 길이 분포는 모델 간 거의 같음 → 길이로 구분 불가
- 형식 단서: Claude·GLM은 곧은 따옴표만, em dash 많음 / Gemini는 em dash 거의 0 / GPT·Kimi는 둥근 따옴표(’ “)
- 헷갈리는 묶음: Claude ↔ GLM(서로 섞임), gemini-3.6/3.7/3.8-flash, gpt-5.5/5.6-*, kimi·qwen ↔ GPT
- **프롬프트 구조 활용**: 같은 프롬프트 답변끼리 묶고 그룹 안에서 1:1 매칭하면 0.34 → 0.44 (오라클). 테스트 그룹은 TF-IDF 클러스터링만으로 ARI 0.93

## 참고한 과거 솔루션
- (조사 예정) LLM - Detect AI Generated Text, LLM Prompt Recovery 등

## 현재 계획 (제약: 무료 자원만 — Kaggle 무료 GPU + 로컬 M4 Pro 48GB, 유료 API·클라우드 X)
자료조사는 `RESEARCH.md`. 목표: 은메달(상위 5%) 확보, 여유 있으면 상위 진입.
1. 로컬(CPU/MPS): Open Pool 기반 CV 하네스 (grouped CV + m=2 컷 + LB 노이즈 계산기), n-gram/TF-IDF/SVM 멤버, 후처리(Sinkhorn·typography) 튜닝
2. Kaggle GPU: 공개 0.667 파이프라인 재현 제출 → 기준점
3. Kaggle GPU 쿼터로 공개 앙상블과 **다른 데이터/구조**의 멤버 1~2개 학습 (다양성), 계열 내 전용 모델
4. 공개 LoRA 가중치·데이터셋 최대 활용, 직접 생성은 하지 않음

## 제출 전략 (하루 5회, 채점 약 8시간)
- 매일 5회를 모두 쓴다. 단, 각 제출은 **질문 하나에 답하는 실험**이어야 함 (한 번에 변경 1개)
- 판정 기준: 같은 코드 재제출 노이즈가 ±0.009 → **+0.010 이상만 개선으로 인정**. 그 이하는 반복 제출하거나 CV로 판단
  - 두 제출의 차이 sd ≈ sqrt(f/1200) (f = 둘 중 한쪽만 맞힌 비율, OOF로 미리 계산)
- 트랙
  1. 메인: 공개 0.667 파이프라인(`kernels/pub667`) + **우리 멤버 추가** (다른 데이터·구조 → 다양성)
  2. 측정: 우리 단독 모델 제출로 CV와 LB의 차이 보정 (`kernels/lite`)
- 최종 2개 선택: public 순위보다 CV + LB 일관성 기준
