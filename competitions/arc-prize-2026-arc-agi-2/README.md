# arc-prize-2026-arc-agi-2

https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2

| 항목 | 값 |
|---|---|
| 유형 | nlp-llm / agent-sim (LLM TTT, program synthesis) |
| 평가지표 | test output 당 2 attempts, exact match. task 안에서 부분점수 (k/N) |
| 마감 | 2026-11-02 23:59 UTC (최종 후보는 10/28 까지 확정) |
| 제출 형식 | 코드 대회. 4×L4, 12h, 인터넷 X. 1회/일, 최종 2개 |
| Public/Private | 서로 **다른** 120 task 씩 (hidden 240개). 제출 1회에 함께 채점 |
| 메달 지급 | 예 |
| 팀 수 / 메달 컷 (9/28 public) | 2226팀 / 금 33.47, 은 32.22, 동 31.53 |
| Best local eval / Public | eval 은 누수로 신뢰 낮음 / public 30.69 (v8) |

## 전략
[research/00-strategy.md](research/00-strategy.md)

## 베이스라인
NVARC 2025 우승 솔루션 (Qwen3-4B `sorokin/qwen3_4b_grids15_sft139` + per-task LoRA TTT + DFS + augmentation scoring)
