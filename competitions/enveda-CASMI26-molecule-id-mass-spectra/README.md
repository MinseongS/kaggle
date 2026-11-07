# enveda-CASMI26-molecule-id-mass-spectra

https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra

| 항목 | 값 |
|---|---|
| 유형 | science (MS/MS → 분자 식별, 후보 검색 + ranking) |
| 평가지표 | MRR@25, 정답 기준은 tautomer 정규화 InChIKey14 |
| 마감 | 2026-12-14 (엔트리·병합 12/07) |
| 제출 형식 | 코드 대회, 9h, 인터넷 X. 5회/일, 최종 2개 |
| Public/Private | 33% (약 130 분자) / 67% |
| 메달 지급 | 예 |
| 팀 수 / 메달 컷 (11/07 11:31 public) | 4893팀 / 금 0.488, 은 0.480 (244위), 동 0.480 (1등 0.537). 0.480 대량 동점 = 공개 노트북 |
| Best CV / LB | 공개 SOTA fork exp014 LB **0.479** (943위). sim2 PubChem 채널 MRR@25: 공개 규칙 0.481 → exp016 규칙 0.520 이지만 LB 는 exp015 0.475 / exp016 0.473 (역방향, 하네스 점검 중). 자체 파이프라인 CV 0.427 (exp013) |

## 전략
[research/00-strategy.md](research/00-strategy.md) (9/28 원안), [research/05-public-sota-0480.md](research/05-public-sota-0480.md) (11/04 공개 SOTA 분석, 현행)

11/04 재개 계획: 공개 0.465/0.480 파이프라인을 기준점으로 삼고, 모든 채널의 후보 evidence 로 학습한 stacker 로 수작업 융합 규칙을 대체. **11/15 중간 점검**: 누수 없는 CV 에서 공개 규칙 대비 +0.01 미만이면 견고한 버전 선택으로 축소.
