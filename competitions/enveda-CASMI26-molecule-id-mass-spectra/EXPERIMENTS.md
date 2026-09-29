# Experiments

| date | exp | change | CV | LB | note |
|---|---|---|---|---|---|
| 2026-09-28 | exp000 | prvsiyan 0.335 설정 포팅 (`uv run casmi cv --n-np 250 --n-e180 250 --tag exp000`). 4채널 + HistGBM 8개, ranker 는 CV fold 안에서 재학습 | np-examples 250: A 0.796 / B 0.619 / C 0 → **가중 0.406**. e180 250: B 0.972 (FPNet 누수) → 가중 0.565. 전체 혼합 0.486 | — | 제출 파일 `outputs/submission.csv` 는 rank_train.npz ranker (원본과 동일). CV 피처 계산 1.28 s/분자 (10분 40초 / 500분자), ranker CV 16분, test 예측 400분자 5분 11초 (setup 포함 6분 24초), RSS 3.4GB |
| 2026-09-28 | exp000-noFP | exp000 에서 FPNet 피처 0 으로 두고 ranker 재학습 (같은 피처 캐시) | np: A 0.799 / B 0.590 → 가중 0.394. e180: B 0.896 → 0.532 | — | FPNet 기여 (np, B): +0.028. e180 은 FP 없이도 0.90 → e180 은 쉬운 분포 (합성 라이브러리의 가까운 analog), 판단 기준으로 쓰지 말 것 |
| 2026-09-28 | exp000-ref | 같은 피처에 rank_train.npz ranker (누수 참고용) | np: A 0.834 / B 0.674 → 가중 0.437 | — | fold 재학습 대비 np B +0.055 → rank_train.npz 는 np-examples 로 학습된 것으로 보임. CV 에서 쓰면 안 됨 |
| 2026-09-29 | exp000-sub | exp000 을 Kaggle 노트북으로 제출 (rank_train.npz ranker) | 가중 0.406 | **0.331** | 원본 0.328~0.335 재현. Kaggle 빌드 33분 + 예측 21분. ref 56658747 |
| 2026-09-29 | exp001-sub | ranker 를 누수 없는 CV rows (exp000 feats) 로 다시 fit | — | pending | CPU 세션 (GPU batch 세션 2개 한도). ref 56662075 |
