# Tabular
- 기본기: GBDT 3종 (LightGBM / XGBoost / CatBoost) + target/count encoding + 집계 feature
- 강한 신호는 feature engineering에서 나온다. 모델 튜닝은 마지막에.
- 시계열이면 시간 분할 CV, 누수(lag feature 계산 시점) 주의
- 마무리: 다양한 모델 OOF로 hill climbing 또는 ridge stacking
