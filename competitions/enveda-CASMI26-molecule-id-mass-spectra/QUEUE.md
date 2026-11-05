# 작업 큐

기준점: exp014 (공개 0.480 fork, 공개 scorer 단독) **public 0.479**. 은메달 경계 0.480 (11/04).
중간 점검 11/15: 누수 없는 CV 에서 공개 규칙 대비 +0.01 이상 못 만들면 견고한 버전 선택으로 축소.

## 다음 제출 (CV 근거가 생긴 것만 위로)
1. **exp015-scorer2**: 삭제된 자체 scorer(`bgunshui/casmi26-scorer-train-public`)를 대신할 scorer 를 직접 학습해 0.480 의 `0.5·z(public) + 0.5·z(ours)` 앙상블 복원.
   - 구조·피처는 `sanpier/casmi-spectrum-model-v1` 의 `model.py`, `features.py`, `config.json` 그대로. 학습 코드는 sanpier 의 공개 노트북이나 디스커션에서 찾고, 없으면 model.py 기준으로 작성.
   - **CV 근거**: identity fold (`data/ext/panel/casmi26-identity-disjoint-validation-folds`) 한 fold 를 빼고 학습해서, 빠진 fold 의 timsTOF·NP 분자에서 공개 scorer 대비 후보 순위(같은 분자식 decoy 기준 MRR)가 같거나 좋으면 통과. 공개 scorer 는 전체로 학습됐을 수 있으니 비교는 "앙상블이 단독보다 나은가"로.
   - 커널: `kernels/scorer2_train` (GPU, 예산 확인) → 가중치를 private 데이터셋 `mins00/casmi26-scorer2` 로 → `kernels/sota0480` 복제본에 붙임 (`_swap_to_ours` 가 `config_used.json` 이 있는 `final_model.pt` 를 자동으로 찾음).

## CV 트랙
1. **종류② 시뮬레이션 생성기** (최우선): train 분자를 "공개 스펙트럼 없음, PubChem 에는 있음" 상태로 만들어 공개 파이프라인의 PubChem 채널 + prior 를 정직하게 측정.
   - 분자 선정: identity fold 0~4 에서 고르게, 천연물 라이브러리(riken, gnps, msdial, massbank, mona, spectraverse) 위주 + enveda-np-examples, test 질량 분포(157~1159, 중앙 348)에 맞춰 약 2000개. test adduct 10종 스펙트럼만.
   - FP 점수: 분자가 속한 fold 를 뺀 `dmitriigluzdov/casmi26-fold-safe-fpnet` 의 `fpnet_fold{k}.pt` (out-of-fold).
   - 후보: `casmi26-pubchem-popularity-prior` 의 PubChem tier (질량순 1.06억 행) 10ppm 창. 후보마다 FP 점수, pop/lsid/lpmid/np/LOTUS/동의어/src_mask/depositor 수 등을 붙이고, 분자별 상위 약 500개 + 정답 행만 parquet 로 내보냄.
   - Kaggle 커널 `kernels/sim2_gen` (FPNet 추론에 GPU, 예산 확인), 출력 내려받아 `data/ext/sim2/`.
   - 로컬: `casmi.panel_stack` 과 같은 방식으로 손 규칙(`z + 0.15·prior`, 가중 0.25/0.5) vs LambdaRank. 이 결과로 공개 노트북의 prior 가중치·승격 게이트가 덜 유명한 분자에서 맞는지 판정.
2. 종류① (라이브러리에 있는 분자) 시뮬레이션: 같은 분자의 다른 라이브러리 스펙트럼만 남기고 timsTOF 스펙트럼을 질의로.
3. 공개 파이프라인 전체를 시뮬레이션 분자에 돌려 모든 채널 evidence 를 모으는 것 (stacker 본게임). 1번 결과를 보고 범위 결정.

## 백로그
- ICEBERG/GLACIER 점수의 out-of-fold 정직성 확인 (MassSpecGym 학습 분자 편향, 10/01 sim-iceberg 결과 참고).
- 공개 0.480 후처리 층(w_mid 0.75, fragment leader move, DocPrior 2~25위)이 CV 에서도 이득인지. public LB 로만 고른 층이라 private 위험.
- 종류③ (de novo): 상위 후보의 메틸/당 편집 유사체를 16~25위에만.

## 완료
- 2026-11-04 exp014 기준점 제출 → public 0.479.
- 2026-11-04 panel-stack: panel 은 prior 만으로 0.97 → 판별력 없음 (EXPERIMENTS.md).
