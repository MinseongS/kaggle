# 작업 큐

기준점: exp014 (공개 0.480 fork, 공개 scorer 단독) **public 0.479**. 은메달 경계 0.480 (11/04).
중간 점검 11/15: 누수 없는 CV 에서 공개 규칙 대비 +0.01 이상 못 만들면 견고한 버전 선택으로 축소.

## 최우선: CV 하네스 점검 (health 🟥 CV↔LB, 2026-11-07)
- (h1) ~~**불가 (2026-11-07)**: commit 출력은 공개 smoke 12분자만 돌아 exp014/015/016 submission.csv 가 바이트 동일(md5 같음). 숨겨진 rerun 은 볼 수 없음 → 테스트 diff 로는 판정 불가, h2 로 대체.~~ 원래: exp014/015/016 커널 출력 submission.csv 를 비교해 테스트(1213 스펙트럼) 중 상위 25 리스트가 바뀐 분자 수와, 바뀐 분자가 어느 채널 경로(PubChem 승격/비승격)였는지 센다. 바뀐 분자가 몇 개뿐이면 LB 차이는 순수 노이즈 → sim2 근거 유지 가능.
- (h2) **[다음 실행 1순위]** sim2 에서 "채널 단독" 이 아니라 공개 최종 융합(pool/library 채널 + PubChem 승격 게이트 + 후처리)을 흉내 낸 최종 MRR 로 exp014 vs exp015 재측정. 최종 리스트 기준으로도 + 면 하네스는 맞고 LB 는 노이즈, − 면 하네스 교체.
- (h3) sim2 분자(NP 라이브러리) 와 테스트의 차이: 테스트 precursor 질량·adduct·instrument 분포와 sim2 분포 비교.

## 다음 제출 (CV 근거가 생긴 것만 위로)
0. ~~exp015 / exp016~~ **판정 2026-11-07 11:31**: exp015 0.475, exp016 0.473 (exp014 0.479 대비 −0.004/−0.006). 각각은 노이즈(±0.016) 안이지만 **둘 다 CV 와 반대 방향** → health 🟥 CV↔LB. sim2 채널 단독 MRR 이 최종 리스트 성능을 대변하지 못한다고 보고, 아래 "하네스 점검" 이 끝날 때까지 PubChem 채널 가중치 계열 제출은 중단. 최종 후보는 당분간 exp014.
1. **exp015-scorer2**: 삭제된 자체 scorer(`bgunshui/casmi26-scorer-train-public`)를 대신할 scorer 를 직접 학습해 0.480 의 `0.5·z(public) + 0.5·z(ours)` 앙상블 복원.
   - 구조·피처는 `sanpier/casmi-spectrum-model-v1` 의 `model.py`, `features.py`, `config.json` 그대로. 학습 코드는 sanpier 의 공개 노트북이나 디스커션에서 찾고, 없으면 model.py 기준으로 작성.
   - **CV 근거**: identity fold (`data/ext/panel/casmi26-identity-disjoint-validation-folds`) 한 fold 를 빼고 학습해서, 빠진 fold 의 timsTOF·NP 분자에서 공개 scorer 대비 후보 순위(같은 분자식 decoy 기준 MRR)가 같거나 좋으면 통과. 공개 scorer 는 전체로 학습됐을 수 있으니 비교는 "앙상블이 단독보다 나은가"로.
   - 커널: `kernels/scorer2_train` (GPU, 예산 확인) → 가중치를 private 데이터셋 `mins00/casmi26-scorer2` 로 → `kernels/sota0480` 복제본에 붙임 (`_swap_to_ours` 가 `config_used.json` 이 있는 `final_model.pt` 를 자동으로 찾음).

## CV 트랙
1. ~~종류② 시뮬레이션 생성기~~ **완료 2026-11-06** (1945분자, 정답 tier 수록 1619, `data/ext/sim2/`). 손 규칙 그리드 결과 → exp015/016. 남은 것: (a) ~~LambdaRank vs 손 규칙~~ **2026-11-06 23:30 완료 (`casmi.sim2_rank`)**: 전체 +0.08~0.10 이나 npx 에서 exp016 규칙보다 −0.05~−0.07 → 보류. 후속 (a2) npx 를 따로 검증하는 nested(npx 가중 학습, 또는 lsyn·np 계수만 손 규칙에 더하는 소형 그리드)로 npx 악화 없는 변형을 찾으면 다음 제출 후보. 원래 (a) 메모: (분자 5-fold, 열: z, ls, lp, syn, np, lotus, src, n_depositors, formula_isomers) — 손 규칙을 크게 넘으면 공개 채널 대체 커널 후보, (b) z 를 공개처럼 상위 5000 pass1 기준으로 재는 sim2 v2 (현재는 상위 ~650행 / 창 전체 두 근사뿐). 이력: 2026-11-05 23:30 커널 `mins00/casmi26-sim2-gen` v1 push (CPU, `kernels/sim2_gen/`, 분자 선택 `select.py`). (출력 수신·평가 완료)
   - 참고 (외부 결과, EXPERIMENTS `prior-oof-ext`): pool 창 out-of-fold 에서 NP 정답은 λ=0.15 최적(+0.075), 비NP 는 λ=0 최적 → 창 단위 NP 비율 등으로 λ 를 조건부로 바꾸는 규칙이 후보.: train 분자를 "공개 스펙트럼 없음, PubChem 에는 있음" 상태로 만들어 공개 파이프라인의 PubChem 채널 + prior 를 정직하게 측정.
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
