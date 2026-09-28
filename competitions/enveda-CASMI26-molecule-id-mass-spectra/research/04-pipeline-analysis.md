# 04. 공개 파이프라인 분석 (prvsiyan analog-propagation 계열)

분석 대상: `ref-notebooks/prvsiyan_analog-propagation-casmi-2026-baseline` (37셀, public 0.335 / 같은 코드 재실행 0.328),
비교 대상: `haideptry_…fast-spectral-cosine-baseline` (0.339), `ahmedberatozer_casmi26-v4f-inference` (0.362).
포팅 코드: `src/casmi/` (이 문서의 절 번호와 모듈 대응은 마지막 절 참고).

---

## 1. 전체 흐름

```
test 스펙트럼 (분자당 1~16개)
  │  adduct 되돌려 중성 질량 → 분자의 target = 스펙트럼별 중성질량의 중앙값
  ├─ ch1 라이브러리 검색   : train 2.54M 스펙트럼 중 target ±10ppm (분자식 질량으로 인덱싱) → entropy similarity
  ├─ ch2 analog 전파       : 구조당 대표 스펙트럼 1개(timsTOF 우선) 중 ±200 Da → 질량 이동 entropy sim, 상위 200개
  ├─ ch4 FPNet             : 스펙트럼 → 6,930비트 지문 logit z (per-spectrum 모델 s2 + merged 모델 m1 평균)
  │
  ├─ 후보 = pool(COCONUT ∪ train 구조) 중 exact mass ±10ppm (없으면 ±30ppm), cap 500
  ├─ ch3 MetFrag-lite      : 후보별 1~2 결합 절단 조각이 관측 피크 강도를 얼마나 설명하는지
  ├─ 31개 피처 (rank_features)
  ├─ HistGBM 8개 (W1 prior 2 × seed 4) 평균 확률
  └─ 상위 25개 SMILES
```

모든 채널은 "후보 창 안의 같은 질량 후보들" 사이의 순서를 정하기 위한 것이다. 후보 풀 자체는 작게 유지한다(창 안 중앙값 약 56개).

## 2. 후보 생성

| 항목 | 값 | 근거 (노트북 측정) |
|---|---|---|
| 풀 | COCONUT 2.0 (첨부 데이터셋 기준 436,389 구조) ∪ train 구조 275,810, InChIKey14 dedup | COCONUT 이 enveda-np-examples 99.6% 커버 |
| 전처리 (COCONUT) | 가장 큰 fragment, 전하 제외, 100–1300 Da, 중원자 5–120 | |
| 창 | exact mass ±10 ppm (상대), 비면 ±30 ppm | 5ppm recall 0.972, 8.5ppm 0.992, 10ppm 1.000. ±20ppm 은 C2 MRR 0.521→0.509 |
| cap | 500 (런타임 보호용, 사실상 안 걸림). 걸리면 `lv*100 + z(f·z)` 순 | cap 80 (질량순) 은 Class-2 정답 24% 삭제 → LB −0.053 |
| 지문 | ECFP4(4096)‖ECFP6(4096)‖RDKitFP(2048, maxPath 6)‖MACCS(167) 중 train 구조에서 빈도 0.5%~99.5% 인 6,930비트 (`fp_bits.npy`), bit-pack | |
| 확장 | ChEBI+LIPID MAPS (`USE_BIO_DB`) 기본 off. PubChem 확장 (`PC_TOPK`) 기본 off | PubChem top-50 → LB 0.205 (−0.13) |

주의: 풀 키는 **train 의 `inchikey14` (tautomer 정규화 전)** 이다. 그래서 같은 metric 키를 가진 중복이 창 안에 들어올 수 있다
(포팅본은 출력 단계에서 metric 키로 dedupe).

## 3. 채널

### ch1 라이브러리 검색 (Class 1 증거)
- 스펙트럼 정리: base peak 의 0.2% 미만 제거 → 강도 상위 256개 → 강도^1 정규화 → **entropy weighting** (스펙트럼 엔트로피 S<3 이면 강도^(0.25+0.25S)).
- 유사도: Li et al. 2021 spectral entropy similarity, 피크 매칭 허용오차 0.01 Da.
- 라이브러리 인덱스는 `molecular_formula` 로 계산한 **분자식 질량** (없으면 precursor+adduct). riken 의 0.005 Da 정밀도, gnps adduct 오라벨에 면역. Class-1 0.9206→0.9253.
- 분자의 여러 스펙트럼 × 창 안 라이브러리 스펙트럼 → 구조(InChIKey14)별 max. 후보 피처 `lv`.
- Class-1 시뮬레이션 MRR ≈ 0.93. library-only 제출 0.151 → Class 1 비중 ≈ 16% 추정의 근거.

### ch2 mass-shifted analog propagation (Class 2 증거, 핵심 아이디어)
- 구조마다 대표 스펙트럼 1개: **test 장비(timsTOF)와 같은 장비 우선, 그다음 raw 피크 수가 많은 것** (c2 0.5241→0.5412).
- target ±200 Da 안의 대표 스펙트럼들과 `max(직접 entropy sim, 질량차만큼 이동시킨 entropy sim)`.
- 상위 N_ANALOG=200 analog (sim 내림차순). 후보 점수 `ap = max_a sim(a)^3 · Tanimoto(f_c, f_a)`.
- Class-2 holdout: 랜덤 0.138, 질량오차 0.164, **analog 0.521**. Class 1 분자는 자기 자신이 shift 0, Tanimoto 1 로 잡혀 자동으로 1등.

### ch3 MetFrag-lite
- RDKit 그래프에서 결합 1개, 2개를 끊어 나오는 연결 성분의 중성 질량 집합 (결합 34개 초과면 분자 전체 질량만).
- 이온 = 조각 + (−2..+2)H ± proton. sqrt 강도 기준으로 0.01 Da 안에 설명되는 비율, 분자의 스펙트럼들 중 max.
- 단독 C2 MRR 0.259, analog 와 상관 0.058 → 합치면 0.545. 이성질체 그룹 내부 MRR 0.278 (랜덤 0.169).
- 비용 약 14 ms/후보 (포팅본은 SMILES 별 캐시).

### ch4 FPNet (spectrum → fingerprint)
- 아키텍처: 피크 토큰 = Linear([sinusoidal(m/z) ‖ sinusoidal(precursor−m/z) ‖ sqrt 강도]) (d=512), 전역 토큰 = sinusoidal(precursor) + CE/100 + mode + log1p(prec)/10 + adduct embedding(26) + 장비군 embedding(5). Pre-LN Transformer 6층 8헤드, 출력 = [CLS ‖ 피크 평균] → MLP(2048) → 6,930 logit. 약 3,600만 파라미터 (체크포인트 144MB, step 24,000).
- 입력 전처리: precursor+1.5 초과 제거, 0.1% floor, 50 Da 버킷당 8개로 다양화한 상위 128 피크, sqrt(강도/최대).
- 점수: `f·z` (Bernoulli 로그우도에서 후보 무관 상수를 뺀 것과 정확히 같음).
- 학습 (노트북 부록 스크립트, 실행은 안 함): BCE(z, f_true) + λ·softmax CE (정답 1 + 같은 ±10ppm 창에서 뽑은 decoy 63개, 점수는 창 평균을 뺀 f·z/√nbits). AdamW lr 3e-4, warmup 2k, cosine, bs 256, fp16 autocast, grad clip 1. 증강: 피크 dropout 0~30%, 강도 log-normal jitter σ=0.25, m/z ±5 ppm. `merge_p=0.6` 이면 60% 입력을 같은 구조 스펙트럼 2~4개 병합으로 대체. **val hard-negative top-1 이 가장 좋은 체크포인트만 저장** (12~20k step 후 과적합으로 붕괴: 0.457 → 69k 에서 0.127). T4 약 2시간.
- 두 모델: `fp_single_s2` (단일 스펙트럼 학습 → 스펙트럼별로 넣고 평균), `fp_merged_m1` (병합 학습 → 분자의 모든 피크를 하나로 병합해 넣음). 최종 z = 두 view 평균. 잘못된 view 로 넣으면 −0.02. 이 **두 view 조합이 LB +0.019** 로 노트북에서 측정된 가장 큰 효과.
- 단독 C2 MRR 0.468~0.517, 4채널 ranker 0.612~0.63.
- FPNet 의 train/val 분할 정보(`is_val`, 109,985 val 스펙트럼)는 공개되지 않았다. 250 NP holdout 은 val 쪽에 있었다고만 적혀 있음.

## 4. Ranker

### 31개 피처 (`features.py`, 순서 고정)
| # | 피처 | 설명 |
|---|---|---|
| 0–4 | lv, rank, lvmax, lv−lvmax, lv>0 | 라이브러리 유사도 |
| 5–8 | ap, rank, apmax, ap−apmax | analog 전파 (sim^3·Tanimoto) |
| 9–13 | a1, best_tan, top_tan, mean_tan, top_sim | sim^1 버전, 최대 Tanimoto, 1등 analog 와의 Tanimoto, 가중평균, 1등 analog sim |
| 14 | log(n_cand) | |
| 15–20 | z(f·z), rank, f·z−max, z(f·z/√bits), rank, is_max | FPNet |
| 21–24 | fr, rank, fr−max, z(fr) | MetFrag-lite |
| 25–30 | lv·(1−mr), ap·(1−mr), agree_lib, agree_analog, agree·lvmax, corr(lv, −mr) | 채널 교차 일치 |

rank/z 는 **후보 집합 기준 상대값**이다 → 추론 때 후보 집합을 바꾸면(cap 등) 피처 분포가 바뀐다.
풀 출처(`src`)·NP-likeness 는 의도적으로 뺐다 (class-2 시뮬레이션에서 정답은 항상 train 구조라 누수, 0.94 가짜 점수).

### 학습
- `HistGradientBoostingClassifier(max_depth=6, max_iter=500, lr=0.03, min_samples_leaf=80, l2=1.0)`.
- sample weight: class-1 시뮬레이션 행(M=0)에 w1, class-2 행(M=1)에 1−w1. w1 ∈ (0.30, 0.60) × seed (0,1,2,3) = 8개 평균.
- W1 은 class 비율(0.16)이 아니라 LB 로 보정한 값: library-only 0.151, 전체 0.233 두 점으로 풀어 w1≈0.42 → 이후 (0.30, 0.60) 헤지로 고정.
- seed 만 바꿔도 LB ±0.007 (HistGBM 기본 random_state=None). 그래서 seed 고정 + bagging.
- 깊이 10, 좁은 prior 등은 로컬(쿼리 hold-out 5-fold 포함)에서 좋아 보였지만 LB 에서 떨어짐 → 노트북 결론: "로컬 숫자는 고장 검출용, 설정 선택용 아님".

### rank_train.npz 는 어떻게 만들어졌나
- 노트북 밖 `make_ranker3.py` 로 만든 결과물만 첨부. 스크립트는 비공개.
- 파일 내용 (직접 확인): `X (142,762 × 31)`, `Y`, `M`, `G`. **분자 식별자(InChIKey/SMILES) 없음.**
- 819 쿼리 그룹 × 2 시뮬레이션 (M=0 class-1, M=1 class-2), 각 71,381 행. 모든 그룹이 두 시뮬레이션을 다 가지며, 그룹당 후보 수가 두 시뮬레이션에서 같다 (같은 후보 창, 증거만 다름). 그룹당 정답 행은 정확히 1개 (= tautomer 정규화 없이 InChIKey14 로 라벨링했을 가능성).
- 그룹 구성: 본문에 따르면 250 NP holdout (enveda-np-examples 로 추정) + 569 "obscure" 구조 = 819.
  - class-1 시뮬레이션: 쿼리의 소스 라이브러리를 통째로 뺀 라이브러리로 검색 (안 빼면 1.000).
  - class-2 시뮬레이션: 그 구조의 스펙트럼을 모든 라이브러리에서 제거 (InChIKey14 단위). 풀에는 남김.
  - FP 피처는 전 행에 채워져 있음 → 만들 때 쓴 FPNet 버전은 알 수 없음 (현재 m1+s2 와 다를 수 있음).
- **결론: CV 안에서는 사용 불가.** 식별자가 없어 우리 holdout 과 겹치는지 확인할 수 없고, enveda-np-examples 를 거의 확실히 포함한다. CV 는 fold 마다 ranker 를 다시 학습한다 (`cv.py`). test 제출용으로는 원본 재현(exp000)을 위해 그대로 쓸 수 있다.

## 5. 런타임 (노트북 기준, Kaggle CPU 4코어 + T4)
- train 구조 지문 재계산 약 5분 (COCONUT 쪽은 첨부).
- 라이브러리 로드 (2.54M 스펙트럼, 4억 피크) 수 분, 메모리 여러 GB.
- 분자당: 라이브러리 검색 + analog (±200 Da, 약 수만~20만 대표 스펙트럼 × 분자 스펙트럼 수), MetFrag-lite 약 14 ms × 후보 수, FPNet 2개. 전체 400 분자 기준 수십 분 수준. 9시간 제한에 여유가 큼.
- 포팅본 로컬 측정치 (Apple Silicon 10코어, CPU 만, RSS 약 3.4GB):
  - `casmi build` 1회: metric 키 275,810개 약 4.7분, 라이브러리 cleaning + 캐시 약 20초, train 구조 지문 76초.
  - test 400 분자 예측 5분 11초 (분자당 0.78초), rank_train.npz ranker 학습 67초 포함 전체 6분 24초.
  - CV 500 분자 피처 (A/B/C 세 split) 10분 40초 (분자당 1.28초: analog 43%, 후보+MetFrag 53%), ranker CV (5 fold × 8 GBM × 2 변형) 16분.

## 5.1 CV 결과 요약 (exp000, `outputs/cv/exp000/summary.txt`)

| ranker | A (np, n=250) | B (np) | B (e180) | 가중 np (16/45/39) |
|---|---|---|---|---|
| fold 재학습, 4채널 | 0.796 | **0.619** | 0.972 | **0.406** |
| fold 재학습, FPNet 제외 | 0.799 | 0.590 | 0.896 | 0.394 |
| rank_train.npz (누수) | 0.834 | 0.674 | 0.943 | 0.437 |

- B(np) 0.619 는 노트북이 적은 4채널 Class-2 MRR 0.612~0.63 과 맞는다 → 포팅이 동작을 재현했다고 본다.
- 단일 채널 (split B, ranker 없음): analog 0.546 (np) / FPNet f·z 0.483 (np) vs **0.898 (e180)** → FPNet 이 enveda-180 구조로 학습됐다는 강한 신호. e180 split 의 FP 포함 숫자는 믿으면 안 된다.
- e180 은 FPNet 없이도 B 0.90 → 합성 라이브러리라 가까운 analog 가 많은 쉬운 분포. 설정 비교는 np 기준으로 한다.
- split C 는 retrieval 만으로는 정의상 0 (정답 metric 키를 풀에서 제거). 생성 후보를 넣는 실험부터 의미가 생긴다.
- rank_train.npz 를 쓰면 np B 가 +0.055 → np-examples 가 ranker 학습에 들어 있다는 뜻.

## 6. haideptry / v4f 가 바꾼 것

### haideptry (0.339, 최다 투표)
prvsiyan 의 더 이른 버전 fork 에 가깝다. 차이:
- `PPM_WIN=8.5` (prvsiyan 은 recall 손실 때문에 10 으로 되돌림), `N_ANALOG=100`, `SIM_POWER=4` (피처 함수의 P_SIM 도 4).
- 라이브러리 인덱스를 precursor+adduct 로 계산 (분자식 질량 수정 없음), analog 대표 스펙트럼을 **richest-only** 로 선택 (장비 매칭 없음).
- `_logits_from` 의 메타데이터 정렬 버그(빈 스펙트럼 제거 후 rows[:B]) 가 그대로 남아 있음.
- 25개 미만이면 `CCO` 로 채움 (유효 SMILES 라 점수엔 무해).
- 나머지(31 피처, rank_train.npz, 같은 FPNet 데이터셋, HistGBM 8개)는 동일.

### ahmedberatozer v4f (0.362, 공개 최고)
완전히 다른 자체 엔진(`casmi` 패키지, 비공개 데이터셋 `casmi26-v4b-models`, `casmi26-v3-models`, `casmi26-iceberg`)을 불러오는 inference 노트북이다. 공개된 코드에서 보이는 구조:
1. **자체 풀** (`pool_meta.parquet`, `pool_frag_*` 로 조각 질량까지 사전 계산) + `score_key` (tautomer 정규화 키) 로 dedupe.
2. **Engine(generate=True)**: 후보 생성(유도체 생성)이 포함된 v1 엔진 + FPNet A (자체 학습, adduct·CE·병합 수 조건). 스펙트럼을 polarity 별로 병합 (`z = 0.5·(per-spectrum 평균 + polarity 별 merged 평균)`).
3. **fe_v4 피처 군**: derivation evidence / class-3 prior, fragmentation 2.0, analog-structure relation, DreaMS-FP view. LightGBM ranker (자체 시뮬레이션 `sim1x` 로 재학습).
4. **ICEBERG 재채점**: ranker 상위 60 후보를 ICEBERG (MassSpecGym 가중치) 로 시뮬레이션해 이성질체 재정렬 (λ=0.5, GPU 2,700초 예산, 라이브러리 매치 없는 분자 우선).
5. **gated PubChem 채널**: 별도 프로세스에서 PubChem ±10ppm 창을 f·z 로 2단계 (ECFP4 부분 f·z 상위 1000 → 전체 f·z) 정렬, 풀에 없는 구조만 상위 25. 라이브러리 max sim ≥ 0.9 면 건드리지 않고, `(PubChem 최고 f·z − 풀 최고 f·z) > 600` 이면 공격적 슬롯 [2,4,6,8,10], 아니면 [4,8,12,16,20] 에 끼워 넣음.
- 요점: prvsiyan 노트북이 "PubChem 확장은 해롭다"고 결론낸 것을 **슬롯 게이팅**으로 우회했고, 이성질체 구분에 ICEBERG 를 추가했다. 두 가지 모두 CV 로 검증해야 할 다음 후보.

## 7. 포팅 (`src/casmi/`) 과 원본의 차이

| 모듈 | 내용 | 원본 대비 |
|---|---|---|
| `config.py` | CFG | 동일 값 |
| `chem.py` | adduct/질량/지문 | 동일 |
| `spectra.py` | numba 커널 | 라이브러리 스펙트럼 cleaning 을 **캐시 빌드 시 1회** 수행 (결과 동일, 매 비교마다 재계산 안 함) |
| `library.py` | 라이브러리 캐시 + 대표 스펙트럼 | lexsort 로 벡터화 (선택 규칙 동일). 구조 id(sid)·metric 키 추가 |
| `candidates.py` | 풀 | train 구조 지문 캐시. COCONUT 은 첨부본 (436,389) |
| `engine.py` | 채널 + 후보 + 피처 | `Mask` 로 CV 시 증거 숨김 |
| `frag.py` | MetFrag-lite | 영구 worker pool + SMILES 캐시 |
| `fpnet.py` | FPNet + 2-view logits | 메타데이터 정렬 버그 수정본 그대로 |
| `ranker.py` | HistGBM 8개 | 동일 |
| `submit.py` | 출력 | **metric 키로 dedupe + RDKit 파싱/정규화 실패 SMILES 제거** (원본은 top-25 그대로). 25개 이상 확보될 때까지 순위를 계속 내려감 |
| `metric.py` | 공식 metric 재구현 | RDKit 2026.03.3 고정 (`pyproject.toml`) |
| `cv.py` | split A/B/C CV | 신규 |

이 dedupe 때문에 원본보다 아주 약간 좋아질 수 있다 (같은 metric 키 중복이 순위를 차지하던 경우).
