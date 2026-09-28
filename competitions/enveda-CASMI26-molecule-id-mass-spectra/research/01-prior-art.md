# 01. 선행 연구 / SOTA 조사 (2026-09-28 기준)

표기 규칙
- **[사실]**: 출처 URL에서 직접 확인한 내용
- **[기억]**: 논문/레포를 과거에 읽은 기억 기반, 이번에 URL로 재확인하지 못함 → 사용 전 재확인 필요
- **[추론]**: 조사 내용을 바탕으로 한 내 판단

---

## 0. 이 대회 핵심 사실 (Overview/Data 탭 직접 확인, 2026-09-28)

출처: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview , /data , /leaderboard , /code

- **[사실] 과제**: 분자 1개당 MS/MS 스펙트럼 1–16개(중앙값 3)를 보고 SMILES 최대 25개 순위 제출. 스펙트럼 단위가 아니라 **분자 단위 집계**.
- **[사실] 평가 지표**: MRR@25. RDKit tautomer canonicalization(2026.03.3 고정) 후 **InChIKey14(첫 블록)** 일치 판정 → 입체화학/토토머 무시. 오답 패널티 없음(순위만 소모).
- **[사실] 테스트**: 약 400분자 / 약 1,500 스펙트럼, **전부 Bruker timsTOF**, 단일동위원소 질량 157–1,159 Da(중앙값 348). 어덕트 10종: [M+H]+, [M+NH4]+, [M-H2O+H]+, [M-2H2O+H]+, [M+Na]+, [M+K]+, [M-H]-, [M-H2O-H]-, [M+CH2O2-H]-, [M+Cl]-. precursor_mz, adduct, collision_energy_ev 제공(분자식은 미제공).
- **[사실] 난이도 클래스 3종** (비율 비공개):
  1. 공개 스펙트럼 라이브러리에 있음 → 학습 스펙트럼과 유사도로 찾을 수 있음
  2. 공개 스펙트럼 없음, 그러나 **PubChem 또는 COCONUT**에 구조 존재 → DB 검색으로 도달 가능
  3. PubChem에 없는 신규 구조 → de novo 필요
- **[사실] 학습 데이터**: ~2.5M 스펙트럼 / ~275k 구조. ingest_lib: enveda-180(1.15M, timsTOF, 합성 drug-like), pluskal_ms2(MSnLib, Orbitrap), riken(식물 2차대사체), gnps, massbank, mona, spectraverse, msdial, drug_plus, **enveda-np-examples(1,151 스펙트럼/250 천연물, 테스트와 같은 장비·파이프라인 — 도메인 적응용 핵심)**, masaryk.
- **[사실] 코드 대회**: CPU/GPU 노트북 ≤ 9시간, **인터넷 차단**, "자유롭게 공개된 외부 데이터·사전학습 모델 허용", 제출 파일 submission.csv.
- **[사실] 리더보드(2026-09-28)**: Public LB = 테스트의 약 33%(≈130분자). 1위 0.425, 10위 0.395, 50위 ≈0.364. 참가팀 약 1,694. 공개 노트북은 0.34–0.36대(예: "Fast Spectral Cosine Baseline" 0.339, "casmi26-v4f-inference" 0.362). 호스트의 "CASMI denovo tutorial notebook" 핀 고정.
- **[사실] 공개 파이프라인 구조 예시**(https://github.com/shiferaxa/casmi26): 후보 풀 = COCONUT ~712k + 학습 구조, 중성 질량 ±8.5 ppm 필터 → 4채널 증거(① 학습 스펙트럼과 spectral entropy 유사도, ② 질량 이동된 유사 스펙트럼 구조와의 Tanimoto "analog", ③ 1–2개 결합 절단으로 설명되는 강도 비율, ④ spectrum→fingerprint transformer 로짓과 후보 FP 내적) → GBDT 랭커. 검증 시 "타깃 자신의 스펙트럼 숨기기 + 후보 풀에서 타깃 제거" 마스크 split만 LB 순위를 잘 예측했다고 보고.
- **[추론]** Public LB가 ≈130분자라 1분자가 rank1로 바뀌면 ±0.0077. 상위권 간 차이(0.01–0.03)는 분자 몇 개 수준 → **Private shake-up 위험 큼**. 로컬 CV(특히 enveda-np-examples + 클래스별 마스킹)를 신뢰 기준으로 삼아야 함.
- **[추론]** 은메달 ≈ 상위 5%(≈85위). 현재 기준 ~0.36–0.37, 종료 시 0.40+로 오를 가능성.

---

## 1. 과거 CASMI 대회

| 대회 | 내용 | 우승 접근 / 결과 | 출처 |
|---|---|---|---|
| CASMI 2016 | 208 챌린지. Cat.2(인실리코 단편화만), Cat.3(메타데이터 허용) | [사실] Cat.2 금 CSI:IOKR(41% 1위), 은 CSI:FingerID(39%), 동 MS-FINDER. Cat.3 금 MS-FINDER+메타데이터(76%), 은 CFM-ID+DB 점수(75%). 단편화만: Top1 >30%, Top10 ~50%. 메타데이터 포함: Top1 70%, Top10 87% | https://pmc.ncbi.nlm.nih.gov/articles/PMC5368104/ |
| CASMI 2017 | 천연물 카테고리 포함 | [사실] CSI:FingerID가 Cat.1(천연물), Cat.2, Cat.4 우승. Cat.4에서 198개 중 66개(33.3%) 정답, 차순위의 6배 이상 | https://bio.informatik.uni-jena.de/2017/11/casmi-2017-results-are-out/ |
| CASMI 2022 | 16개 그룹. 어덕트/분자식/화합물 클래스/2D 구조 4개 영역 | [사실] 어덕트 94%(Nikolic), 분자식 94%·클래스 69%(Dührkop, SIRIUS), 2D 구조 최고 38%(Nikolic, KUCA) | https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results |

**교훈**
- [사실] 방법끼리 상호보완적 → **consensus/앙상블이 개별 방법보다 강함**(CASMI 2016 논문).
- [사실] **메타데이터(인용수·참조수 같은 "유명도" prior, RT)** 가 Top1을 30%→70%로 끌어올림. [추론] 이 대회에서 RT는 없지만 "후보의 prior"(PubChem 특허/문헌 수, COCONUT 포함 여부, 천연물 유사도)는 강력한 랭킹 피처가 될 가능성 큼. 단, 클래스 2는 "공개 스펙트럼 없는" 구조라 유명도가 낮을 수 있음 → prior 과신 주의.
- [사실] 분자식을 먼저 맞히는 것(SIRIUS)이 강력. [추론] 분자식 필터 적중률 ≥90%면 후보 수가 ±ppm 필터보다 크게 줄어 검색 정확도 상승.
- [사실] 진짜 unknown(후보 리스트에 없음)은 여전히 거의 못 푼다.

---

## 2. ML 기반 MS/MS → 구조 (2023–2026)

### 2.1 MassSpecGym 벤치마크
출처: https://arxiv.org/abs/2410.23326 , https://github.com/pluskal-lab/MassSpecGym , 리더보드 https://massspecgym.onrender.com
- [사실] 231k 스펙트럼 / 29k 분자(MoNA, MassBank, GNPS, 자체 측정 ~10k 분자). 주로 [M+H]+, [M+Na]+, 98% Orbitrap/QTOF (**timsTOF 아님**). MCES 거리≥10 기반 split. 코드 MIT.
- [사실] 과제: (a) de novo 생성(top-k 정확도, MCES, Tanimoto), (b) 검색(후보 ≤256, 질량 또는 분자식 매칭; hit@k, MCES@1), (c) 스펙트럼 시뮬레이션(cosine, JS, 시뮬레이션 기반 검색 hit@k). 각 과제에 "분자식 제공" 보너스 변형.
- [사실] 베이스라인: 검색 MIST hit@1 14.64%(질량 후보), 9.57%(분자식 후보, 더 어려움). de novo SMILES Transformer 정확도 0%. 시뮬레이션 FraGNNet cosine 0.52, 검색 hit@1 31.93%(분자식 후보).
- [기억] 검색 hit@1: random 0.37%, DeepSets 1.47%, FFN fingerprint 2.54%, DeepSets+Fourier 5.24%; MIST hit@20 ~59%.
- **[사실] 평가 함정 논문 "MassSpecGym in the Wild"(2026-06)** https://arxiv.org/html/2606.19624 :
  - MIST 배치 추론 **padding mask 버그**로 다운스트림 성능이 ~20%p 부풀려짐. MIST+MolForge de novo top-1: 버그 포함 31.75% → 수정 후 **10.73%**.
  - non-canonical SMILES 형식만으로 hit@1 82%, PubChem 순위 편향만으로 49% 가능(스펙트럼 무시) → **후보 SMILES는 반드시 동일 RDKit 정규화**.
  - 분자식 누설로 hit@1 +9%p. 테스트와 Tanimoto≥0.7 분자로 디코더 사전학습 시 성능 부풀림.
  - [추론] 논문 숫자는 절반쯤 할인해서 받아들일 것. 특히 MIST 파생 결과(MolForge 28%, MS-GPT 23.9% 등)는 이 버그 영향 여부 확인 필요.

### 2.2 모델별 정리

| 모델 | 무엇을 하나 | 가중치/라이선스 | 컴퓨트 | 보고 수치 |
|---|---|---|---|---|
| **SIRIUS / CSI:FingerID** | 분자식(단편화 트리, isotope) → FP 예측 → DB 검색. CANOPUS(클래스) | SIRIUS 코드 공개(AGPL) [기억], **CSI:FingerID는 웹서비스 + 계정/라이선스 필요** → [추론] 인터넷 차단 Kaggle에서 FP 예측 불가, 분자식 계산만 가능성 | CPU | CASMI 2016–2022 다수 우승(§1) https://github.com/sirius-ms/sirius |
| **MIST** (Goldman 2023) | 분자식 기반 subformula 토큰 transformer → fingerprint 예측 → 검색 | MIT [기억]; DiffMS가 MassSpecGym용 MIST 가중치 Zenodo 공개 https://zenodo.org/records/15122968 | 단일 GPU, 수 시간 | MSG hit@1 14.64% |
| **MIST-CF** | MIST 구조로 분자식/어덕트 후보 순위화 | MIT [기억] | 경량 | [기억] SIRIUS급 분자식 정확도 |
| **BUDDY** (2023) | bottom-up 분자식 annotation (MLR 재순위화) | 공개(msbuddy, pip) [기억] | CPU | [기억] SIRIUS와 비슷하거나 약간 우세 |
| **JESTR** (2024) | 스펙트럼–분자 joint embedding(대조학습) 검색 | 코드 공개 | 단일 GPU | [사실] MSG hit@1 15.62%(질량), 11.85%(분자식) https://arxiv.org/abs/2411.14464 |
| **CSU-MS2** (Anal.Chem 2025) | 대조학습 cross-modal 검색 | 논문 공개 | 단일 GPU | [사실] 1M 라이브러리에서 Recall@1 75.45%(자체 셋; CFM-ID 68%, SIRIUS 65%) — 벤치마크 조건 달라 비교 주의 https://pubs.acs.org/doi/10.1021/acs.analchem.5c01594 |
| **DreaMS** (Nat Biotech 2025) | 116M 파라미터 BERT류, GeMS 24M+ 비라벨 스펙트럼으로 masked peak 예측 사전학습, 1024-d 임베딩 | **MIT, 가중치 Zenodo(10997887)** https://github.com/pluskal-lab/DreaMS | 추론 T4로 충분, 파인튜닝 단일 GPU | 스펙트럼 유사도/분자 속성 개선. [사실] MSAlign(2026)이 DreaMS+MolDeBERTa 정렬로 검색 SOTA 주장 https://arxiv.org/abs/2605.19752 |
| **PRISM** (Enveda) | 1.2B 스펙트럼 BERT류 사전학습(Enveda 사설 600M 포함) | [사실] 가중치 공개 언급 없음 | — | 스펙트럼 매칭 23% 상대 개선 https://enveda.com/prism-a-foundation-model-for-lifes-chemistry/ |
| **ICEBERG / MARASON / SCARF** (Coley) | 구조→스펙트럼 시뮬레이션(단편 그래프 생성 + 강도 예측). MARASON은 retrieval-augmented + neural graph matching | **MIT**, MassSpecGym 학습 가중치 Dropbox 제공(NIST 학습본은 라이선스 필요) https://github.com/coleygroup/ms-pred | 후보당 GPU 수십 ms급 [기억] | [사실] MARASON MSG 검색 hit@1 34.03%(FraGNNet 31.93% 대비), PubChem 검색 27.8%(ICEBERG 18.7%) https://arxiv.org/abs/2502.17874 |
| **FraGNNet** | 확률적 단편 분포 → 스펙트럼 | 코드 공개 https://github.com/LZhang98/fragnnet (라이선스 미확인) | GPU | MSG 시뮬 cosine 0.52, hit@1 31.93% |
| **3DMolMS / NEIMS / MassFormer / CFM-ID** | 시뮬레이션 베이스라인 | ms-pred에 구현 포함. CFM-ID는 느림(분자당 초~분) [기억] | — | MARASON/ICEBERG보다 열세 |
| **MSNovelist** (2022) | CSI:FingerID FP → SMILES LSTM 디코더 | 공개 [기억] | GPU | [기억] 저조(MSG에서 한 자릿수%) |
| **Spec2Mol** (2021) | 스펙트럼→SMILES 인코더-디코더 | 공개 [기억] | GPU | [기억] 정확도 매우 낮음 |
| **MADGEN** (ICLR 2025) | 스캐폴드 검색(대조학습) → 스캐폴드 조건 생성 | 코드 공개 | GPU | [사실] MSG top-1 1.31%(예측 스캐폴드), 10.5%(oracle 스캐폴드) https://arxiv.org/abs/2501.01950 |
| **DiffMS** (ICML 2025) | MIST 인코더 + 분자식 조건 이산 그래프 diffusion, 대규모 FP→분자 사전학습 | 코드·가중치 공개(Zenodo) | 학습 수 일/A100급 | [사실] MSG top-1 2.30%, top-10 4.25%, Tanimoto 0.28 |
| **MIST + MolForge** (2025) | FP 이진화(0.5) → FP→분자 디코더(2M 분자 학습) | 공개 | 디코더 학습 A40 3일 | [사실] 보고 28.27% top-1 → **버그 수정 후 10.73%** |
| **Test-time tuned LM** (2025) | 분자식+스펙트럼 → SMILES transformer, 시뮬레이션 사전학습 + test-time tuning | — | GPU | [사실] MSG top-1 3.16%, NPLIB1 12.88% https://arxiv.org/abs/2510.23746 |
| **MS-GPT** (2026-07) | FP posterior → 분자 LM 조건 질의(LoRA), 생성 빈도 합의로 순위 | 코드·체크포인트 공개 | GPU | [사실] MSG top-1 23.9%, NPLIB1 29.8% (MIST 버그 영향 여부 미확인) https://arxiv.org/abs/2607.23607 |
| **MBGen / GLMR** 등 | diffusion 개량, 생성 기반 검색 | — | — | [사실] MBGen MSG top-1 7.58%; GLMR recall@1 54%(자체 설정) https://arxiv.org/abs/2602.01643 , https://arxiv.org/abs/2511.06259 |
| **MS2Mol** (Enveda 2023) | 스펙트럼→SMILES transformer (호스트 자체 모델) | — | — | https://www.researchgate.net/publication/371764523 |

**[추론] 요약**
- 검색(retrieval) 계열이 de novo보다 확실히 강함. 현실적 de novo top-1은 MassSpecGym에서 **≈3–11%**(버그 제거 기준). 클래스 3에서 기대치는 매우 낮고, 25개 슬롯의 꼬리에 넣어 부분 점수를 얻는 정도.
- 대부분 논문 모델은 Orbitrap/QTOF·[M+H]+ 위주로 학습 → **timsTOF 10종 어덕트에 도메인 갭**. enveda-180(timsTOF 1.15M) + enveda-np-examples로 파인튜닝/보정하는 것이 차별점.
- 논문 수치(Hit@1)와 이 대회 MRR@25는 비례하지만, 후보 풀 크기가 다르다(MSG ≤256 vs 여기선 PubChem/COCONUT 질량 윈도우 수천~수만).

---

## 3. 비슷한 Kaggle 대회 — 옮겨 쓸 교훈만

- **BMS Molecular Translation (2021, 이미지→InChI)**: [기억] 상위권은 beam search 다수 후보 → RDKit 유효성/정규화 검사 → 재순위화(normalize InChI) + 앙상블. [추론] 여기서도 "생성 → 정규화(InChIKey14 dedupe) → 재순위화" 루프가 핵심. 25개 슬롯에 **InChIKey14 중복 제거는 필수**(같은 키 두 번은 슬롯 낭비).
- **Leash BELKA (NeurIPS 2024, 소분자 결합 예측)**: [기억] 테스트에 학습에 없는 빌딩블록 → 공개/비공개 LB 괴리가 컸고, **새 화학공간 일반화 검증 split**을 만든 팀이 이김. [추론] 여기서도 클래스 2/3는 학습 구조와 겹치지 않음 → "타깃 구조를 학습·후보에서 제거한 마스킹 CV" 필수(shiferaxa 레포 보고와 일치).
- **Stanford RNA 3D Folding (2025)**: [기억] 템플릿(검색) 기반이 순수 de novo보다 우세. [추론] 동일 원리: 검색 + 유사체(analog) 조합이 기본.
- **NeurIPS Open Polymer Prediction (2025)**: [기억] 외부 데이터 정제·중복 제거·라벨 노이즈 관리가 순위를 좌우. [추론] 이 대회 train도 라이브러리별 전처리·라벨 품질이 달라 `precursor_error_ppm` 기반 필터링이 유효할 것.
- 공통: 코드 대회 9시간 제한 → 무거운 후보 임베딩(PubChem 수천만 FP 등)은 **사전 계산해 Kaggle Dataset으로 업로드**하고 추론 시엔 조회만.

---

## 4. 외부 데이터 (규칙: "자유롭게 공개된 외부 데이터·사전학습 모델 허용")

| 데이터 | 내용 | 라이선스/사용 가능성 | 비고 |
|---|---|---|---|
| MassSpecGym (HF) | 231k 스펙트럼, 후보 셋(1M 생물/환경, 4M 다양, PubChem 118M 기반) | 코드 MIT, 데이터 공개 [사실] | 구성 라이브러리 대부분이 이미 train에 포함 → 추가 이득은 주로 **in-house 10k 분자** + 후보 셋 [추론] |
| GNPS / MoNA / MassBank / RIKEN / MS-DIAL | 공개 라이브러리 | 대체로 CC0/CC BY [기억] | 이미 train에 들어있음. 최신 GNPS 버전은 추가분 있을 수 있음 |
| MSnLib (Pluskal) | pluskal_ms2 원본(MSn 포함) | 공개 [기억] | train에 포함 |
| GeMS (DreaMS 사전학습용) | 수천만 비라벨 스펙트럼 | 공개(MIT/Zenodo) [사실] | 자기지도 사전학습용 |
| **NIST20/23 MS/MS** | 최대 상업 라이브러리 | **유료·상업 라이선스** [사실] → "freely available" 아님 → **사용 불가**. NIST 학습 가중치도 불가 | |
| **PubChem** | ~118M 구조 | 퍼블릭 도메인 [기억] | 클래스 2 도달 필수. 질량 윈도우당 후보 수천~수만 → 사전 필터(원소 조성, 천연물 유사도) 필요 |
| **COCONUT 2.0** | 약 70만 천연물 구조 | CC0 [기억]; 공개 파이프라인은 ~712k 사용 [사실] | Enveda 테스트(천연물·유사체)와 가장 직접 관련. 1순위 후보 풀 |
| PubChemLite / LOTUS / NPAtlas / HMDB | 소형 생물 관련 DB | 대부분 공개(HMDB는 비상업 조건 확인 필요) [기억] | 후보 prior 피처로 유용 |

- [사실] Kaggle 스펙트럼 데이터는 이미 GNPS/MoNA/MassBank 등 공개 라이브러리 "편의 통합본"이다.
- [추론] 가장 큰 외부 레버리지는 스펙트럼이 아니라 **후보 구조 DB(COCONUT + PubChem 부분집합)** 와 **사전학습 가중치(DreaMS, ms-pred MassSpecGym 가중치, MIST/DiffMS Zenodo)** 다.

---

## 이 대회 접근법 후보 (기대 점수 × Kaggle 무료 GPU 실행 가능성 순)

**1순위: 공개 멀티채널 GBDT 랭커 강화 (현재 0.34–0.36 → 목표 0.40+)**
- 기반: COCONUT + PubChem 부분집합 + train 구조를 질량(ppm)·어덕트 필터로 후보 생성 → 채널: 라이브러리 유사도(entropy/modified cosine, 다중 CE 스펙트럼 집계), analog Tanimoto, 단편 설명률, spectrum→FP 모델 점수, 후보 prior(COCONUT 포함, 문헌 수, NP-likeness) → LightGBM LambdaRank.
- 개선 레버: (a) **timsTOF 도메인 적응**: FP 모델을 enveda-180로 사전학습 → enveda-np-examples/천연물 라이브러리로 파인튜닝, (b) 분자식/어덕트 추정(BUDDY·MIST-CF식)으로 후보 축소, (c) 클래스 인지 CV(자기 스펙트럼 숨김 + 후보 제거) 기반 랭커 학습, (d) InChIKey14 dedupe.
- 실행성: 높음. FP 모델은 Kaggle T4/P100에서 수 시간 학습, 추론은 CPU 중심.

**2순위: 스펙트럼 시뮬레이션(MARASON/ICEBERG) 재순위화 채널 추가**
- 1순위 파이프라인의 상위 100–300 후보에 대해 시뮬레이션 스펙트럼 vs 관측 스펙트럼 cosine을 피처로 추가. 클래스 2(공개 스펙트럼 없음)에서 라이브러리 채널이 무력할 때 가장 효과적.
- 근거: MSG 검색 hit@1 MARASON 34% vs MIST 15%(단, 분자식 후보 조건). ms-pred MIT + MassSpecGym 가중치 공개.
- 리스크: timsTOF·비[M+H]+ 어덕트 갭(가능하면 enveda-180로 추가 학습), 400분자×300후보=12만 시뮬레이션 → GPU 9시간 내 가능하지만 환경 설정(오프라인 wheel) 부담. 실행성: 중.

**3순위: DreaMS 임베딩 기반 대조학습 검색(JESTR/MSAlign식) 채널**
- DreaMS(MIT, Zenodo) 스펙트럼 인코더 + 분자 인코더(FP/GNN)를 train으로 대조학습 → 후보 점수 채널. 어덕트·CE를 조건으로 넣고 timsTOF 데이터로 정렬.
- 실행성: 중(116M 모델 파인튜닝은 Kaggle GPU 주당 30시간 한도 내 가능하나 빠듯). 기대 이득: FP 채널과 부분 중복, 앙상블 효과 중간.

**4순위: 클래스 3용 de novo 꼬리 채우기 (저비용 버전만)**
- 호스트 "CASMI denovo tutorial notebook"을 출발점으로, 또는 상위 검색 결과의 analog 변형(작용기 추가/제거, 메틸화·글리코실화 등 천연물 변형 규칙)으로 PubChem 밖 후보 생성 → 동일 랭커로 점수 → 25개 슬롯 하위(예: 16–25위)에 배치.
- 근거: 최신 de novo도 top-1 3–11%(버그 수정 기준) → MRR 기여는 작음. 풀 모델(DiffMS/MS-GPT) 학습은 무료 GPU로 비현실적. 규칙 기반 analog 확장이 가성비 최고.
- 실행성: 규칙 기반 높음 / 딥 생성 모델 낮음.

**[추론] 운영 원칙**: Public LB(≈130분자)는 노이즈가 크므로 로컬 클래스 마스킹 CV로 결정하고, 최종 2개 제출은 "CV 최고"와 "LB·CV 균형"으로 분산.
