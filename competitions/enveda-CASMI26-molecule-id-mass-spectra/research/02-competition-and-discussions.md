# 02. 대회 규정 · 데이터 · 디스커션 · 코드 · 리더보드 조사

조사일: 2026-09-28 (대회 시작 2026-09-14, 14일차) · 조사 방식: Kaggle 페이지(비로그인 렌더링) + `kaggle` CLI (읽기 전용)

기본 URL: `C = https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra`

표기: **[호스트]** = 개요/데이터/규칙 페이지 또는 호스트·Kaggle 스태프 답변으로 확인된 사실, **[커뮤니티]** = 참가자 주장/추정(검증 안 됨).

---

## 1. 대회 개요 · 평가 · 규칙

### 과제 [호스트]
- LC-MS/MS 스펙트럼에서 **2D 구조(SMILES)를 식별**. 분자(`molecule_id`)당 최대 25개 후보 SMILES를 순위대로 제출. 스펙트럼 단위가 아니라 **분자 단위 예측** — 한 분자의 여러 스펙트럼(충돌 에너지·adduct 다름)을 합쳐 하나의 순위 목록을 만들어야 함. ([C/overview](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview), [C/data](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data))
- 성격: 후보 검색 + 랭킹과 de novo 생성이 모두 가능. 후보 DB를 쓰든 생성하든 형식은 자유.

### 평가지표 [호스트]
- **MRR@25** = 분자별 1/(처음 맞춘 순위)의 평균, 25위 안에서 못 맞추면 0.
- 정답 판정: 제출 SMILES와 정답 모두 **RDKit 2026.03.3 tautomer canonicalization → InChIKey 첫 블록(14자)** 으로 비교. 입체화학·토토머는 틀려도 됨.
- 공식 metric 노트북: https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank (스태프가 개요에 링크 추가, [741851](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741851))
- 경계 사례 [호스트, [742274](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742274)]: 빈 항목(`A;;B`)은 순위 매기기 전에 제거됨(B=2위), 끝의 `;`는 무시.
- 형식 오류는 거부: 컬럼 누락, 빈 파일, null, `molecule_id` 중복, 25개 초과.
- [커뮤니티, [742088](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742088)] **파싱 불가능한 filler SMILES로 25개를 채우면 제출 전체가 0.000**(두 번 측정). 공식 metric v13 로컬 동작과 다르다는 문의에 호스트는 "Kaggle 측에 확인하겠다"고만 답함([742265](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742265)). → **RDKit으로 파싱되는 SMILES만 제출**하는 것이 안전.

### 제출 형식 [호스트]
```
molecule_id,smiles
m_0014ef,CC1=CC(=O)C=CC1=O;OC(=O)c1ccccc1O;...
```
파일명 `submission.csv`.

### 코드 대회 요건 [호스트] ([C/overview#Code Requirements](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview))
- 노트북 제출만 가능. **CPU ≤ 9시간, GPU ≤ 9시간**, **인터넷 차단**.
- 무료·공개 외부 데이터 및 사전학습 모델 허용.
- 보이는 `test.parquet`는 **train에서 뽑은 예시**이며, 재실행 시 숨겨진 테스트(비슷한 크기)로 교체됨.
- GPU 종류(P100 허용 여부)는 호스트가 "Kaggle 측에 확인 예정" — 미확정 ([742265](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742265)).
- [커뮤니티] 채점이 수 시간 걸린다는 보고 ([742789](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742789)). 인터랙티브로는 돌지만 숨겨진 테스트 재실행에서만 실패하는 사례 있음 → 스태프는 개별 지원 불가, 디버깅 문서만 안내 ([742463](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742463)).

### 제출/팀/일정 [호스트] ([C/rules](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules), 개요 Timeline)
| 항목 | 값 |
|---|---|
| 일일 제출 | **5회** |
| 최종 선택 | **2개** |
| 팀 최대 | 5명 (병합 시 합산 제출수 ≤ 일일한도×경과일) |
| 시작 | 2026-09-14 |
| Entry / Team Merger 마감 | **2026-12-07** 23:59 UTC |
| 최종 제출 마감 | **2026-12-14** 23:59 UTC |
| 상금 | 총 $50,000 (16k/12k/9k/7k/6k) |
| 메달 | 지급 ("Awards Points & Medals") |
| 수상자 라이선스 | 오픈소스 MIT |
| 데이터 라이선스 | CC BY-NC 4.0 (단 호스트: train은 오픈소스 라이브러리 모음이라 거기서 학습한 모델은 오픈소스 가능, [743236](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743236)) |

### Public / Private 분할 [호스트] ([C/leaderboard](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard))
- "This leaderboard is calculated with approximately **33%** of the test data. The final results will be based on the other **67%**."
- 테스트는 **~400 분자 / ~1,500 스펙트럼** → public ≈ **~130 분자**, private ≈ **~270 분자** (분할 단위가 분자인지 스펙트럼인지는 공지되지 않았으나 채점이 분자 단위이므로 분자 단위일 가능성이 높음 — 추정).
- 같은 분포인지 공지 없음. 클래스(1/2/3) 비율과 분자별 클래스는 **대회 중 비공개** ([C/data](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)).
- 산술: public에서 1분자를 1위로 맞추면 ≈ **+0.0076**, private에서는 ≈ +0.0037.

### 외부 데이터 · 사전학습 모델 [호스트]
| 자원 | 판단 | 출처 |
|---|---|---|
| 허용 조건 일반 | 공개·동등 접근·무료(또는 합리적 비용), **상업적 이용/재배포를 금지하는 라이선스는 수상 불가** | [741359](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741359), Rules 2.6 |
| **NIST, METLIN, 장비사(Thermo/Bruker/SciEx) 라이브러리** 및 거기서 학습한 가중치 | **불가** | [742193](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742193), [742991](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742991) |
| **MassSpecGym** 및 이를 학습한 permissive 모델(ICEBERG MSG 체크포인트, MIST 등) | **허용** | [742991](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742991) |
| train.parquet로 학습한 오픈 모델(타인의 공개 가중치 포함) | 허용 | [742265](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742265) |
| 공개 가중치 일반 | 학습 데이터 라이선스를 지키며 배포된 경우 허용 (일괄 승인은 불가) | [742193](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742193), [741844](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741844) |
| **PubChem** 구조(검색용, 분자식으로 필터링한 subset을 데이터셋으로 패키징) | 허용 | [741857](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741857) |
| **COCONUT 2.0** | 허용 (COCONUT 자체 라이선스로 충분) | [743234](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743234) |
| COCONUT 구조로 CFM-ID 생성 스펙트럼 / LGPL 런타임 의존 | 허용 | [741912](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741912) |
| **CFM-ID 4 기본 가중치**(METLIN 학습) | 허용 ("주요 저널 게재라 배포 권한 있다고 간주") | [743774](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743774) |
| Enveda-180 Zenodo 원본(CC BY 4.0) | 허용 | [743569](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743569) |
| enveda-np-examples | 오픈소스로 간주 | [742193](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742193) |
| 외부 클라우드(예: HF)에서 train 데이터로 학습 후 가중치만 반입 | 허용 | [741876](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741876) |
| 이온 이동도 / CCS | **테스트에 없음 (의도적 제외)** | [743368](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743368) |

데이터 페이지 추천 자원 [호스트]: MIST-CF, SIRIUS, matchms, GNPS propagated annotations(train에서 제외), MassIVE, **GeMS/DreaMS**(사전학습 임베딩), FragHub.

---

## 2. 데이터 [호스트] ([C/data](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data))

- 전체 3.04 GB. `train.parquet` ≈ **2.5M 스펙트럼 / ~275,810 고유 구조**, 1행 = 1스펙트럼. `test.parquet` 1행 = 1스펙트럼. `sample_submission.csv`.
- 대회 초기 train 갱신(water-loss adduct 추가) → **재다운로드 권장** ([741471](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741471)).

### 숨겨진 테스트
- ~1,500 스펙트럼, ~400 분자, 분자당 1–16 스펙트럼(중앙값 3), **전부 Bruker timsTOF**. 단일동위원소 질량 157–1,159 Da(중앙값 348).
- **Novelty 클래스**: 1 = 공개 스펙트럼 라이브러리에 있음 / 2 = 스펙트럼은 없지만 PubChem 또는 COCONUT에 구조 있음 / 3 = PubChem·COCONUT 어디에도 없음(호스트 확인, [742274](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742274)). 비율은 비공개.
- 최소 정제: 전구체 질량 불일치 제거, base peak < 1,000 counts 제거, precursor+2 Da 초과 피크 제거.
- adduct 10종: [M+H]+, [M+NH4]+, [M-H2O+H]+, [M-2H2O+H]+, [M+Na]+, [M+K]+, [M-H]-, [M-H2O-H]-, [M+CH2O2-H]-, [M+Cl]-. (페이지 컬럼 설명엔 "seven"으로 남아 있음 — 오기)
- **후보 리스트 제공 없음.** 후보 풀은 직접 구축.

### test.parquet (12 컬럼)
`molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities(base=1.0), base_peak_intensity, adduct, ionization_mode, instrument_type(항상 timsTOF), precursor_mz, collision_energy_ev(리스트; 여러 값이면 병합 스펙트럼), collision_energy_orig, collision_energy_orig_units(항상 eV)`

### train.parquet (18 컬럼)
test 공통 컬럼 **중 molecule_id·spectrum_id 제외** + `normalized_smiles`(라벨), `inchikey`, `inchikey14`, `molecular_formula`, `ingest_lib`, `adduct_orig`, `precursor_error_ppm`(정제 안 됨 → 라벨 품질 신호), `num_peaks`. ([742042](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742042)에서 참가자가 18컬럼 확인)
- `instrument_type`은 자유 텍스트/널, `base_peak_intensity`는 일부 라이브러리 null. **timsTOF 일치 라이브러리는 enveda-180과 enveda-np-examples뿐.**

| ingest_lib | 스펙트럼 | 구조 | 비고 |
|---|---|---|---|
| enveda-180 | 1,153,785 | 182,941 | 테스트와 동일 timsTOF, 단 합성 약물류(화학공간 다름) |
| pluskal_ms2 (MSnLib) | 527,581 | 46,821 | Orbitrap, NCE |
| riken | 347,171 | 15,892 | 식물 2차대사체 |
| gnps | 220,849 | 45,750 | 가장 이질적 |
| massbank | 101,727 | 9,180 | |
| mona | 92,416 | 11,681 | |
| spectraverse | 50,933 | 9,631 | |
| msdial | 40,765 | 9,127 | |
| drug_plus | 2,545 | 2,539 | CE 메타 없음 |
| **enveda-np-examples** | 1,151 | 250 | **테스트와 같은 장비·파이프라인**, 흔한 NP라 다른 라이브러리에도 존재 |
| masaryk | 652 | 416 | |

- 충돌 에너지는 `collision_energy_ev` 사용 권장 (NCE→eV는 근사).

### 데이터 관련 커뮤니티 발견 [커뮤니티]
- train의 `inchikey14`는 **tautomer canonical이 아님**: NP 라이브러리 66,490 구조 중 4.4%가 metric 키와 다름, 63,449개로 붕괴 → 후보 dedup·매칭은 **반드시 metric과 같은 키로** ([742042](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742042), [743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254): holdout의 ~8%에서 정답의 토토머 복제본 존재).
- Enveda adduct 라벨은 깨끗함(np-examples 250/250 정상), 라이브러리 쪽은 formate 오라벨 다수 → 테스트에 adduct 재가설 불필요, 학습 시 정제 ([742087](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742087)).
- 양이온 모드 m/z에 ~0.5 mDa 오프셋 가능성(Enveda-180 보정 코드의 Na 원자질량) — 좁은 fragment formula 매칭 시만 영향, 호스트 미답 ([743395](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743395)).
- timsTOF 전구체 오차 99%가 ~5 ppm 이내, 8.5→10 ppm 창은 후보 수 거의 불변 ([743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254)); ±10 ppm 창 중앙값 56 후보, +1.4 ppm 계통 오프셋 (prvsiyan 노트북).

---

## 3. 디스커션 요약

### 호스트 확인 사항 (요약)
위 1절 외부 데이터 표 참조. 추가: Class 3는 PubChem·COCONUT 모두에 없음, CCS 없음, 빈 항목 처리 규칙, GPU 종류와 filler-SMILES 채점 차이는 **미답**.

### 커뮤니티 주요 인사이트 [커뮤니티]
1. **클래스 비율 추정 (LB probing)** — 라이브러리 검색만 = public 0.151, Class 1 MRR ≈ 0.93 → Class 1 ≈ **16%**; Class 2 ≈ 45%, Class 3 ≈ 39% (뒤 두 개는 가정에 의존). 출처: prvsiyan 노트북, [741597](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741597). **public 130분자 기반이라 private에서 동일하다는 보장 없음.** haideptry는 Tier 1 10–15%, 2 45–55%, 3 30–40% 추정 ([741745](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741745)).
2. **retrieval-only 상한** ≈ 0.16 + 0.45 ≈ 0.61; 실측 파이프라인은 top-25 안에 정답이 있는 비율 ~40% → 완벽한 재정렬 상한 ≈ 0.40, 실제 0.324 (81% 추출) ([742088](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742088)). → **풀(pool) 품질이 재정렬보다 큰 레버**라는 주장. 반면 Adarsh는 "Class 2는 top-25 recall이 높고 ranker가 병목"이라 주장 ([743974](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743974)).
3. **3위(Udam Liyanage, public 0.414)의 교훈** ([743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254)):
   - DB 경로 오류의 ~98%는 **같은 분자식 이성질체** 혼동 → 분자식·질량창은 레버가 아님, **구성 이성질체 구분**(in-silico fragmentation 등 결합 연결을 "보는" 증거)이 레버.
   - PubChem ∪ COCONUT 외 ChEBI/LIPID MAPS/NPAtlas는 커버리지 추가 없음. curated-DB 멤버십 플래그는 holdout에선 좋고 LB에선 해로움.
   - "정답 구조를 DB에서 삭제 후 복원" 식 novel 시뮬레이션은 도달률을 ~3배 과대평가. 생성 후보 삽입은 아직 LB 이득 없음.
   - GBDT ranker 시드만 바꿔도 holdout ±0.007 → 0.005 미만 차이는 추격하지 말 것.
   - 두 프록시(공개 라이브러리 class-2식 holdout, 소규모 timsTOF NP 패널)가 서로 다른 변화를 예측 — 둘 다 추적.
4. **CV 전략**: enveda-np-examples 250개를 라이브러리명으로 빼는 holdout은 **구조가 다른 라이브러리에 모두 존재 → 누수**, CV 0.33–0.37 vs LB 0.15 ([741597](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741597)). holdout 구축 시 누수 2가지: (a) 풀 출처 피처(`src`, `np_likeness` NaN 패턴) → 0.94 가짜 점수, (b) Class 1 시뮬레이션에서 쿼리의 소스 라이브러리를 통째로 제외하지 않으면 1.000 (prvsiyan 노트북). Class 2 시뮬레이션은 InChIKey14 단위로 모든 라이브러리에서 제거.
5. **de novo 생성의 인센티브 갭**: MRR 형태상 1위 정답을 2위로 밀면 −0.5, 생성으로 5위에 새로 맞추면 +0.2 → confidence gating 없이 생성 후보를 섞으면 손해 ([741659](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741659)). "answer not in pool" 게이트 AUC 0.627로 약함 (prvsiyan).
6. **PubChem 전체 이성질체로 풀 확장 → 악화** (Class 2 MRR 0.52→0.35, LB 0.205) (prvsiyan); 꼬리 슬롯만 PubChem → +0.001~0.002 (노이즈 이내) ([742088](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742088)). 단 상위 v4f 노트북은 "gated PubChem channel"로 0.358에 기여했다고 기술.
7. **LB 노이즈**: 동일 노트북 재제출 0.335 vs 0.328, 0.292 vs 0.298 (HistGBM `random_state=None`). 계열 평균 0.329±0.004 (prvsiyan). 로컬 holdout 2종이 모두 LB와 반대 방향을 가리킨 사례(−0.019)도 있음.
8. 무료 GPU 30h/주 부족 불만 → 참가자: TPU 20h/주 추가 활용, Colab 유료 시 추가 quota (비공식) ([742082](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742082)).

리크: 현재까지 공개된 **데이터 리크는 없음**. 보이는 test.parquet는 train에서 뽑은 것이라 분석 무의미(호스트 명시 + [743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254)).

---

## 4. Code 탭 (public 노트북)

`https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/code?sortBy=scoreDescending` 기준, 점수는 public LB.

| 노트북 | Public | 투표 | 요지 |
|---|---|---|---|
| [ahmedberatozer/casmi26-v4f-inference](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4f-inference) | **0.362** | 15 | 자체 `casmi` 엔진 + FPNet 뱅크 + **ICEBERG 이성질체 재채점** + gated PubChem 채널 + 온건한 생성 + LightGBM ranker. 첨부 데이터셋이 메타데이터상 비공개(빈 문자열)라 그대로 fork 재현 불가할 수 있음 |
| [ahmedberatozer/casmi26-v3-inference](https://www.kaggle.com/code/ahmedberatozer/casmi26-v3-inference) | 0.358 | 27 | v4f의 전 단계 |
| [ahmedberatozer/casmi26-v2-baseline](https://www.kaggle.com/code/ahmedberatozer/casmi26-v2-baseline) | 0.354 | 32 | train ∪ COCONUT 검색 + 다채널 증거 + LightGBM |
| [evgendvorkin/enveda-casmi-2026](https://www.kaggle.com/code/evgendvorkin/enveda-casmi-2026) | 0.342 | 100 | prvsiyan pv + megayak Two-Ranker 블렌드(w=0.88), DreaMS 모델 첨부 |
| [raunakdey07/casmi-26-two-ranker-adduct-shifted-engine](https://www.kaggle.com/code/raunakdey07/casmi-26-two-ranker-adduct-shifted-engine) | 0.342 | 14 | 동일 계열 |
| [denpugovkin/casmi-2026-protected-bio-db-tail](https://www.kaggle.com/code/denpugovkin/casmi-2026-protected-bio-db-tail) | 0.341 | 42 | 꼬리 슬롯 bio-DB |
| [llccqq624/casmi26-next-direct-w088](https://www.kaggle.com/code/llccqq624/casmi26-next-direct-w088) | 0.341 | 34 | two-ranker 가중치 변형 |
| [haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline](https://www.kaggle.com/code/haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline) | 0.339 | **219** | 4채널(라이브러리 entropy / mass-shift analog / MetFrag-lite / FPNet f·z) + HistGBM, prvsiyan 데이터셋 사용 |
| [malikhammadfarooq/casmi26-quad-channel-evidence-ranker](https://www.kaggle.com/code/malikhammadfarooq/casmi26-quad-channel-evidence-ranker) | 0.339 | 54 | 동일 계열 |
| [beraterolelk/0-336-sota-enveda-casmi26-analog-ranker](https://www.kaggle.com/code/beraterolelk/0-336-sota-enveda-casmi26-analog-ranker) | 0.337 | 78 | 동일 계열 |
| [megayak/casmi26-two-rankers-one-engine](https://www.kaggle.com/code/megayak/casmi26-two-rankers-one-engine) | 0.337 | 30 | Ranker B(51 피처, FPNet 없음) |
| [prvsiyan/analog-propagation-casmi-2026-baseline](https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline) | 0.335 | 110 | **원조 파이프라인** + 가장 상세한 실험 로그(아래) |
| [inversion/casmi-denovo-tutorial-notebook](https://www.kaggle.com/code/inversion/casmi-denovo-tutorial-notebook) | – | 199 | Kaggle 스태프 de novo(encoder-decoder) 튜토리얼. 커뮤니티 노트북 제목상 "호스트 GPU 0.174" 언급(미확인) |
| 단순 라이브러리 검색류 | 0.14–0.16 | | Class 1만 커버 |

### 지배적 공개 파이프라인 (prvsiyan 계열) 구조
1. 후보 풀 = train 구조(275,810) ∪ COCONUT 2.0(462,028) → InChIKey14 dedup 711,705, ±10 ppm 중성질량 창(중앙값 56 후보).
2. 채널: (a) entropy 스펙트럼 유사도 라이브러리 검색(Class 1 MRR ~0.93), (b) **mass-shifted analog propagation** max sim(a)^p·Tanimoto (Class 2 0.16→0.52), (c) MetFrag-lite(1–2 결합 절단), (d) **FPNet**: 6층 Transformer(d=512) → 6,930비트 FP 로짓 z, 후보 점수 f·z; 같은 질량창 decoy 63개 softmax CE로 학습, T4 ~2시간. per-spectrum 모델 + merged 모델 두 "뷰" 조합이 +0.019로 LB 최대 효과.
3. 31 피처 → HistGBM(시드·class prior 평균), Class1/Class2 시뮬레이션 혼합 학습, W1(클래스 가중) ≈ 0.42가 최대 레버.
4. 실패: 후보 cap(Class 2 recall −24%), 전체 PubChem 확장, NP-likeness prior, 명시적 분자식 예측, consensus FP, DreaMS 임베딩을 analog 유사도로 대체(노트북에 "도움 안 됨" 섹션).

### 받은 노트북 (`research/ref-notebooks/`)
- `ahmedberatozer_casmi26-v4f-inference/` (0.362, 최고 공개 점수)
- `ahmedberatozer_casmi26-v2-baseline/` (0.354)
- `evgendvorkin_enveda-casmi-2026/` (0.342)
- `haideptry_enveda-casmi-2026-fast-spectral-cosine-baseline/` (0.339, 최다 투표)
- `prvsiyan_analog-propagation-casmi-2026-baseline/` (0.335, 원조 + 실험 로그, FPNet 학습 스크립트 부록 포함)

---

## 5. 리더보드 형태 (2026-09-28 07:35 UTC 다운로드, 1,694팀)

메달 규칙(1000+팀): Gold = 상위 10+0.2%·N = **13위**, Silver = 상위 5% = **84위**, Bronze = 상위 10% = **169위**.

| 구분 | 순위 | 현재 public 컷 |
|---|---|---|
| 1위 | 1 | 0.425 (Ozymandias31415, 50회 제출) |
| Gold | 13 | **0.389** |
| Silver | 84 | **0.354** |
| Bronze | 169 | **0.346** |
| 200 / 300 / 500 | | 0.343 / 0.341 / 0.334 |
| 800 / 1000 | | 0.315 / 0.238 |

- 상위권: 0.40 이상 9팀, 0.38 이상 21팀, 0.36 이상 61팀.
- **공개 노트북 클러스터**: 0.328에 140팀(prvsiyan 0.335 fork의 재실행 노이즈), 0.341에 96팀, 0.342에 55팀, 0.339에 41팀. 0.336–0.345 구간에 **290팀**, 0.339 이상이 403팀.
- 0.346–0.353(동메달대)에 80팀 → 동메달 컷은 공개 fork(0.341)보다 불과 +0.005, 이는 **시드 노이즈 폭(±0.007)** 과 같음.
- 공개 최고 노트북 v4f(0.362)는 현재 57위 수준으로 **public 기준 은메달권**. 단 비공개 데이터셋 의존 가능성.
- 제출 수 중앙값 5회, 최대 70회.

---

## public/private 상관관계 판단

- **Public은 ~130분자뿐** → 분자 1개 = ±0.0076. 은/동 컷 차이(0.354 vs 0.346 = 0.008)가 **분자 1개**에 해당. 공개 노트북 fork 수백 팀이 0.33–0.35에 몰려 있어 **private에서 대규모 셔플이 거의 확실**.
- 동일 코드 재제출 편차 0.006–0.007, HistGBM 시드만으로 흔들림 → public 소수점 셋째 자리 차이는 대부분 노이즈.
- 클래스 비율 추정(16/45/39)은 **public 130분자에 대한 LB probing** 결과. 호스트가 클래스 배정을 숨겼고 public/private가 같은 비율이라는 공지는 없음 → private에서 Class 3 비중이 다르면 "Class 1 과대 가중(W1≈0.42)" 같은 **LB 보정 상수는 과적합일 수 있음**.
- 여러 참가자가 "local holdout이 LB와 반대 방향" 사례 보고. LB를 많이 탐색한 팀(50–70회 제출)일수록 public 과적합 위험이 큼.
- 결론: **public-private 상관은 "방법 계열 수준"에서는 양(+)이지만, 0.01 이하 차이에서는 신뢰하기 어렵다.** 최종 선택은 public 점수보다 (a) 누수 없는 다중 로컬 CV, (b) 여러 시드/재실행 평균이 좋은 제출을 기준으로 해야 함. 현재 public 0.34–0.35대 fork들은 private에서 은메달을 지키기 어렵고, 은메달 안정권은 private 기준 대략 **공개 fork 대비 +0.02~0.03 이상의 실질 개선**이 필요할 것으로 추정.

## 솔로·무료 GPU 기준 전략 시사점

1. **베이스라인은 prvsiyan/haideptry 계열로 시작** — CPU 위주 retrieval + GBDT, FPNet 학습도 T4 ~2시간이라 무료 quota(30h/주)로 충분. Mac에서는 후보 풀/피처 생성·GBDT 학습·CV, Kaggle GPU는 FPNet 계열 학습에만 사용.
2. **평가 파이프라인을 metric과 100% 동일하게**: RDKit 2026.03.3 tautomer canonical InChIKey14로 dedup(슬롯 낭비 방지, ~4–8% 영향), 파싱 불가 SMILES 절대 금지, 9시간·인터넷 없음에서 도는지 매 버전 확인(숨겨진 테스트 재실행에서만 실패하는 사례 다수 → 예외 처리 + 안전한 fallback).
3. **CV 설계가 승부처**: InChIKey14 단위로 모든 라이브러리에서 제거한 Class 2 시뮬레이션, 소스 라이브러리 전체를 뺀 Class 1 시뮬레이션, timsTOF NP 패널(enveda-np-examples, 단 구조 누수 제거) 등 **2개 이상 프록시**를 두고, 풀 출처 피처 금지. 쿼리 그룹 단위 fold, 시드 평균.
4. **가장 큰 레버(커뮤니티 합의)**: 같은 분자식 **이성질체 구분** — ICEBERG(MassSpecGym 가중치, 허용)·CFM-ID 4(허용)·MetFrag 개선 등 구조 의존 증거를 ranker 피처로. 최고 공개 노트북(0.362)도 ICEBERG 이성질체 재채점이 핵심. 사전학습 모델은 추론 비용이 크므로 상위 N 후보(예: 60)에만 적용해 9시간 내 유지.
5. **풀 확장은 신중히**: PubChem 전체는 decoy 희석으로 악화, gated(신뢰도 낮을 때만)·꼬리 슬롯 전략만. Class 3 생성 후보는 MRR 구조상 상위 정답을 밀어내지 않도록 **하위 슬롯에만 삽입**하는 방식이 안전.
6. **LB 제출 운용**: 일 5회 중 1회는 동일 코드 재제출로 노이즈 폭 측정. 0.005 미만 차이는 무시. 단일 변수 A/B만.
7. **최종 2개 선택**: 하나는 로컬 CV(다중 프록시) 최고, 하나는 public·CV 절충의 견고한 앙상블. public 순위 추격용 튜닝(W1 등 LB 보정 상수)은 private에서 역전될 위험이 있으므로 prior를 넓게 평균하는 쪽이 안전.
8. **라이선스 체크**: NIST/METLIN 학습 가중치(CFM-ID 4는 예외로 허용) 금지. 메달 자체에는 무관하지만 규정 위반 시 실격 가능하므로 사용 자원과 버전을 기록.
