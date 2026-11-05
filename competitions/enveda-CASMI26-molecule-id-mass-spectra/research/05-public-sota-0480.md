# 공개 SOTA 분석 (2026-11-04, public 0.480)

출처: `alexchilton/casmi26-lb-0-480-neural-ensemble`, `heliosli/casmi26-0-481` (둘 다 T4 GPU 노트북).
0.480 노트북은 **0.465 파이프라인 노트북을 base64 로 통째로 내장**하고 그 위에 후처리 층을 쌓은 구조.
(내장본 출처: dmitriigluzdov 의 attributed v4g reproduction 계열, canqiang E006 기반. 수십 명의 fork 가 누적된 결과)

## 리더보드 (11/04)
- 4,650팀. 1위 0.525, 금 경계(~19위) 0.485, **은 경계(~232위) 0.480**, 동 경계(~465위) 0.479
- 0.480 동점 312팀, 0.477 이상 700팀+ → 대부분 같은 공개 노트북. Public 은 약 124분자라 분자 1개 = 0.008
- 동점이면 private 점수도 거의 같고, 먼저 제출한 팀이 앞선다 → **공개 노트북 그대로는 은메달 불가**

## 0.465 파이프라인 구성 (내장 노트북, 59 cells)
1. **Engine 1 (v4n, Ahmed Berat Özer)**: train 스펙트럼 라이브러리 + 후보 풀(COCONUT, ChEBI, LIPID MAPS, train 구조) + FPNet A/B, full1 + LightGBM ranker 4개 + 유사체 생성기
2. **Engine 2 (two-ranker BIO + AFIX)**: 별도 프로세스, PubChemLite 후보 포함, depth 6 트리
3. **PubChem 채널**: PubChem 약 1.06억 구조를 질량순으로 저장, 10ppm 창에서 FP 점수
   - `S = z(FP 점수) + 0.15·pop`, `pop = ln(1+substance 수) + ln(1+PubMed 수)` (+ 스펙트럼 라이브러리 기탁 +0.5, NP DB 수록 +1.0, LOTUS 참조, 동의어 수)
   - pop prior 만으로 panel MRR 0.512 → 0.951 (정답 풀에서 제거 상황). **"얼마나 알려진 분자인가" 가 최대 레버**
4. **Gated PubChem merge**: 라이브러리 매치가 약할 때(lib_max < 0.9)만 PubChem 후보를 2,4,6,8,10 또는 4,8,…,20 자리에 끼움. 1등 승격 조건 `S > 6 & pop ≥ 5` 또는 margin > 150
5. **두 엔진 RRF 융합**: `1/(3+r_main) + 0.6/(3+r_e2)`, top 40
6. **ICEBERG + GLACIER 순방향 시뮬레이터**: 같은 분자식 그룹 내 `z(score) + z(ICE) + z(GLACIER)` 재정렬. GLACIER 는 [M+H]+ 만 (+0.006 LB)
7. metric identity 감사, fragment 점수, Engine 2 1등을 2위에 삽입 등

버전 이력 (public): v1 0.450 → v9 0.455 (트리 깊이) → v13 0.456 → … 0.465

## 0.465 → 0.480 후처리 층 (전부 public LB 로 튜닝)
- 신경망 scorer (`sanpier/casmi-spectrum-model-v1`) 점수로 RRF (V1 w=0.25 → 0.466), 2~25위만 w=0.5 (V2 0.468), CUSTOM w_mid 0.75 (0.477)
- 공개 모델 + 자체 모델 z 평균 앙상블
- fragment check 로 1위 교체, PubChem pop/스펙트럼 라이브러리/NP 보너스로 2~25위 재정렬, InChIKey14 중복 제거, 빈 칸 채우기
- **해석**: 0.465 → 0.480 은 분자 약 2개 차이. 124분자 public 에 맞춘 가중치라 private 에서 대부분 평균 회귀할 가능성

## 데이터셋 크기 (로컬 디스크 여유 31GB)
| dataset | 크기 |
|---|---|
| casmi26-pubchem-popularity-prior | **약 80GB** (fingerprints-*.bin 12개 × 5.8GB, pc_smiles 5.5GB) |
| casmi26-attributed-v4g-reproduction-assets | 약 4GB |
| casmi26-fold-safe-fpnet | 약 1GB |
| natural-product-panel-benchmark | 0.23GB (250 np-examples 분자의 후보·점수 replay, identity/scaffold fold 포함) |
| natural-product-knowledge-table | 0.14GB |
| casmi-spectrum-model-v1 | 0.06GB |

→ 전체 파이프라인은 **로컬에서 못 돌림. Kaggle 에서 실행**하고 후보 evidence 만 내려받아 로컬에서 분석·학습한다.

## 함의
- 현재 상위권 = 사람 손으로 튜닝한 규칙(RRF 가중치, 게이트, 슬롯)의 더미. 검증은 250개 np panel(249개가 COCONUT 수록, 모델이 본 분자일 수 있음) + public LB 뿐
- 우리 차별점 후보: 모든 채널의 evidence(엔진 점수, FP, pop prior, 라이브러리 유사도, ICEBERG/GLACIER, 신경 scorer)를 한 테이블로 모아 **누수 없는 hold-out 분자로 학습한 stacker(LambdaRank)** 로 규칙들을 대체
