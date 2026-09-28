# enveda-CASMI26 은메달+ 전략 (2026-09-28 기준)

근거 문서: [01 선행 연구](01-prior-art.md), [02 대회 규칙·디스커션](02-competition-and-discussions.md), 공개 노트북 `ref-notebooks/`

## 1. 대회 요약
- MS/MS 스펙트럼(분자당 1~16개) → 분자당 SMILES 최대 25개 순위 제출. **MRR@25**
- 채점: RDKit 2026.03.3 으로 tautomer 정규화한 뒤 InChIKey 앞 14자리가 같으면 정답. 입체와 tautomer 는 무시
- 코드 대회: 9h, 인터넷 X. 5회/일, 최종 2개. 엔트리·병합 마감 12/07, 최종 12/14
- Test 약 400 분자 / 1500 스펙트럼 (Bruker timsTOF, adduct 10종). **Public 33% (약 130 분자) / Private 67%**
- 분자 3종류 (비율 비공개, LB probing 추정 16/45/39%): ① 공개 스펙트럼 라이브러리에 있음 ② 라이브러리에는 없고 PubChem/COCONUT 에 있음 ③ 어디에도 없음 (de novo)
- 외부 데이터: PubChem, COCONUT, MassSpecGym, ICEBERG/MIST, CFM-ID 4, Enveda-180 허용. **NIST, METLIN 금지**

## 2. Public ↔ Private 판단
- Public 약 130 분자 → 분자 하나가 1등으로 올라가면 약 0.008점. 같은 노트북을 다시 돌려도 최대 0.007 차이
- 은(0.354) 과 동(0.346) 차이가 public 분자 하나. 0.336~0.345 에 290팀이 몰려 있음 (공개 노트북 fork)
- **결론:** 방법 차이는 private 까지 전달되지만 0.01 미만 차이는 믿을 수 없다. 은메달을 안정적으로 따려면 공개 노트북보다 **실제로 +0.02~0.03** 높아야 한다. 판단 기준은 로컬 CV

## 3. 핵심 관찰
- 오답의 약 98% 가 **분자식이 같은 이성질체**를 1등으로 고른 경우 → 같은 분자식 후보들 사이의 순위 정하기가 본 게임
- PubChem 전체를 후보 풀에 넣으면 점수가 **떨어짐** → 후보 풀 크기보다 사전 정보(prior)가 중요
- 파싱이 안 되는 SMILES 가 하나라도 섞이면 제출 전체가 0점 → 제출 전 RDKit 검증 필수
- train 의 `inchikey14` 는 tautomer 정규화가 안 돼 있음 → 채점과 같은 키로 중복 제거
- enveda-np-examples 로 검증하면 점수가 부풀려짐 (다른 라이브러리에도 있는 구조)
- 올바른 CV: **정답 분자의 스펙트럼을 숨기고, 후보 풀에서도 빼는** 분할만 LB 순서와 맞았음 (shiferaxa)

## 4. 실행 계획

### Phase 0 — 베이스라인 재현과 CV (10/1 ~ 10/12)
1. 공개 파이프라인(prvsiyan 계열) 재현: COCONUT + train 구조 후보 → 라이브러리/유사체 매칭 + 조각 체크 + 스펙트럼→지문 Transformer + HistGBM ranker
2. **누수 없는 CV 두 벌:** ① 분자 단위 hold-out (스펙트럼과 후보 풀에서 모두 제거) ② 종류 ③ 흉내 (PubChem/COCONUT 에서도 제거)
3. 3종류별로 CV 점수를 따로 보고, 추정 비율 16/45/39 로 가중한 종합 점수를 기준으로 삼음
4. 같은 설정 2회 제출 → LB 노이즈 실측 (5회/일이라 여유 있음)

### Phase 1 — 같은 분자식 이성질체 순위 (10/13 ~ 11/15) ← 주력
1. 스펙트럼→지문 모델을 timsTOF 데이터(enveda-180, np-examples)로 fine-tune + adduct 조건 입력
2. 상위 후보에만 **스펙트럼 시뮬레이션 재채점** (ICEBERG / MARASON / CFM-ID)
3. 후보 prior 특징: COCONUT 포함 여부, NP-likeness, PubChem 참조 수 등 "얼마나 알려진 분자인가" (CASMI 2016 에서 top-1 을 30% → 70% 로 올린 요소)
4. 분자식/adduct 추정으로 후보 줄이기

### Phase 2 — 종류 ③ (de novo) 보강 (11/16 ~ 12/01)
- 상위 후보에서 메틸기, 당 등을 붙이거나 떼서 만든 유사체를 **낮은 순위 칸(16~25)** 에만 채움 (상위권을 망치지 않는 선에서)
- DreaMS 임베딩 기반 대조 학습 검색 채널 (시간이 되면)

### Phase 3 — 최종 선택 (12/02 ~ 12/10)
- 로컬 CV 종합 점수 기준으로 2개 선택. 하나는 최고 CV, 하나는 성격이 다른 강한 설정

## 5. 컴퓨트 (Kaggle 무료 GPU 만)
- 지문 Transformer 학습은 T4 기준 약 2h → 무료 quota 로 충분
- 무거운 전처리(후보 풀 구축, 지문 계산)는 Mac 에서 CPU 로 하고, 결과를 Kaggle dataset 으로 올림
- ICEBERG 같은 시뮬레이터는 상위 N 개 후보에만 적용해서 9h 안에 맞춤
