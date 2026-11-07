# Experiments

| date | exp | change | CV | LB | note |
|---|---|---|---|---|---|
| 2026-11-04 | e000 | char_wb(1-5)+word(1-2) TF-IDF + LR(C=10), GroupKFold(5, prompt_id) | 0.3434 | - | 계열(family) 정확도 0.707 |
| 2026-11-04 | e000h | e000 + 프롬프트별 Hungarian 매칭 (정답 prompt_id 사용, 오라클) | 0.4438 | - | 테스트에서는 클러스터링으로 그룹 복원 필요. TF-IDF 군집 ARI 0.93, 최근접 이웃 같은 프롬프트 99.7% |
| 2026-11-04 | e001 | TF-IDF char1-5+word1-2, LinearSVC(C=0.3, balanced), train only | 0.3426 | - | m2 0.340. +sinkhorn 0.346 |
| 2026-11-04 | e002 | e001 + Open Pool v12 2500/label (match_weight 샘플링, pool_w 0.5) | 0.4708 | - | m2 0.472. +sinkhorn 0.482 (m2 0.477). 풀 학습 모델은 CV가 LB보다 0.03~0.04 높다는 보고 있음 |
| 2026-11-04 | e003 | Qwen3-1.7B-Base 고정, 레이어별 mean-pool + LR (train only) | 0.3952 | - | 레이어 2가 최고, 28(마지막)이 최저. 레이어 2,4,6,20 concat 0.4141 (m2 0.414) |
| 2026-11-04 | e004 | blend e001 + e003(2,4,6,20), log-prob 가중 2:1.75 (train only) | 0.4404 | - | m2 0.442. sinkhorn 효과 없음 |
| 2026-11-04 | s001 | 공개 0.667 노트북 (zbigniewkirchner) 그대로 재현 제출 (kernels/pub667, mins00/ntl-pub667 v1) | - | 0.663 | 기준점. 재실행 약 8시간. 원본 공개 0.667과 노이즈 범위 안 |
| 2026-11-04 | s002 | ntl-lite: e002 + 전체 sinkhorn (kernels/lite, CPU) | 0.4816 | 0.436 | CV-LB 차이 -0.046 (풀 학습 모델 보정치) |
| 2026-11-04 | e005 | Qwen3-1.7B 임베딩(레이어 2,4,6,20) + LR, train + 풀 2500/label | 0.4804 | - | m2 0.471, +sinkhorn 0.497. 풀 학습 |
| 2026-11-04 | e006 | blend e002 + e005 (log-prob 1:1) | 0.5356 | - | m2 0.537, +sinkhorn 0.538. 두 멤버가 잘 보완됨 |
| 2026-11-04 | s003 | ntl-lite2 v2: e006 + 전체 sinkhorn (kernels/lite2, T4) | 0.5382 | 0.504 | CV-LB 차이 -0.034. v1은 레이어 인덱스 버그(14 vs 20)로 폐기, 제출 안 함 |
| 2026-11-05 | e007 | e006 후처리 비교: 없음 / 전체 sinkhorn / 부분(strength 0.3, train prior 0.35). m2 컷 30회, 균형 vs Dirichlet(2) 불균형 클래스 | 0.5379 (부분, 균형) | - | 균형: none 0.537 / full 0.536 / 부분 0.538. 불균형: none 0.546 / full 0.510 / 부분 0.543 → 전체 sinkhorn은 불균형 테스트에서 -0.035 위험 |
| 2026-11-05 | s004 | ntl-lite3 v1: lite2와 같고 전체 sinkhorn → 부분 sinkhorn(0.3, prior 0.35) (kernels/lite3) | 0.5379 | 0.503 | s003 0.504와 같음 (노이즈 안). CV-LB -0.035. public에서는 전체/부분 sinkhorn 차이 없음 → public 클래스 불균형이 크지 않거나 이 모델엔 영향 작음 |
| 2026-11-05 | e008 | e002 TF-IDF SVC, 풀 2500 → 6000/label | 0.4923 | - | m2 0.497 (+0.025 vs e002). +sinkhorn 0.503. 학습 약 12분 |
| 2026-11-05 | e009 | blend e008 + e005(풀 2500 임베딩) 1:1 | 0.5445 | - | m2 0.5425 (e006 대비 +0.009, 경계선). 임베딩도 6000으로 늘려야 판단 가능 |
| 2026-11-05 | s005 | ntl-main v1: pub667 + lite2 확률(TF-IDF+Qwen3-1.7B 임베딩, 1:1) log 공간 가중 0.12, comp stack 뒤·ATLAS 앞 (kernels/main, build.py로 생성) | - | 0.655 | 기준 s001 0.663 대비 -0.008 (노이즈 범위 안이지만 하락 방향). 효과 없음 판정 → 가중 0.25 한 번 시험(s007) |
| 2026-11-05 | e010 | e005 임베딩 head, 풀 2500 → 6000/label (`src/embed_extend.py`로 새 59,107행만 임베딩) | 0.5197 | - | m2 0.514 (+0.039 vs e005). +sinkhorn 0.530 |
| 2026-11-05 | e011 | blend e008 + e010 (둘 다 풀 6000, 1:1) | 0.5583 | - | m2 0.556 (e006 대비 +0.023, 명확한 개선). +전체 sinkhorn 0.560 (m2 0.549) |
| 2026-11-05 | s006 | ntl-lite4 v1: lite3과 같고 두 멤버 모두 풀 6000/label (kernels/lite4, 데이터셋 mins00/ntl-emb-head-p6000) | 0.5583 | 0.510 | s004 0.503 대비 +0.007 (+0.010 기준 미달, 개선 인정 안 함). CV-LB 차이 -0.048로 커짐 (풀 비중이 커질수록 CV 과대). 기준 s004 0.503. CV-LB 차이 -0.035면 약 0.523 예상. head in-sample 0.952 (로컬과 일치, 풀 비중이 커져 train 적합이 느슨해짐) |
| 2026-11-05 | s007 | ntl-main v2: s005와 같고 mine 가중 0.12 → 0.25 (kernels/main build.py MINE_W) | - | 0.641 | 기준 s001 0.663 / s005 0.655. 큐 판정 규칙의 마지막 시험: 오르지 않으면 lite 계열 혼합 접기 |
| 2026-11-05 | e012 | e011 blend 가중 스캔 (TF-IDF 0.3~0.7) + 길이/계열 분해 | 0.5605 (w0.4) | - | m2 0.557 (1:1 대비 +0.0005, 무의미). 정확도 60단어 이하 0.447 → 150+ 0.674. 계열 정확도 0.813, 오류의 58%가 계열 내부. 최약 클래스 glm-5.3 0.23, claude-fable-5 0.28, gemini-3.7 0.29 |
| 2026-11-05 | e013 | 계열 전용 LR head (Qwen3-1.7B 임베딩, train+풀6000, src/fam_head.py)를 e011의 계열 내 조건부 확률에 log 혼합 a | 0.5583 (a=0) | - | C0.02: a0.3 m2 0.5588 (+0.002, 노이즈), a1.0 0.528. C0.1: 전부 하락. 같은 특징으로는 계열 전용 head 효과 없음 → 병목은 특징. 접음 |
| 2026-11-06 | s008 | 공개 0.672 ozertuu dual-svc-blend residual refinement v5 그대로 재현 (kernels/ozer672, mins00/ntl-ozer672 v1) | - | 0.668 | **새 기준점** (s001 0.663 대비 +0.005, 공개 0.672 재현 범위). 기준 교체 후보. commit 정상(약 7분, 3행 예측 생성). 기준 s001 0.663 |
| 2026-11-06 | s009 | ozer672 + `--sinkhorn_target_mix` 0.35 → 0.50 한 가지 (aboodps/ntl-672-sinkhorn-050 그대로, kernels/ozer672-sk050 v1) | - | 0.669 | s008 대비 +0.001 (노이즈, 효과 없음). s008과 비교. commit 정상 (Traceback 0, 3행 예측) |
| 2026-11-06 | s010 | ozertuu/0-670-residual-sibling-gating-m4-apex 그대로 (kernels/m4apex v1). ozer672 대비 ours_svc_w 1.2→0.85, svc_w 10→12, top34_w 0.20→0.15, sinkhorn 0.30→0.42 | - | 0.669 | s008 대비 +0.001 (노이즈, 효과 없음). 작성자 주장 0.670+. s008과 비교. commit 정상 |
| 2026-11-06 | s011 | ozer672 + `--atlas_w` 0.22 → 0.30 한 가지 (kernels/ozer672-atl030 v1) | - | 0.668 | 기준 s008 0.668과 같음 (효과 없음). ATLAS 비중 증가 |
| 2026-11-06 | s012 | ozer672 + `--sinkhorn` 0.30 → 0.20 한 가지 (kernels/ozer672-sk020 v1) | - | 0.666 | 기준 s008 대비 -0.002 (노이즈, 효과 없음). 손잡이 연속 4회 노이즈 → **손잡이 튜닝 방향 종료**. s010(0.42)·s009(target_mix↑)가 무변화 → 반대 방향 확인. s004 로컬 근거(불균형 private 대비) |
| 2026-11-06 | e014 | 부분 sinkhorn 손잡이 스캔 (strength 0~0.6 × target_mix 0~1), e011 OOF, m2 컷 30회 균형/Dirichlet(2) 불균형 (src/sk_scan.py) | 0.5636 (s0.6,ρ0.8 균형) | - | 전체 범위가 균형 0.558~0.564, 불균형 0.550~0.568 (차이 ≤0.006, 노이즈 수준). 불균형 최고 s0.1~0.2/ρ0.35. **sinkhorn 손잡이는 점수 레버가 아님** (s009/s010 LB 무변화와 일치). proxy(자체 OOF) 한계 있음 |
| 2026-11-07 | s013 | areebbb/0-670-this-actually-works 그대로 (kernels/areeb670 v1). ozer672 대비: DeBERTa-v3 분류기 멤버(nuridudak/ntl-two-stage-encoders desklib, hf_cls, w 2.2) 추가, svc_w 10→4, ours_svc_w 1.2→6, top34 0.20→0.15, atlas_w 0.22→0.20, atlas_margin 0.36→0.32 | - | pending | 기준 s008 0.668. 작성자 공개 0.670. commit 정상 (Traceback 0, 3행) |
| 2026-11-07 | s014 | ozer672 + DeBERTa-v3 분류기 멤버만 (w 2.2, 손잡이는 기준 그대로; kernels/ozer672-debcls v1) | - | pending | 기준 s008 0.668. 새 멤버 단독 효과. s013과 비교하면 손잡이 묶음 효과도 분리 |
| 2026-11-07 | s015 | ozer672 + ModernBERT 분류기 멤버만 (nuridudak/ntl-two-stage-encoders modernbert, hf_cls, w 2.2, 손잡이 기준 그대로; kernels/ozer672-mbert v1) | - | pending | 기준 s008 0.668. 원작자 fusion 가중 modernbert 0.44 > desklib 0.29. commit은 메인 추론을 건너뛰어(3행 더미) 로컬 MPS로 로드·라벨 확인: 18 라벨 일치, train 200행 정확도 0.78 |
