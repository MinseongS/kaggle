# 실험 큐

각 항목: 질문 → 방법 → 결과/결론. 끝난 항목은 "완료"로 옮긴다.

## 채점 중 (2026-11-07 09:10 실행에서 제출)
- **s013 ntl-areeb670 v1**: areebbb/0-670-this-actually-works 그대로. ozer672 + DeBERTa-v3 분류기 멤버(nuridudak/ntl-two-stage-encoders desklib, w 2.2) + svc_w 4, ours_svc_w 6, top34 0.15, atlas_w 0.20, atlas_margin 0.32. 기준 s008 0.668, ≥0.678이면 개선
- **s014 ntl-ozer672-debcls v1**: ozer672 + DeBERTa-v3 분류기 멤버만 (손잡이는 기준 그대로). 새 멤버 단독 효과 측정
- **s015 ntl-ozer672-mbert v1**: ozer672 + ModernBERT 분류기 멤버만 (w 2.2). s014와 비교해 어느 인코더가 나은지, 둘 다 오르면 다음은 두 멤버 동시(+fusion 비율 가중)

## 판정 끝남
- **s011 atlas_w 0.30 → 0.668 / s012 sinkhorn 0.20 → 0.666** (2026-11-07): 둘 다 노이즈. s009~s012 손잡이 변경 4회 연속 노이즈 → **손잡이 튜닝 방향 종료** (DAILY 3단계 규칙). atlas_margin·comp_w 후보도 같은 종류라 내지 않음
- **s008 ozer672 0.668 / s009 sk050 0.669 / s010 m4apex 0.669** (2026-11-06): 셋 다 노이즈 범위 안 → **기준 = s008 ozer672 원본** (규칙대로). 공개 가중 손잡이(sinkhorn_target_mix, svc_w, ours_svc_w, top34_w, sinkhorn↑)는 public에서 효과 없음. 공개 점수 0.672 vs 우리 0.668: 재현 노이즈 -0.004
- **s007 ntl-main v2** (mine 가중 0.25): **0.641** (s005 0.655, s001 0.663). 가중을 올릴수록 단조 하락 → **lite 계열을 공개 파이프라인에 섞는 방향 종료**. kernels/main 은 더 쓰지 않음
- s005 (mine 0.12) 0.655, s004 lite3 0.503 (전체/부분 sinkhorn 차이 없음 → 부분 sinkhorn 기본), s006 lite4 0.510 (+0.007, 미달)

## 다음 제출 (위에서부터)
0. s013/s014/s015 결과 확인 (약 8시간 뒤, 21:10 실행). 오른 멤버를 기준에 넣는다. 둘 다 오르면 `ozer672 + desklib + modernbert` 동시 제출. 둘 다 노이즈면 인코더 멤버 방향 1회 더(가중 4.0) 후 종료 판단
   - 21:10 실행 남은 한도 2회 후보: (a) 두 인코더 동시 (kernels/ozer672-enc2 준비됨, 아직 push 안 함, 각 w 2.2) (b) light_members.npz 내용 조사 후 결정
1. 기준(ozer672 계열) 위 한 가지 변경 시험 후보 (공개 변형들이 이미 탐색한 손잡이는 피한다):
   - (제출함 s011) `--atlas_w` 0.30, (제출함 s012) `--sinkhorn` 0.20
   - ~~`--atlas_margin`, `--comp_w`~~: 손잡이 방향 종료(2026-11-07)로 취소
   - 새 멤버 후보: ntl-two-stage-encoders/modernbert. config 확인함(ModernBertForSequenceClassification, 18 라벨) → hf_cls 분기 그대로 재사용 가능. 원작자 fusion.json 가중: modernbert 0.44 > svc4 0.33 > desklib 0.29 (oof_nll 1.049) → desklib보다 강할 가능성. → s015로 제출함 (2026-11-07, w 2.2)
2. 매 실행 공개 노트북 점검: 0.672보다 높다고 주장하는 노트북은 diff를 먼저 보고, ozer672 대비 가중 변경뿐이면 한 제출로 재현
3. (보류) omerfarukmerey 0.668 와의 블렌드: 두 파이프라인 모두 약 8시간이라 한 노트북 9시간 안에 같이 못 돌림 → 불가로 판정, 큐에서 제외

## 검토한 공개 노트북 (재현 안 함, 이유)
- 2026-11-07 ozertuu/0-672-dual-svc-better: ozer672 대비 svc_w 12, ours_svc_w 0.85, top34 0.15, sinkhorn 0.42 → s010 m4apex(0.669)와 손잡이 동일
- 2026-11-07 ozertuu/0-672-dual-svc-residual-refinement-apex: 같은 손잡이 + SVC v5 분기 삭제 → s010과 동일
- 2026-11-07 lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble: 0.666, 기준보다 낮음
- 2026-11-06 ozertuu/0-670-residual-sibling-gating-apex: m4apex(s010, 0.669)와 코드 diff가 죽은 hf_cls 분기 삭제뿐 → 같은 노트북
- 2026-11-06 aboodps/ntl-672-q14top-025: ozer672 + q14top 멤버(w 0.25). 어댑터 `q14top/meta.json`을 담은 데이터셋이 비공개(dataset_sources에 빈 항목) → 재현 불가
- 2026-11-06 whitegg/ntl-fuse33-0-668-fork-response-corpus: 0.668 계열 fork, 기준 후보 ozer672(0.672)보다 낮음
- 2026-11-06 alexchilton/name-that-llm-0-668-all-public-what-failed: 0.668, 기준 후보보다 낮음 (실패 목록은 참고용)
- 2026-11-06 omerfarukmerey/0-668-verifier-matched-rows-pool-prompts: 0.668, 블렌드 불가(시간)로 재현 보류

## 로컬 실험
- **(2026-11-06 신설, 우선) 금 컷(+0.037)용 새 신호 조사**: 상위권(0.74)과 공개(0.67) 차이는 손잡이로 안 메워짐. 디스커션·최신 공개 노트북에서 새 멤버(다른 base LLM LoRA, 응답 corpus 확장, 프롬프트 재생성 데이터)를 조사하고, ozer672에 멤버 하나를 더하는 제출 후보를 만든다. 9시간 런타임 여유 확인 필요 (ozer672 commit 로그의 단계별 시간)
- (부분 완료 e014) sinkhorn/target_mix는 자체 OOF에서 차이 ≤0.006 → 레버 아님. svc_w 등 멤버 가중은 공개 멤버 OOF가 없어 로컬 평가 불가. 남은 것: n-gram SVC 멤버 재현 후 svc_w 평가 (우선순위 낮춤)
- ~~(우선, 2026-11-06 신설) 공개 파이프라인용 후처리 로컬 평가~~: ozer672의 `infer.py`/`comp_stack.py`(outputs/k_ozer672)에서 CPU로 돌릴 수 있는 멤버(n-gram SVC, gauss)만 train OOF로 재현해 sinkhorn / sinkhorn_target_mix / svc_w 손잡이를 CV로 고른다. LB 노이즈(±0.009)로 손잡이를 고르는 대신 로컬 근거를 만든다
- 아래 자체 lite 계열 항목은 혼합 방향 종료로 우선순위 낮음
- 임베딩 모델 크기: Qwen3-4B-Base 초반 레이어 (맥 MPS, 약 2~3시간). 1.7B 대비 CV 비교
- 레이어 0~2(임베딩 바로 위) 포함 여부
- 풀 가중치 pool_w 0.25 / 0.5 / 1.0
- 풀 10000+/label: 6000에서 TF-IDF가 +0.02 올랐으므로 더 늘리는 효과 확인
- 짧은 지문(60단어 이하) 전용 처리
- (전략, 2026-11-06 확정) 자체 lite 계열(0.51)을 섞는 방식은 접음 (s005 0.655, s007 0.641). 은 컷 0.666, 금 컷 0.70대 → 0.667보다 높은 최신 공개 노트북·데이터셋을 ego-browser로 조사해 기준 파이프라인 교체 여부 검토

## 완료
- e013 계열 전용 head (같은 Qwen3-1.7B 임베딩): 효과 없음 (+0.002 이하, C 크면 하락). 계열 내 오류(58%)는 같은 특징으로 못 풀고 새 특징이 필요 → 4B 임베딩이나 다른 레이어가 우선
- e012 blend 가중 스캔: 1:1 유지 (차이 없음). 짧은 지문(60단어 이하) 정확도 0.447로 긴 지문(0.674)보다 크게 낮음
- 풀 6000/label (e008, e010, e011): TF-IDF 0.4708→0.4923, 임베딩 0.4804→0.5197, blend 0.5356→**0.5583** (m2 0.537→0.556). → s006 lite4로 제출
- s001 pub667 재현: LB 0.663 (원본 0.667과 노이즈 범위 안). 기준점 확보
- s002 lite (TF-IDF + 풀): CV 0.482 → LB 0.436. CV-LB 차이 -0.046
- s003 lite2 (TF-IDF + Qwen3-1.7B 임베딩): CV 0.538 → LB 0.504. 차이 -0.034. 임베딩 멤버로 +0.068
