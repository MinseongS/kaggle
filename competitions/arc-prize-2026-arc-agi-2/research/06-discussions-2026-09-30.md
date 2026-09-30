# 06. Discussion / 공개 노트북 업데이트 (2026-09-30)

- 조사일: 2026-09-30 (마감 11-02, D-33). 이번에는 Kaggle에 로그인한 상태로 봤다. 02/03 문서에 이미 있는 내용은 되풀이하지 않고, 새로 나온 것만 적었다.
- 표기: [확인]은 페이지·코드·로그에서 직접 본 것이다. [추론]은 그것을 바탕으로 내가 해석한 것이다.
- 노트북 코드는 `kaggle kernels pull`로 받아 grep했다. 노트북 점수는 각 페이지의 "Public Score(현재 버전)"와 "Best Score(최고 버전)"를 기준으로 적었다.

---

## 0. 현재 LB 상태 (전체 CSV 기준, 2,260팀)

- [확인] 상위권: 1위 Tufa Labs 83.06(제출 39회), 2위 rabbithole 79.72(103회, 09-29에 76.94에서 올라옴), 3위 nvbanana 75.42(130회), 4위 Yi-Chia Chen 55.14. 5~13위는 37.50~34.03이다. 5위 Kha Vo 37.50, 6위 okiek 37.36은 제출 7회, 9위 Anil Thomas 36.81은 12회다. LB 히스토리: https://arc3.huikang.dev/leaderboard/?comp=arc2
- [확인] 메달 컷(현재 public 기준): gold는 14위로 33.89다(10 + 0.2%×2260 ≈ 14). silver는 113위로 32.22, bronze는 226위로 31.67이다.
- [확인] 점수 분포는 NVARC 계열 값에 몰려 있다. 32.22가 55팀, 31.81이 78팀, 31.39가 114팀, 30.97이 132팀, 30.56이 133팀, 30.14가 123팀, 29.72가 88팀이다. 32.22보다 높은 팀은 82팀뿐이다.
- [추론] 32.22 동점 55팀이 82~137위에 걸쳐 있다. Kaggle은 동점이면 먼저 낸 팀을 위에 두므로, 지금 32.22를 받아도 순위는 약 137위라 silver 밖이다. public 기준으로 silver를 안전하게 받으려면 32.36 이상, 여유 있게는 32.64 이상이 필요하다. gold는 34.03 이상이어야 한다. 다만 메달은 private 점수로 정해지므로, 이 컷은 참고용일 뿐이다.

## 1. 공개 노트북: 32점 이상이 무엇을 바꿨나

**요약: 최근 2주 동안 새로 32점 이상을 받은 공개 노트북은 없다.** 32점 이상은 전부 2~5개월 전 버전이고, 최근 버전들은 31.81이 최고다.

| 노트북 | 변경점 | 버전별 점수 | 판정 |
|---|---|---|---|
| [koushikrudra/failed-in-aimo](https://www.kaggle.com/code/koushikrudra/failed-in-aimo) | NVARC 원본(4B sft139, r256, lr 5e-5, 128 TTT, 16 view, p≥0.2, kgmon) | V1 33.89(4월). 최신 V2(8월 Save)는 Qwen2.5-coder-7B 경로를 추가했지만 model_sources에는 없음 | 시드 운 [추론] |
| [mikelou1/...minimal-perfpatch](https://www.kaggle.com/code/mikelou1/arc-agi2-lb33-89-minimal-perfpatch) | logits 핫스팟 GPU 패치(03 문서 참조) | V2 32.22. 포크들은 31.81, 31.39 | 운 |
| [luxluxshan/arc2-nvarc-v1](https://www.kaggle.com/code/luxluxshan/arc2-nvarc-v1) | 절반 비용 2-pass 풀링 | V1 32.22에서 **V3 29.03** | 운. 풀링 효과는 확인되지 않음 |
| [rokaiyasomapti/reproduce-nvarc-2025-results](https://www.kaggle.com/code/rokaiyasomapti/reproduce-nvarc-2025-results) | 원본 재현 | V2 32.22 | 운 |
| [chiakazirim/learned-from-aimo](https://www.kaggle.com/code/chiakazirim/learned-from-aimo) | 포크 | V1 32.22에서 V7 16.39 | 운(그리고 망가진 버전) |

- [확인] **같은 코드를 그대로 다시 제출하면 29.3~31.8 사이로 나온다.**
  - [manderson240/arc-agi-2-fork-lb33-89-20260903](https://www.kaggle.com/code/manderson240/arc-agi-2-fork-lb33-89-20260903): "No hyperparameters, code, or logic ... changed" 상태로 11버전을 냈고, best는 V4 31.81, 현재는 29.31이다.
  - [qiuqiuh/arc-highscore-lb3389-replica](https://www.kaggle.com/code/qiuqiuh/arc-highscore-lb3389-replica): 15버전, best V1 31.81, 현재 30.56, 그 포크는 28.47이다.
  - [yusuketogashi/arc-baseline-rebuild](https://www.kaggle.com/code/yusuketogashi/arc-baseline-rebuild): 100버전, best V73 31.81, 현재 26.53이다.
- [확인] **33.89가 perfpatch 수치 연산 덕분이라는 가설은 반증됐다.** [johntaylorai/swarm-arc2-numerics-control](https://www.kaggle.com/code/johntaylorai/swarm-arc2-numerics-control)은 arc_solver를 원본 수치 연산으로 되돌리고 28.89를 받았다. 같은 저자의 perfpatch 버전([swarm-arc2-lb3389-baseline](https://www.kaggle.com/code/johntaylorai/swarm-arc2-lb3389-baseline))은 29.72였다. 저자 메모: "the original notebook as published cannot run today - stale model path. Anyone currently scoring 33.89 is running a path-fixed variant."
- [확인] **2025 NVARC 원본 제출은 23.33, 같은 코드의 올해 재현은 32.22였다.** CPMP는 public 데이터가 같다고 확인했다: "It is the semi private data ... It looks like we were unlucky last year in our sub." (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/743880). 같은 코드의 run 간 폭이 **약 23~34**라는 뜻이다.
- **selection과 scoring 변경: 효과 없음 또는 노이즈 수준**
  - [확인] [dragoctlin/arc-selection-stability-vs-likelihood](https://www.kaggle.com/code/dragoctlin/arc-selection-stability-vs-likelihood): 같은 후보 풀(eval 38 puzzle, commit 로그)에서 kgmon 11.3 = probmul_3 11.3, stability_consensus 10.8, stability와 worst_case 각 10.3이었다. public은 30.14.
  - [확인] [medvax/...symbolic-fallback](https://www.kaggle.com/code/medvax/arc-agi-2-qwen-ttt-with-symbolic-fallback): eval 120 전체 run에서 kgmon 28.33, probmul_3 29.17이었다(task 1개 차이). V1 public은 28.89. 간단한 rule search는 eval에서 0 task를 풀었다.
  - [확인] [chrisvas123/...invariant-verifier](https://www.kaggle.com/code/chrisvas123/arc-agi-2-nvarc-invariant-verifier): train 쌍에서 뽑은 invariant(palette, hist 등 6종)를 어기는 후보에 kgmon −1 패널티를 줬다. public 29.72, 4 task 로그에서는 baseline과 같았다. Paper Track writeup이 있다고 함.
  - [확인] [nitish5236goel/arc2-ttt-v2](https://www.kaggle.com/code/nitish5236goel/arc2-ttt-v2): TTT wall-clock cap(TrainerCallback), DFS 내부 task deadline, score_aug 길이 정규화, RRF(kgmon+probmul) scorer, DSL 후보 투표 추가, smart [[0]] fallback을 넣었다. best V3 29.86. 코드에 qwen3_2b_grids15_sft141와 TRM(cpmpml/arc-prize-trm-031) 경로가 있다. 로그 실측: task `981571dc`는 TTT에 812.6s, 전체 1211.8s가 걸렸다.
- **예산과 재시도 계열: 견고성은 좋아지지만 점수 이득은 확인되지 않음**
  - [확인] [yusuketogashi](https://www.kaggle.com/code/yusuketogashi/arc-baseline-rebuild) Program079: Primary pass 뒤 빈 출력(관측 10개)과 후보 1개뿐인 출력(관측 17개)만 최대 24 task, 75분 안에서 다시 돌린다. 이때 Top1은 고정하고 attempt_2에만 추가한다. exact symbolic rule은 빈 출력과 singleton 출력 **0개**를 덮었다. 9h30 안전 목표를 두고 NumPy 경로 오염으로 291초 만에 죽는 사고를 고쳤다. public은 26.53~31.81 사이다.
  - [확인] [ghazarosbarseghyan91/fork-of-nvarc-plus-adaptive](https://www.kaggle.com/code/ghazarosbarseghyan91/fork-of-nvarc-plus-adaptive): Phase 2에서 남은 시간을 도달하지 못한 task와 후보가 2개 미만인 출력에 쓴다(새 seed, 24 view, DFS p 0.1). 저자가 기록한 **실패 실험**: extra 7B 추가 27.64, MCTS 포크 28.06, T4에서 Unsloth docker 없이 0.00, JEPA로 TTT 대체. Qwen3.8-27B-fp8 모델도 입력에 넣었다. best V1 31.81.
  - [확인] [dalezhong/arc-agi2-nvarc-plus-ckpt](https://www.kaggle.com/code/dalezhong/arc-agi2-nvarc-plus-ckpt): NVARC+ v2 2-pass에 submission.json을 조금씩 계속 저장하는 daemon을 붙였다. 31.39.
- **기타**
  - [확인] 모델 조합: 확인한 공개 노트북 20여 개가 전부 `qwen3_4b_grids15_sft139` 하나만 쓴다. qwen3_2b sft141로 점수를 공개한 노트북은 없다.
  - [확인] 합성 데이터와 추가 SFT: [manderson240/arc-agi-2-v8-synth-lora-20260916](https://www.kaggle.com/code/manderson240/arc-agi-2-v8-synth-lora-20260916)은 이름과 달리 markdown에 "verbatim fork"라고 적혀 있고 점수는 29.31이다.
  - [확인] 코드 모델 추가: [denizeryilmaz/arc2-qwen4b-plus-coder7b-induction-exp](https://www.kaggle.com/code/denizeryilmaz/arc2-qwen4b-plus-coder7b-induction-exp)는 28.89다.
  - [확인] 학습셋 isomorphism retrieval: finalsunflower/qwen3-specialist-v3는 best 30.14였고 최신 버전은 0.00이다. tharunkumar369/ultra-reasoner는 best 30.56이었고 최신 버전은 0.00이다.
  - [추론] 위 결과를 합치면 결론은 하나다. 30~32 구간의 공개 노트북 변경은 모두 run 노이즈(σ 약 1~1.5점, 최대 폭 약 5점) 안에 있다. 효과가 입증된 변경은 하나도 없다.

## 2. public/private, 변동성, 런타임, 큐, 마운트 경로

- [확인] **L4 큐 정체 (9/28~9/30)**: commit이 7~17시간씩 대기했다(ACoder, NguyenThanhNhan, Giba). CPMP도 "my submission did not kick off at all"이라고 했다. Kaggle Staff Dustin: "Demand has increased considerably, I increased capacity". 9/30 고정 댓글: "Queue should be cleared, and capacity held across peak today". María Cruz는 "looking at ways to prioritize certain workflows"라고 했다. L4x4를 쓰는 Gemma 4 Developer Agent 대회가 원인으로 거론됐다. 링크: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/743927 , https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/689054
- [확인] 한 참가자는 submission은 바로 시작되고 commit만 오래 대기하는 것 같다고 관찰했다(LavanBth99). CPMP의 경험은 이와 반대였다. [추론] 마감 직전 주에 다시 막힐 가능성이 크다.
- [확인] **입력 마운트 경로가 run마다 다르다.** johntaylorai: "Kaggle mounts the competition data with or without a `competitions/` prefix on different runs". 대응은 두 경로를 차례로 확인하는 `_arc_dir()` resolver다. 모델 경로도 `/kaggle/input/qwen3_4b_grids15_sft139/...`(옛 경로)와 `/kaggle/input/models/sorokin/...`(현재 경로)로 갈린다. V1과 dalezhong은 `/kaggle/input/**/config.json` glob을 쓴다. 우리의 FileNotFoundError와 같은 현상이다.
- [확인] CPMP의 "weird timeout" 스레드(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/738313)에 후속이 있다. 다른 참가자가 9시간 이내에 system error를 받았고 **Kaggle이 일일 제출 기회를 돌려줬으며**, 재제출은 성공했다.
- [확인] rerun 구조를 다시 확인했다(CPMP, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/740422). 240개를 한 번에 돌리고, 마감 뒤 따로 다시 돌리지 않는다. 85% 보너스는 private 기준이라고 봄(CPMP, 742796).
- [확인] LB 히스토리에서 본 실행 시간: Tufa Labs와 rabbithole은 거의 매번 12h00~12h30(난독화 ±10분 포함), nvbanana는 8~11시간이다.
- [추론] public/private shake-up을 직접 다룬 새 스레드는 없다. 다만 NVARC 계열이 약 700팀이고 σ가 1점 이상이므로, 이 구간은 private에서 크게 섞일 것이다.

## 3. 상위 3팀, 규정, 합성 데이터

- [확인] 상위팀 구성(LB CSV)
  - Tufa Labs: driessmit1, jeroencottaar, pressman1, stefano1283
  - rabbithole: cookizesong, xiezejian, yuhuake240506
  - nvbanana: cpmpml, darraghdog
- [확인] 방법은 여전히 비공개다. CPMP: "you'll have to wait till competition end ... I reckon it is the same for the other team above 60%" (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/741315).
- [확인] "LLM-agent is all you need" 스레드(17위 Van-Phuc Huynh, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/742796)
  - 로컬에서 30~50%를 봤다는 주장이 있다(Russell Kirk).
  - 반례로 Qwen3.8 기반 agent 제출은 LB 13.75%였다(Gladstone025).
  - [추론] 상위팀이 12시간을 다 쓰는 것은 대형 모델 추론이나 search 계열과 맞지만 근거는 없다.
- [확인] **Writeup 제출처와 라이선스가 확정됐다.** Solution writeup은 Paper Track(https://www.kaggle.com/competitions/arc-prize-2026-paper-track)에 내고 CC BY 4.0으로 자동 공개된다. 코드는 CC BY 4.0, MIT, CC0 중 무엇이든 된다(Addison Howard, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/743581).
- [확인] OSAID와 incompatible-license carve-out 충돌 질문(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/740268)은 **아직 답이 없다**(댓글 0).
- [확인] NVARC 체크포인트: 4B sft139와 2B sft141 둘 다 CPMP가 "tunable"로 바꿨다(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/735058).
- [확인] AI 코딩 어시스턴트: CPMP는 "the only real limit is to not spend unreasonable amount of money"라고 했다(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/737679). 공식 답변은 아니다.
- [확인] 합성 데이터와 continual SFT 결과(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/733930)
  - Koushik Rudra: "I trained it again & again with a synthetic puzzle but couldn't get the expected score."
  - Ravindra: NVARC 체크포인트에 추가 SFT를 했지만 "my dataset is too easy for that model", 어려운 ARC-AGI-2 task에서는 개선이 적었다.
  - CPMP: "We shared everything you need to train that model"(1ytic/NVARC GitHub와 2025 writeup).
  - [추론] 쉬운 합성 데이터로 추가 SFT를 하면 효과가 없다. 효과를 보려면 NVARC 수준의 어려운 합성 데이터(수십만 개 규모)가 필요하고, 그만한 컴퓨트는 Kaggle 밖에서 구해야 한다. 스폰서 컴퓨트 요청에는 답이 없었다고 함(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/742074).

## 4. 4×L4 하드웨어

- [확인] 로그에 찍힌 GPU 정보는 "NVIDIA L4 ... Max memory: 22.034 GB"(Unsloth 배너)다.
- [확인] TTT 실측(dragoctlin 로그, 128 step 기준): 작은 task는 70~115s, 큰 task는 350~410s, `981571dc`는 852s였다. task 전체 시간은 180~1320s였다.
- [확인] 조사한 노트북은 대부분 `OMP_NUM_THREADS=12`(4 worker)를 쓰고, ultra-reasoner만 4를 쓴다.
- [미발견] nproc와 RAM을 실측한 글이나 로그는 찾지 못했다. 05 문서의 48 vCPU/192GB 가정은 여전히 검증되지 않았다. 직접 Save를 돌려 `nproc; free -g`를 찍어야 한다.

---

## 우리에게 주는 시사점 (기대값 순)

1. **경로와 모델을 glob으로 찾는 resolver를 필수로 넣는다.** 데이터는 `competitions/` 접두사가 있는 경로와 없는 경로를 차례로 확인하고, 모델은 `/kaggle/input/**/config.json` glob으로 찾는다. 이미 제출 1회를 FileNotFoundError로 잃었다. 비용은 0이고 실패 확률만 줄인다.
2. **공개 노트북에서 가져올 "개선"은 없다고 결론낸다.** 32점 이상은 전부 옛 버전의 시드 운이다. 같은 코드도 23~34까지 흔들리고, 최근 2주 동안 나온 selection, invariant, DSL fallback, 7B 추가, 재시도 계열은 모두 29~31.8에 머문다. 공개 포크를 A/B 비교하는 데 제출 기회를 쓰지 않는다.
3. **분산 축소와 커버리지 방어에 집중한다.**
   - TTT wall-clock cap, DFS 내부 deadline, cheap-first 순서, 빈 출력과 singleton 출력 재시도, 부분 submission 저장을 넣는다. 이런 변경은 평균을 조금 올리고 꼬리 위험(타임아웃, 빈 task)을 없앤다.
   - yusuketogashi 로그 기준으로 빈 출력은 약 10개, singleton은 약 17개였다. private에서도 몇 task가 걸린 문제다.
4. **public 컷의 의미를 바로 본다.** public silver 컷 32.22는 동점 55팀 때문에 사실상 32.36 이상이고, gold는 34.03 이상이다. 하지만 NVARC 계열 약 700팀이 private에서 섞이므로 public 점수를 쫓는 것은 의미가 약하다. 로컬 eval(120)에서 paired 비교로 이득이 확인된 변경만 넣는다.
5. **메달을 노릴 수 있는 차별화 경로는 여전히 "모델 자체"다.**
   - 쉬운 합성 데이터 SFT는 실패 사례가 두 건 있다. 하려면 NVARC 레시피(1ytic/NVARC) 수준의 어려운 데이터와 외부 컴퓨트가 필요하다.
   - 2B sft141을 4B와 앙상블한 공개 결과는 없다. 로컬 eval로 직접 확인할 가치가 있는 미탐색 영역이다.
6. **마감 운영 계획을 세운다.** L4 큐 대기가 7~17시간까지 늘어난 적이 있다. 최종 후보는 10월 넷째 주까지 확정하고, 마지막 주는 제출이 막힐 수 있다고 가정한다. system error가 나면 일일 제출 기회를 돌려받을 수 있다.
7. **상금은 우리 목표가 아니지만 기록은 남긴다.** writeup은 Paper Track에 내고 코드 라이선스는 CC BY 4.0, MIT, CC0 중에서 고른다. Qwen 계열의 OSAID 문제는 아직 답이 없다.
