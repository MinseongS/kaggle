# ARC Prize 2026 – ARC-AGI-2: 대회 규정 및 Discussion/Code 조사

- 조사일: 2026-09-28 (최종 제출 마감 2026-11-02, D-35)
- 출처: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2 (Overview / Data / Rules / Leaderboard / Discussion / Code), 커뮤니티 LB 히스토리 https://arc3.huikang.dev/leaderboard/?comp=arc2
- 표기: **[공식]** = 대회 페이지·Rules·호스트(Greg Kamradt)·Kaggle Staff 발언. **[탑팀]** = CPMP(nvbanana, 현재 3위, 2025 우승팀 NVARC) 발언. **[커뮤니티]** = 일반 참가자의 측정·추측.
- 참고: 조사에 쓴 브라우저 세션은 Kaggle에 로그인되지 않은 상태였음. 공개 페이지만 읽었고 Data 파일 미리보기 등 로그인 전용 영역은 보지 못함.

---

## 1. 대회 규정 요약

### 1.1 평가 지표 [공식]
- 태스크의 test output마다 **2 attempts**(attempt_1, attempt_2)를 냄. 둘 중 하나라도 ground truth와 **완전히 일치**하면 그 output은 1점, 아니면 0점. 셀 단위 부분 점수는 없음.
- **태스크 안에서는 부분 점수가 있음**: test input이 N개인 태스크에서 k개를 맞히면 그 태스크 점수는 k/N. 전체 점수는 태스크 점수의 평균 (CPMP 설명, 호스트가 반박하지 않음: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/699858).
- submission.json에는 모든 task_id가 있어야 하고 attempt_1/attempt_2 키가 모두 있어야 함 (Overview > Evaluation).

### 1.2 테스트셋 구성 [공식]
- 재실행(rerun) 시 **숨겨진 240개 태스크**(`arc-agi_test_challenges.json`)로 교체됨. 대부분 test input이 1개이고 일부는 2개 (Data 탭).
- **Public LB = 120개 "semi-private" 태스크, Private LB = 다른 120개 "private" 태스크**. Greg Kamradt(호스트) 답변: "Public leaderboard is based on 120 semi private tasks, the private leaderboard is 120 private tasks" (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/699858). LB 페이지 문구: "approximately 50% of the test data … final results will be based on the other 50%".
- 즉 public과 private는 **한 번의 실행에서 채점되는 서로 다른 두 절반**임. private가 public을 포함하는 구조가 아님. ARC Prize Foundation 쪽 구분으로 보면 두 세트는 성격이 다름 (semi-private vs private).
- 재실행은 따로 없음 [탑팀]: "The hidden scores of these two submissions are retrieved (nothing is rerun)" (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/740422). 노트북은 제출할 때 240개 전체를 한 번에 돌리므로, 마감 뒤 private 재실행에서 타임아웃이 나는 식의 위험은 이 대회 구조에 없음. 성공적으로 채점된 제출은 private 점수가 이미 계산되어 있음.
- [탑팀] 테스트 데이터는 2025년 대회와 같고, public LB 데이터는 arcprize.org ARC-AGI-2 리더보드의 semi-private 세트와 같음 (CPMP: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/685142).

### 1.3 하드웨어 / 런타임 [공식]
- 노트북 제출만 가능. **CPU/GPU 모두 12시간 이하**, **인터넷 차단**, 출력 파일명 `submission.json`.
- GPU: **L4 x4 (총 96GB VRAM)**. L4는 이 대회에 붙인 노트북에서만 쓸 수 있고, L4 세션은 인터넷이 꺼져 있어야 함. 개발용 노트북은 T4x2/P100보다 GPU 쿼터를 2배 빨리 씀 (Overview > Upgraded Accelerators, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/689054).
- H100은 ARC-AGI-3 전용이고 ARC-AGI-2에는 제공하지 않음 (Kaggle Staff Addison Howard, 689054).
- 제출 런타임 표시는 난독화되어 있음: 같은 제출도 점수 수신 시각이 최대 10분 달라질 수 있음 (Code Requirements).
- [탑팀] 제출 실행에 드는 GPU는 개인 주간 쿼터(30h)에서 빠지지 않음. 쿼터는 개발/테스트용 (CPMP, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/694745).
- [탑팀] 이 대회는 가속기가 자동으로 꺼지므로 제출 전에 먼저 Save로 검증하라고 권함 (CPMP, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/697859).

### 1.4 제출 한도 / 팀 [공식]
- **하루 1회 제출**, 최종 **2개 선택**. 팀은 최대 5명. Entry/Team Merger 마감 2026-10-26, 최종 마감 11-02, 발표 12-04 (Rules 2.1, 2.2, Timeline).
- 제출 한도를 늘려 달라는 요청에 호스트는 "The Semi Private set shouldn't be used to test against but rather used to validate your submission in general"이라고 답함 (Greg Kamradt, 694745). 호스트가 public LB를 튜닝 대상으로 쓰지 말라고 분명히 밝힌 셈임.
- ARC-AGI-2 대회는 올해가 마지막이고, 그래서 Grand Prize 지급을 보장함 (Greg Kamradt, 694745).

### 1.5 상금 / 메달 / 오픈소스 [공식]
- 총 $700K. Progress Prize $275K(1위 $75K부터 8위 $15K까지), Grand/Innovation Prize $275K(Solution Writeup을 6개 기준으로 심사), Bonus $150K(LB **85% 이상**을 달성한 상위 5팀에 분배).
- **메달과 포인트 지급함** ("Awards Points & Medals" 표시).
- 상금 자격: 솔루션을 오픈소스로 공개하지 않으면 제외. Winner License는 **CC BY 4.0**. Rules 2.5.a는 OSAID 기준의 open source system/model/weights를 요구하지만, 같은 조항에 라이선스가 맞지 않는 pretrained model/data는 예외로 한다는 문구가 있음. 이 충돌에 대한 질문(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/740268)에는 아직 공식 답변이 없음.
- Grand Prize writeup: 마감 후 7일 안에 제출해야 하는데 Writeups 탭이 없음. 라이선스도 Kaggle(CC BY 4.0)과 arcprize.org(CC0/MIT-0) 표기가 다름. 둘 다 질문만 있고 답변이 없음 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/743581).
- [탑팀] 85% 보너스 기준은 private 점수 기준일 것이라고 봄 (CPMP, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/742796). 공식 확인은 안 됨.

### 1.6 외부 데이터 / 모델 / 도구 [공식]
- 무료로 공개된 외부 데이터와 pretrained model 사용 가능 (Code Requirements, Rules 2.6). 비용은 "Reasonableness" 기준으로 판단.
- AI 코딩 어시스턴트 사용 가능. 최종 노트북만 오프라인에서 돌아가면 됨 (Greg Kamradt, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/701528).
- Rules 3.4.b: 검증/테스트 데이터를 손으로 라벨링하거나 사람이 예측해서 쓰는 것도 허용한다는 Kaggle 기본 문구가 있음. 다만 test는 숨겨져 있어서 실제 의미는 없음.
- [탑팀] NVARC Qwen3 체크포인트(`sorokin/qwen3_4b_grids15_sft139`, `qwen3_2b_grids15_sft141`)는 처음에 Kaggle Models에서 "not fine-tunable"로 설정되어 있었고 약 1달 전에 tunable로 바뀜. "You should always read the license of every model you use" (CPMP, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/735058).

---

## 2. 점수 양자화 (step 약 0.28 / 0.14)

- 계산: public 점수 = 100 x (태스크 점수 합) / 120. 태스크 하나를 통째로 맞히면 +0.833, 2-output 태스크에서 하나만 맞히면 +0.417, 3-output 태스크에서 하나만 맞히면 +0.278.
- 검증: 현재 LB 점수 x 1.2가 모두 1/6의 배수임. 예: 33.89 x 1.2 = 40.667 (40 + 2/3), 33.47 → 40.167 (40 + 1/6), 83.06 → 99.667, 76.94 → 92.333, 75.42 → 90.5.
- 따라서 **0.28 = 3-output 태스크의 output 1개**(1/3 태스크), **0.14 = 1/6 태스크**(1/2 + 2/3 같은 조합). 커뮤니티 분석(Boronic)과 CPMP 설명이 일치함 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/699858).
- 실무적 의미: 33.89와 33.06의 차이는 **태스크 1개(0.83점)** 수준이라 run-to-run 노이즈 안에 들어감. public 120개에서 태스크 1개 = 0.83%p이므로 0.3~1점 차이로 우열을 가리면 안 됨.
- [커뮤니티] 로컬 eval split은 태스크의 40.8%가 test input을 2~3개 가짐(train은 6.9%). 그리드 크기도 train보다 약 2.8~3.6배 큼 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/742277, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/734018). 숨겨진 셋의 분포는 알 수 없음.

---

## 3. Discussion 주요 내용

### 3.1 public vs private 상관관계 / 과적합
- [공식] public 120개와 private 120개는 서로 다른 태스크임 (699858). 호스트는 semi-private를 테스트 대상으로 쓰지 말라고 함 (694745).
- [탑팀] 2025년 NVARC: 로컬 eval pass@2 30.5%(pass@10 약 40%), public LB 27%, private 24%. "Our scorer algorithm missed 10%" (CPMP, 697859). public에서 private로 약 3점 떨어짐.
- [탑팀] 2025년 상위팀은 모두 public eval에서 public LB로 갈 때 크게 떨어졌고, 50% 하락도 드물지 않았음. TRM은 eval 22%, LB 10.83%. Qwen 기반은 eval 30%, LB 27%로 하락폭이 작았음 (CPMP, 697859).
- [커뮤니티] Jack Cole: 제출 한도를 줄인 이유는 과적합 우려였을 것이고, private 셋이 따로 있으니 과적합은 불가능하다는 의견 (694745). "private에 과적합하는 것은 불가능"하다는 뜻이지 public 과적합이 private에 전이되지 않는다는 뜻은 아님.
- [커뮤니티] "그럼 70점대도 120개 기준이니 memorizing 아닌가" 같은 추측 (740422). 근거는 없음.
- probing / LB 탐색을 대놓고 다룬 스레드는 찾지 못함. 하루 1회에 12시간 실행이라 probing 비용이 매우 큼.

### 3.2 LB33.89 공개 노트북 포크의 재현성 문제
- 원본: `koushikrudra/failed-in-aimo`(33.89, 566 votes, Gold), `mikelou1/arc-agi2-lb33-89-minimal-perfpatch`(현재 버전 32.22). 둘 다 NVARC(Qwen3-4B SFT + 태스크별 LoRA TTT + DFS 디코딩 + kgmon 선택) 기반.
- [커뮤니티] 바이트 단위로 같은 코드를 두 번 제출하니 29.86과 30.14가 나옴. 같은 노트북을 쓴 팀들이 날짜별로 **28.06~33.47**에 흩어짐. 원인 하나: `arc_solver.py`의 candidate-scoring augmentation seed가 `hash(bk) % 1024**2`인데, `PYTHONHASHSEED`를 설정하지 않아서 프로세스마다 str hash가 달라짐. 또 bf16 TTT 커널의 비결정성 때문에 후보 풀 자체가 달라짐. **공개 노트북 점수는 기대값이 아니라 여러 번 재제출한 결과 중 최대값**이라는 지적 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/742027).
- [커뮤니티] 포크했더니 29.72/29.31, 원저자는 32.22 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/742271). 같은 설정으로 돌렸는데 계속 28~29 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/733930). NVARC 원본 23.33 vs 재현본 32.22 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/743880, 답변 없음).
- [커뮤니티] Adarsh: "NVARC notebook is near-optimized … You need to get really lucky with seeds to get a 33 just with NVARC." nll threshold, selection 전략, TRM 앙상블, 2차 디코딩을 모두 시도했지만 LB에서 유의미한 개선이 없었음. batch-invariant ops와 adapter 캐시로 결정적 버전을 공개함 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/729743, 733930).
- [탑팀] "our code is not deterministic"이고, 작년에도 같은 코드를 재제출하니 점수가 달랐음 (685142). "simple tweaks to our last year winning solution won't be enough to win this time" (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/691081).
- 결론: 현재 LB 5위 이하(약 37점까지)는 거의 전부 NVARC 계열이고, 32~34 구간은 시드 노이즈로 순위가 정해지는 구간임.

### 3.3 런타임 실패 / 인프라
- [탑팀] 평소 10시간 이내로 끝나던 노트북이 한 번 타임아웃됨. 재제출하니 9h10에 끝남. 태스크별 time gating이 있는데도 2시간이 더 걸려서 인프라 문제로 봄 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/738313). 다른 참가자는 system error가 난 뒤 Kaggle이 일일 제출 기회를 돌려줬다고 함.
- [탑팀팀원 Darragh] 늘 몇 시간 걸리던 노트북이 15분 만에 0.00으로 끝남. 다음 날 같은 제출이 약 10시간 걸려 46%가 나옴 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/704408).
- [커뮤니티] 최근 이틀간 노트북이 몇 시간씩 큐에 대기하고 제출이 안 된다는 보고 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/743927). 마감이 다가오면 더 심해질 수 있음.
- [커뮤니티] GPU 4개에 태스크를 정적으로 나눴다가 한 GPU에 어려운 태스크가 몰려 전체 타임아웃이 남. 동적 큐와 큰 태스크 우선 배치를 권장 (697859, 729743). `/kaggle/working`에 wheel을 설치하면 출력 커밋 단계에서 ERROR가 남 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/732723).
- LB 히스토리 사이트 기준으로 Tufa Labs와 rabbithole의 제출은 **거의 매번 12h00~12h26**으로 표시됨(난독화 ±10분 포함). 12시간 한도에 붙여서 쓰고 있음. nvbanana는 9~10시간대.

### 3.4 하드웨어 관련 팁 (커뮤니티)
- 35B MoE(Qwen3.5-MoE 계열)를 vLLM fp8/W4A16으로 L4x4에 올린 실험에서 eval 0/64. 대형 LLM의 zero-shot 추론은 효과가 없었음 (732723).
- FlashInfer sampler를 컴파일할 수 없음 → `VLLM_USE_FLASHINFER_SAMPLER=0` 필요 (732723).
- 결정성 확보: `zlib.crc32`로 seed 만들기, `CUBLAS_WORKSPACE_CONFIG`, deterministic 알고리즘 사용. flash-attn backward는 여전히 비결정적이고, 8k 토큰에서는 SDPA math가 L4 메모리에 들어가지 않음 (742027).

### 3.5 상위팀 공개 발언
- **nvbanana (3위, 75.42)** = CPMP(Jean-François Puget) + Darragh 등 2025 우승팀 NVARC 계열(CPMP의 "3rd in this Competition" 배지와 Darragh 배지로 확인). 방법은 대회가 끝날 때까지 비공개라고 함 ("you'll have to wait till competition end", https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/741315). 모델 이름도 공개하지 않음 (704408). "We have some interesting things in the pipe" (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/694676). 2위팀이 70%를 넘었을 때 "Our lead is quite small now"라고 했고, 이 시점 이후 Tufa Labs에 역전당함 (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/724647). 다른 재귀 모델도 시도했지만 TRM보다 나은 결과는 없었다고 함 (729743).
- **rabbithole (2위, 76.94)**, **Tufa Labs (1위, 83.06)**: Kaggle 포럼에서 방법에 대한 공개 발언을 찾지 못함. LB 히스토리상 Tufa Labs는 8월 초 32.5(NVARC 수준)에서 시작해 40.8 → 43.8 → 45 → 48.6 → 49.9 → 58.3 → 71.1 → 73.1 → 73.5를 거쳐 약 5일 전 83.06이 됨(제출 37회). rabbithole은 15 → 20 → 34 → 38 → 40.7 → 44.9 → 47.8 → 50.4 → 70.4 → 73.3 → 76.7 → 76.94(제출 101회).
- [커뮤니티 추측] 상위팀의 급상승은 LLM-agent / reasoning LLM 기반일 것이라는 추측 (Van-Phuc Huynh, 742796). 다만 다른 참가자의 agent 제출은 13.75%에 그쳤다는 반례가 있음. 근거 없는 추측으로 분류함.
- 4위 Yi-Chia Chen은 제출 11회로 39 → 55.14. 역시 공개 발언 없음.

---

## 4. Code 탭 상위 공개 노트북

| 노트북 | 표시 점수 | 방법 |
|---|---|---|
| koushikrudra/failed-in-aimo ("LB33.89") | 33.89 (566 votes, Gold) | NVARC 포크: Qwen3-4B(grids15_sft139) + LoRA r=256 TTT + DFS p>=0.2 + kgmon 선택 |
| mikelou1/arc-agi2-lb33-89-minimal-perfpatch | 32.22 (현재 버전) | 위 노트북의 성능 패치판. 포크가 가장 많음 |
| rokaiyasomapti / madarshbb reproduce-nvarc-2025-results | 32.22 | NVARC 재현, batch-invariant 결정적 버전 |
| luxluxshan/arc2-nvarc-v1, finalsunflower/arc-agi-2-nvarc-plus | 32.22 / 31.39 | NVARC+ 변형 |
| christopherdaleman/arc-2026-nvarc-trm-evidence-cost-v1 | 31.11 | NVARC + TRM 후보 결합, cost ordering |
| sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue (NVARC 원저자) | 30.56 | 2025 우승 원본 (4xL4) |
| jonathanchan/arc26-2025-winning-solution-v1 | (CPMP가 언급) | 2025 우승 노트북 그대로 |
| nihilisticneuralnet/baseline-nvarc-...-t4x2 | 11.67 / 19.17 | T4x2 시절 축소판 |
| mirzamilanfarabi/arc2-qwen3-unsloth-... | 10.42 | Qwen3 Unsloth TTT |
| softkleenex/arc-2025-compressarc-method | 1.67 | CompressARC (P100) |
| dominic789654/arc-agi-2-program-synthesis-harness | - | GPT로 오프라인에서 미리 만든 transform 함수 67개를 lookup (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/733501) |

- 상위 공개 노트북은 **전부 NVARC 계열이고 30~34 구간**에 몰려 있음. 공개 코드와 상위 3팀(75~83) 사이 격차가 40점 이상임.
- Code 탭의 "Score"는 노트북 최신 버전의 점수임. 같은 노트북도 버전이나 실행에 따라 29~34로 흔들림.
- [탑팀] TRM 단독 LB 약 10점. Qwen이 못 푸는 태스크 중 TRM이 푸는 것은 1~2개 정도 (729743).

---

## 5. public/private 상관관계 판단

1. **구조**: public(semi-private 120)과 private(private 120)는 같은 240개 숨겨진 셋을 반으로 나눈 서로 다른 태스크 집합임 [공식]. 한 번의 실행에서 둘 다 채점되므로 재실행이나 타임아웃 위험의 차이는 없음. 차이는 **태스크 표본 차이**와 **두 세트의 난이도 차이**에서만 생김.
2. **과거 데이터**: 2025년 우승 솔루션은 public 27에서 private 24로 약 3점(상대 -11%) 떨어짐 [탑팀]. ARC Prize 쪽에서도 private가 semi-private보다 약간 어렵다는 경향이 알려져 있음(CPMP 수치로 뒷받침됨). 올해 테스트 데이터가 작년과 같다는 CPMP 발언을 따르면 비슷한 수준의 하락을 예상할 수 있음.
3. **표본 노이즈**: 120개 태스크에서 정답률 p=0.33이면 이항 표준오차가 약 ±4.3%p, p=0.8이면 약 ±3.7%p. 여기에 run-to-run 비결정성 ±2~3점이 더해짐. 따라서 **public 1~3점 차이는 private 순위를 거의 예측하지 못함**. 반대로 NVARC 33과 상위 75~83처럼 **수십 점 차이는 확실히 유지**될 것임.
4. **public 과적합 가능성**: 하루 1회, 12시간 실행이라 public을 대량으로 probing하기는 사실상 불가능함. 다만 공개 노트북 점수는 여러 번 재제출한 결과의 최대값(selection bias)이라 private에서 평균 쪽으로 돌아갈(regression to mean) 가능성이 높음 [커뮤니티, 742027]. 그리고 public 점수를 보고 하이퍼파라미터나 선택 규칙을 조정할수록 그만큼 부풀려짐.
5. **종합**: 상관은 "대략 강함"(방법 수준 차이는 보존됨). 하지만 같은 방법의 변형 사이에서는 상관이 "약함"(노이즈가 지배함). 예상 private 점수 ≈ public - 2~4점. NVARC 포크 구간(32~34)은 private에서 약 27~31로 섞일 가능성이 큼 (추정).

## 6. 제출 전략 시사점

1. **LB33.89 포크로는 메달권 밖**: 상위 3팀이 75~83이고, 공개 NVARC 계열은 32~34에 수백 팀이 몰려 있음. 이 구간의 순위는 시드 운으로 정해짐. 의미 있는 순위 상승은 방법을 바꿔야만 가능함(더 많은 합성 데이터로 SFT, 더 큰 base 모델, 더 나은 selection/scorer 등).
2. **결정성부터 확보**: `PYTHONHASHSEED`/`zlib.crc32` 기반 seed 설정, 태스크별 재시드, 가능한 범위의 deterministic 커널 적용. 로컬 eval(120)에서 개선이 노이즈보다 큰지 확인한 뒤에만 제출함. 하루 1회 제출을 A/B 테스트에 쓰지 말 것.
3. **로컬 검증 기준**: public eval 120개로 측정함(train은 분포가 쉬움). 단 eval과 public LB 사이에도 하락이 있음(Qwen 계열 약 -10% 상대, TRM 약 -50%).
4. **최종 2개 선택**: (a) public 최고점이 아니라 **로컬 eval과 public이 함께 높고 여러 번 재현된 안정적인 제출** 1개, (b) 방법이 다른(다양성 있는) 강한 제출 1개. 최종 점수는 둘 중 private 최고점이 반영됨. public 0.3~1점 차이(태스크 1개 이하)로 고르지 말 것.
5. **런타임 여유**: 인프라 문제로 +2시간 타임아웃이 난 사례가 있음. 태스크별 time gating, 동적 큐, 큰 태스크 우선 배치를 쓰고, 목표 실행 시간은 10~10.5시간 이내로 잡음. 타임아웃이 나도 빈 attempt가 아니라 fallback 예측을 쓰도록 부분 저장 구조를 만들 것.
6. **마감 일정**: 큐 지연이 이미 보고되고 있음(743927). 10-26 팀 병합/Entry 마감 전에 rules 동의와 팀 구성을 끝내고, 11-02 직전 며칠은 제출이 막힐 수 있다고 보고 최종 후보를 1주 전까지 확정함. 제출 전에 반드시 Save로 검증함(가속기 자동 해제 사고 예방).
7. **상금 자격 리스크**: base 모델의 OSAID 충족 여부(Qwen 등), NVARC 체크포인트 라이선스와 tunable 설정, CC BY 4.0 vs MIT-0 문제는 아직 공식 답변이 없음. 상금을 노린다면 사용하는 모델과 데이터의 라이선스를 기록해 두고 공개할 수 있게 준비할 것.
