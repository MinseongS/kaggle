# 01. 과거 우승 솔루션과 LB 신뢰도 조사 (ARC Prize 2024 / 2025 → 2026 ARC-AGI-2)

조사일: 2026-09-28 (마감 2026-11-02까지 약 35일)
표기 규칙: **[사실]**은 출처에서 직접 확인한 내용, **[추론]**은 내 해석이나 추정. 모든 사실에는 출처 URL을 달았다.

---

## 0. 한눈에 보는 결론

1. **[사실]** 2026 ARC-AGI-2 대회는 "ARC Prize 2025의 relaunch"다. 12h 런타임, L4x4 GPU, 240개 hidden task, public LB가 약 50%라는 조건이 2025와 같다. 대회 3위(nvbanana)인 CPMP(NVARC 2025 우승자)도 "올해 환경은 작년과 완전히 같다"고 했다.
   - https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview
   - https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data
   - https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/691081
2. **[추론, 강함]** test set도 2025와 같을 가능성이 높다. 근거는 두 가지다. 2025 NVARC 원본 노트북을 다시 돌리면 23.33, 재현 노트북은 32.22가 나오는데 2025 public 27.64와 같은 범위다(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/743880). 그리고 두 대회 모두 "240 unseen tasks"다. 그렇다면 **private set = 2025 private set**이고, 2025 NVARC의 private 점수는 **24.03%**였다.
3. **[사실]** 현재(2026-09-28) public LB는 다음과 같다.
   - 상위권: 1위 Tufa Labs 83.06, 2위 rabbithole 76.94, 3위 nvbanana(CPMP 팀) 75.42, 4위 55.14, 5~13위 약 34~37.5
   - 그 아래: **약 32.2~33.5% 구간에 100팀 이상이 몰려 있다**(99위 32.22). 대부분 NVARC 2025 기반 공개 노트북("lb33-89-minimal-perfpatch" 계열)을 fork한 팀이다.
   - 총 2,227팀
   - https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard
   - https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/code?sortBy=scoreDescending
4. **[사실]** 2025에는 public에서 private으로 넘어가며 **심한 shake-up**이 있었다.
   - 상위 3팀은 점수가 13~24% 줄었다.
   - 중위권은 public 7.08 → private 약 4.2처럼 약 40% 줄었다.
   - public 344위였던 팀(Lonnie)이 private 4위가 됐다. 이 팀은 "random seed만 바꿨다"고 밝혔다.
   - https://www.kaggle.com/competitions/arc-prize-2025/leaderboard
   - https://www.kaggle.com/competitions/arc-prize-2025/writeups/arc-prize-2025-competition-writeup-5th-place
5. **[추론]** 은메달은 **private 기준 상위 5%(2,227팀이면 약 111위 이내)**다. 32~33% plateau에 있는 팀들의 private 순위는 사실상 **추첨**이다. 은메달을 확실히 하려면 공개 노트북보다 **기대값(expected value)이 뚜렷이 높은** 파이프라인이 필요하다. rerun의 최댓값만 높은 파이프라인으로는 부족하다.

---

## 1. ARC Prize 2025 상위 솔루션

### 1.1 최종 순위 (Kaggle, public ↔ private)

**[사실]** 출처: Kaggle 2025 LB(Public/Private 탭 직접 확인) https://www.kaggle.com/competitions/arc-prize-2025/leaderboard , 기술 보고서 https://arxiv.org/abs/2601.10904

| Private 순위 | 팀 | Public | Private | Public→Private 변화 |
|---|---|---|---|---|
| 1 | NVARC (Ivan Sorokin, Jean-François Puget/CPMP) | 27.64 | **24.03** | -13% |
| 2 | the ARChitects | 21.67 | 16.53 | -24% |
| 3 | MindsAI & Tufa Labs | 15.42 | 12.64 | -18% |
| 4 | Lonnie | (public 344위) | 6.67 | +339계단 |
| 5 | Guillermo Barbadillo | 11.94 | 6.53 | -45% |
| 6 | ippeiogawa | 10.28 | 6.25 | -39% |
| 8 | rxe | 10.42 | 6.11 | -41% |
| (public 4위) | The North Stars | 12.08 | private 상위 49위 밖(<4.17) | 붕괴 |

- **[사실]** 참가 규모: 1,455팀, 15,154 submission. 우승 비용은 약 $0.20/task(https://arxiv.org/html/2601.10904v1).
- **[사실]** private 49위의 점수는 4.17이었다. 이 구간에서 public은 약 7.08이었다(같은 LB).
- **[사실]** 2025 LB 문구는 "The private leaderboard is calculated with approximately 50% of the test data"다. 공식 설명에 따르면 Semi-Private 120 task(API 노출로 유출 위험이 있음)와 Private 120 task(완전 비공개)로 나뉜다(https://arxiv.org/html/2601.10904v1).
- **[추론]** Kaggle public LB는 Semi-Private 120 task, private LB는 Private 120 task에 해당하는 것으로 보인다. 공식 문서가 두 LB와 두 set의 대응을 명시하지는 않는다.

### 1.2 1위 NVARC (24.03% private)

출처: Kaggle writeup https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc , 코드 https://github.com/1ytic/NVARC , NVIDIA 블로그 https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/ , Trelis 해설 https://trelis.substack.com/p/nvarc-2025-arc-prize-winners

**[사실]**
- **구성**: (1) 다단계 합성 데이터(SDG), (2) 개선한 ARChitects 방식(Qwen3 4B), (3) TRM 앙상블(실험적)
- **합성 데이터 파이프라인** (주로 gpt-oss-120b와 NeMo-Skills 사용, 8xH100 한 노드에서 15k tok/s)
  1. H-ARC(인간 1,700명 이상이 쓴 자연어 풀이 설명)와 BARC 설명 160개를 모아 ARC-AGI-2 training 716개 퍼즐의 설명을 확보했다.
  2. 두 설명을 섞어 더 복잡한 퍼즐 설명 266,593개를 만들었다("concept mixing").
  3. 입력 그리드 생성 Python 코드를 만들고 unit test로 검증했다. 결과는 126,901개다.
  4. 출력 변환 코드를 여러 번 생성한 뒤, 출력이 서로 일관된 것만 남겼다. 최종 103,253개 퍼즐이다.
- **학습 데이터**: 총 3.2M augmented sample, 샘플당 최대 7 pair. 구성은 MINI-ARC, ConceptARC, RE-ARC, ARC-AGI-2(각 256 aug)와 NVARC 합성 약 10.3만 개(24~32 aug)다. "NVARC full" subset에는 **ARC-AGI-2 evaluation 퍼즐도 포함**됐다.
- **모델**: Qwen3 4B. 토큰은 16개만 쓴다(숫자 10개, 줄바꿈, user/assistant, 특수 토큰). NeMo RL(Megatron)로 **full fine-tuning을 4노드 × 8xH100 × 27시간** 했다.
- **TTT**: 퍼즐마다 독립적으로 LoRA를 붙인다(r=256, alpha=32). bf16을 쓰고 4bit 양자화와 gradient checkpointing은 뺐다. Unsloth와 FlashAttention2를 사용했다.
- **디코딩**: batch DFS를 쓴다. batch-invariant 버전은 더 정확했지만 Kaggle에서 17% 느려서 최종 제출에는 쓰지 않았다(nondeterminism의 원인).
- **후보 scoring**: 모든 후보에 **같은 8개 augmentation**을 적용해 비교 가능성을 높였다. 대회 후에는 "DFS에서 몇 번 발견됐는지" 빈도와 augmentation별 log-prob의 geometric mean을 결합한 방식이 더 낫다고 보고했다.
- **로컬 검증**: augmented eval 120 task에 대한 validation loss가 public LB와 잘 상관됐다고 보고했다.
- **TRM 파트**: 사전학습 후 Kaggle에서 test data로 fine-tune했다(약 2h, H cycles 4, halt max steps 10, 2000 epoch). TRM 단독 점수는 7.5%, 대회 후 4k epoch로 10.0%까지 올랐다. Qwen3 앙상블에서는 2B 모델이 21.53 → 22.50으로 올랐지만 4B 모델에서는 27.22 → 27.22로 효과가 없었다.
- **하드웨어/런타임**: Kaggle 4xL4, 12h 제한. 2026 재제출 평균은 약 10h였다(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/738313).
- **이후 결과**: ARC-AGI-1 eval에서 95% 이상. 다만 AGI-1 eval 대부분이 AGI-2 training에 들어 있어서 과대평가라고 본인들이 밝혔다(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/694698).
- **공개 자산**: 모델(`sorokin/qwen3_4b_grids15_sft139`), 합성 퍼즐 데이터셋, 노트북 `arc2_qwen3_unsloth_flash_lora_batch4_queue`, TRM 노트북

### 1.3 2위 the ARChitects (16.53% private)

출처: https://www.kaggle.com/competitions/arc-prize-2025/writeups/the-architects-solution-2025 , 기술 보고서 https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/

**[사실]**
- **최종 모델**: masked-diffusion LLM인 **LLaDA-8B**
  - 2D RoPE("Golden Gate RoPE")를 적용했다.
  - soft-masking과 recursive latent refinement로 출력을 반복 수정한다. 51 step을 2 round 돌려 총 102 step이다.
  - 가장 많이 방문한 후보를 선택한다.
  - 출력 크기는 별도 LLaDA shape predictor(eval 정확도 약 85%)가 예측한다.
- **학습**: ReARC, ARC-GEN-100K, ARC1/2, ARC-Heavy, ConceptARC 사용. 사전학습은 175k step, batch 8, 8xH100에서 39h. TTT는 task당 128 step, batch 1, LoRA r=32(L4).
- **컴퓨트 스폰서(Lambda)**: GH200 1~3대를 상시 사용했고, 마지막 2주에는 H100 16장을 썼다.
- **점수**
  - 로컬 기대치는 약 26%였다(shape를 알 때 30.5% × shape 정확도 85%).
  - 실제 점수는 **public 21.67 / private 16.53**이었다.
  - **다른 하이퍼파라미터로 선택한 제출은 public 19.17 / private 19.17**이었다. public이 낮아 최종 선택하지 않았는데, 선택했다면 private 점수가 더 높았다.
- **전반기 AR 모델**: Mistral-NeMo-Minitron-8B, 퍼즐별 TTT, speculative decoding DFS, 확률 cutoff 17→7%, scoring augmentation 8→32. public 16.94까지 올랐다.
- **안 된 것**: 합성 데이터(새롭다고 할 만한 것이 1~2%뿐), Canon layer, H-Net, 중간 reasoning token

### 1.4 3위 MindsAI & Tufa Labs (12.64% private, public 15.42)

출처: https://www.kaggle.com/competitions/arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution , https://github.com/jcole75/arc_2025_mindsai

**[사실]**
- **모델**: CodeT5-Large를 pruning한 **660M** encoder-decoder(encoder 24층, decoder 16층). TPU에서 1억 개 이상의 예제(ARC형 약 7천만)로 수개월 학습했다.
- **TTT**: permutation 기반 labeling으로 약 45k step
- **AIRV**(Augment-Inference-Reverse-Vote): task당 10k augmented inference
- **효과**: TTT와 AIRV는 거의 완전히 가산적이다(zero-shot 대비 8~12배).
- **추가 기법**: mixup/combine augmentation, BPE dropout, 서로 다른 seed의 checkpoint 2개 self-ensemble. 같은 compute에서 샘플 수를 2배 늘리는 것보다 self-ensemble이 +6.2% 더 좋았다.
- **추론**: 4xL4에서 약 11h
- **간소화 버전**: 77M 모델, P100에서 10~60분. 전체 이득의 90~95%를 얻는다.
- **팀 소감**: "ARC-AGI-2는 현재 TTT/AIRV 패러다임에 부분적으로 adversarial하다."
- **데이터 공개**: https://huggingface.co/datasets/mindware/arc-agi-mega

### 1.5 4~6위: 2024 ARChitects 베이스라인을 살짝 바꾼 것

**[사실]**
- **4위 Lonnie** (private 6.67, public 344위): 2024 ARChitects 기반 공개 노트북을 fork했다. 핵심 변경은 **random seed를 19920627로 바꾼 것**뿐이다. 본인도 "120 task split에서는 1~2 task로 순위가 크게 흔들린다"고 적었다. (https://www.kaggle.com/competitions/arc-prize-2025/writeups/arc-prize-2025-competition-writeup-5th-place)
- **5위 Barbadillo** (6.53): 주 연구는 search-and-learn 방식의 program synthesis였는데 private task를 하나도 풀지 못했다. 최종 점수는 2024 transduction+TTT 방식을 조금 바꿔서 얻었다. (https://www.kaggle.com/competitions/arc-prize-2025/writeups/exploring-the-combination-of-search-and-learn-for , https://ironbar.github.io/arc25/05_Solution_Summary/)
- **6위 ippeiogawa** (public 10.28 → private 6.25): 2024 ARChitects를 조금 수정했다(TTT 24 epoch, DFS top_k=4). "점수가 불안정해서 최고 LB 제출 대신 일관적인 제출 2개를 골랐다." (https://www.kaggle.com/competitions/arc-prize-2025/writeups/lb10-00-with-small-change-to-2024architects)

### 1.6 Paper Award (2025)

출처: https://arcprize.org/blog/arc-prize-2025-results-analysis , https://arcprize.org/competitions/2025

| 순위 | 논문 | 핵심 | 성능 |
|---|---|---|---|
| 1 | **TRM** (Jolicoeur-Martineau) | 2층, 약 7M 파라미터. latent state와 답을 최대 16 step 재귀적으로 개선하고 deep supervision과 halting을 쓴다. | ARC-AGI-1 45%, ARC-AGI-2 약 8%. AGI-2 학습에 4xH100으로 약 3일 (https://arxiv.org/abs/2510.04871 , https://github.com/SamsungSAILMontreal/TinyRecursiveModels) |
| 2 | **SOAR** (Pourcel, Colas, Oudeyer) | LLM 진화적 program search와, 그 search trace를 hindsight relabel해 fine-tune하는 과정을 반복한다. | ARC-AGI-1 test 52% (https://arxiv.org/abs/2507.14172) |
| 3 | **CompressARC** (Liao & Gu) | MDL 기반. 사전학습 없이 퍼즐마다 76K 파라미터 신경망을 처음부터 학습한다(2000 step). | AGI-1 eval 20%, AGI-2 약 4%. RTX 4070에서 퍼즐당 약 20분 (https://arxiv.org/abs/2512.06104) |

- **[사실]** Honorable mention에 Franzen 외(ARChitects), Cole & Osman, Sorokin & Puget, Barbadillo 등이 있다.
- **[사실]** 2026 공개 노트북 중 CompressARC 방식은 P100에서 LB 1.67이다(https://www.kaggle.com/code/baidalinadilzhan/prev-year-s-compressarc-method-p100-gpu-lb-1-67). 그대로는 경쟁력이 없다.
- **[사실]** 2025의 핵심 트렌드는 "refinement loop"(피드백 신호를 받는 per-task 반복 최적화)다(https://arxiv.org/abs/2601.10904).

---

## 2. ARC Prize 2024 (아직 유효한 것만)

출처: https://arxiv.org/abs/2412.04604 , Kaggle LB https://www.kaggle.com/competitions/arc-prize-2024/leaderboard

**[사실]**
- **하드웨어**: P100 1장, 12h, 인터넷 없음
- **LB 구조**: "The private leaderboard is calculated over the same rows as the public leaderboard". 즉 **public과 private이 같은 100 task**였다. 그래서 shake-up은 없었지만, 주최 측은 "약 1만 번의 private score가 참가자에게 노출되어 overfitting 위험이 있다"고 인정했다. 2025부터 set이 분리된 이유다.
- **순위**: ARChitects 53.5, Barbadillo 40.0, alijs 40.0, William Wu 37.0, PoohAI 37.0. MindsAI는 55.5였지만 코드를 공개하지 않아 수상 대상에서 빠졌다.
- **ARChitects 2024** (https://arxiv.org/abs/2505.07859 , ICML 2025)
  - Mistral-NeMo-Minitron-8B, LoRA TTT, 줄인 vocabulary
  - DFS로 확률이 높은 후보를 여러 개 생성한다.
  - 여러 augmentation(D8 대칭과 color permutation)에서 계산한 log-prob를 합쳐 후보를 고르는 "Product of Experts" scoring을 쓴다.
  - AGI-1 public eval 71.6%
  - **2025/2026 대부분 솔루션의 뼈대**다.
- **Omni-ARC** (Barbadillo, https://ironbar.github.io/arc24/05_Solution_Summary/): Qwen2.5-0.5B로 여러 과제(출력 예측, 입력 분포 학습 등)를 사전학습하고 TTT를 한 뒤, program synthesis와 앙상블했다. "transduction이나 induction 단독으로는 약 40%가 한계라서 앙상블이 필요하다."
- **여전히 유효한 교훈**: TTT, augmentation 기반 scoring과 voting, transduction+induction 앙상블. 셋 다 2025 우승 솔루션에 그대로 남아 있다.

---

## 3. 최근 관련 대회에서 옮겨올 교훈 (간단히)

- **NeurIPS 2025 Google Code Golf** (ARC-AGI-1 training 400 task, ARC-GEN 추가 테스트, https://www.kaggle.com/competitions/google-code-golf-2025 , https://www.luke-g.com/the-2025-google-code-golf-championship-part-1/)
  - **[사실]** AGI-1 training 400 task 전부에 대해 사람이 쓴 정답 Python 프로그램이 공개됐다.
  - **[사실]** ARC-GEN(https://arxiv.org/abs/2511.00162)은 AGI-1 task의 procedural generator다.
  - **[추론]** 두 자산은 합성 데이터 seed나 program-synthesis 학습 데이터로 쓸 수 있다. 다만 AGI-2와는 난이도 차이가 크다.
- **NVIDIA Nemotron Reasoning Challenge** (2026, https://developer.nvidia.com/blog/lessons-from-the-leaderboard-what-5000-kagglers-taught-us-about-improving-ai-reasoning/)
  - **[사실]** 답만이 아니라 추론 trace까지 검증해 합성 데이터를 정제했다.
  - **[사실]** token budget에 맞춰 trace를 압축했다.
  - **[사실]** LoRA(r 32 이하)를 고품질 합성 데이터로 학습했다.
  - **[사실]** 노이즈를 걸러내려고 반복 실행 안정성을 추적했다.
- **AIMO3**: **[사실]** 110문제, 올림피아드 수준. 세부 우승 해법은 이번 조사에서 확인하지 못했다(https://aimoprize.com/updates/2025-11-19-third-progress-prize-launched). **[추론]** 이 대회에서 옮겨올 수 있는 교훈은 "소수 문제에서 rerun 분산이 크니 제출 선택을 기대값으로 하라" 정도다.

---

## 4. Public vs Private LB 상관관계 (가장 중요)

### 4.1 사실관계
- **2024**: public과 private이 같은 set이었다. shake-up은 없고 overfitting만 있었다. **[사실]** 위 2절 출처
- **2025**: 240 task를 50:50으로 나눴다(Semi-Private 120과 Private 120으로 보임). **[사실]** 1.1절 표 참고
  - **상위 3팀**: private가 public의 **0.76~0.87배**
  - **중위권**(public 7~12): private가 public의 **약 0.55~0.6배**
  - **public 4위**(12.08)는 private 상위 49위 밖으로 떨어졌다.
  - **public 344위**가 private 4위가 됐다.
- **ARChitects의 선택 실패**: public 21.67 제출은 private 16.53, public 19.17 제출은 private 19.17이었다. **[사실]** https://www.kaggle.com/competitions/arc-prize-2025/writeups/the-architects-solution-2025
- **2026 rerun 분산**
  - **[사실]** 바이트 단위로 같은 노트북이 29.86과 30.14를 냈다. 같은 노트북이 다른 계정에서는 33.89였다.
  - **[사실]** 공개 tracker 기준으로 같은 노트북 팀들이 28.06~33.47에 분포한다. 원인은 `hash(str)` seed(PYTHONHASHSEED 미설정)와 bf16/flash-attn의 nondeterminism이다. https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/742027
  - **[사실]** 복사한 노트북은 29.72/29.31, 원저자 노트북은 32.22였다. https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/742271
- **Private 난이도**: **[사실]** ARC-AGI-2는 인간 테스트로 난이도를 calibration했다. 공식적으로 public eval, semi-private, private가 비슷한 난이도로 맞춰졌다고 한다(https://arxiv.org/abs/2505.11831). 그럼에도 2025에는 모든 팀이 private에서 체계적으로 점수가 떨어졌다.

### 4.2 해석 [추론]
- **분산이 크다.** 120 task에서 1 task는 약 0.83%p다. 점수가 30% 근처일 때 binomial 표준편차는 약 4.2%p다. 여기에 rerun 노이즈(±2%p)가 더해진다. 그래서 **5%p 이내의 public 차이는 대부분 노이즈**다.
- **체계적으로 떨어진다.** 이유는 세 가지로 본다. (a) public 점수가 여러 번 제출한 것 중 최댓값이라는 selection bias가 가장 크다. (b) public 기준으로 하이퍼파라미터를 조정한 효과가 있다. (c) semi-private set이 API에 노출돼 일부 유출됐을 수 있다. 따라서 **private ≈ public × 0.75~0.85** 정도를 예상한다.
- **2026 private 예상.** test set이 2025와 같다면, 32~33% plateau에 있는 NVARC 계열의 private 기대값은 약 24~26%다(2025 NVARC private 24.03 참고). 100팀 이상이 같은 분포 안에서 추첨하게 된다.
- **은메달 cut 예상.** 5~13위가 34~37.5다. 상위 약 111위까지 은메달이므로, private에서 **NVARC 기대값보다 약 +3~5%p 높은 파이프라인**이면 은메달이 꽤 안정적이라고 본다. 동등한 수준이면 은메달 확률은 약 50% 이하, 동메달(약 223위) 확률은 높다.
- **private 85% 조건.** CPMP는 "85%는 private 기준"이라고 언급했다(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/742796). 상위 3팀(75~83)도 private에서 떨어질 수 있다.

---

## 5. ARC Prize 2026 ARC-AGI-2 규칙 (확인된 것)

출처: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview , https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules , https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data , https://arcprize.org/competitions/2026/arc-agi-2

**[사실]**
- **일정**: 시작 2026-03-25. Entry와 Team merger 마감 **2026-10-26**. 최종 제출 **2026-11-02 23:59 UTC**. 발표 2026-12-04.
- **하드웨어**: 노트북 제출만 허용. CPU/GPU 모두 **≤12h**, 인터넷 불가. **L4x4(총 96GB VRAM)** 풀을 쓸 수 있고, quota는 T4x2/P100의 2배 속도로 소모된다(주간 GPU quota 약 30h, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/732723). 런타임은 obfuscate되어 최대 10분의 편차가 있다.
- **제출**: **하루 1회**. 최종 선택은 2개. 팀은 최대 5명.
- **데이터**: rerun 때 **240 unseen task**로 채점한다. 대부분 test input이 1개다. public LB는 약 50%, 나머지 50%가 최종 순위다. 로컬에 있는 test 파일은 evaluation set으로 만든 placeholder다.
- **평가**: test output마다 2 attempt 중 하나가 정확히 일치하면 점수를 받는다.
- **외부 데이터와 사전학습 모델**: 공개되어 있고 무료거나 합리적인 비용이면 허용된다.
- **상금**: 총 $700k. Progress Prize 1~8위($75k~$15k), Grand Prize $275k(writeup 평가), 85% Bonus $150k.
- **오픈소스 의무**: 수상하려면 CC BY 4.0 라이선스로 코드를 공개하고, OSI Open Source AI 정의에 맞는 오픈 모델과 가중치를 써야 한다.
- **관찰된 이슈**
  - 마감 직전 큐 대기가 수 시간 걸린다(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/743927).
  - 같은 노트북이 원인 불명으로 timeout된 사례가 있다(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/738313).
  - `/kaggle/working`에 wheel을 풀면 output commit 단계에서 ERROR가 난다(discussion 732723).

**[사실] 2026 LB에서 관찰한 것**
- **CPMP 팀(nvbanana)**: 평균 약 10h로 75.42. CPMP는 "2위 팀은 작년 솔루션의 knob 조정을 넘어섰다"고 했다(https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/724647).
- **Tufa Labs 점수 추이**: 8월 초 32 → 9월 43~49 → 9월 중순 58~73 → 9월 22일경 83. 모든 제출이 약 12h를 꽉 채웠다(https://arc3.huikang.dev/leaderboard/?comp=arc2).
- **[추론]** 상위 3팀의 방법은 공개되지 않았다. 더 큰 합성 데이터와 새 사전학습을 썼거나, LLM 기반 program synthesis나 refinement를 섞었을 것으로 추정한다.
- **LLM 에이전트 시도**: 공개된 결과로는 Qwen 계열 LLM 에이전트 단독이 LB 13.75에 그쳤다(discussion 742796). 35B MoE reasoning 모델(3B active)은 zero-shot으로 eval 0/64였다(discussion 732723). **[추론]** 순수 zero-shot LLM으로는 Kaggle compute에서 경쟁력이 없다.

---

## 6. 은메달 전략에 주는 시사점 (우선순위순)

1. **[최우선] 결정론(determinism)을 확보하고 평가 체계를 고친다.**
   - NVARC 계열 코드에서 `hash(str)` seed를 `zlib.crc32`로 바꾸고, `PYTHONHASHSEED`를 설정하고, task별로 seed를 고정한다(discussion 742027).
   - 이렇게 해야 개선 효과를 노이즈와 구분할 수 있다. 하루 1회 제출로 남은 제출은 약 35회뿐이라 모든 제출이 A/B 실험이어야 한다.
   - 로컬 held-out을 만든다. NVARC "full" 모델은 AGI-2 evaluation 120개로도 학습됐으니, eval로 검증하려면 **eval을 학습에서 뺀 NVARC training subset 모델이나 checkpoint가 필요**하다. 가능하면 eval의 augmented 버전에서 loss와 pass@2를 측정한다.
2. **[높음] 베이스라인은 NVARC 2025(Qwen3-4B, TTT, DFS, aug-scoring)로 하고, 기대값을 올리는 저비용 개선부터 한다.**
   - 대회 후 공개된 **re-scoring**(DFS 발견 빈도와 augmentation log-prob geometric mean 결합)을 적용한다.
   - scoring augmentation 수와 공통 augmentation set을 튜닝한다.
   - 약 10h인 런타임을 12h 한도(안전 마진 약 40분)까지 늘려 TTT step이나 DFS 후보를 늘린다.
   - 시간이 초과된 task는 fallback으로 처리한다(time gating).
   - MindsAI식 **self-ensemble**(seed나 checkpoint 2개의 후보 합집합에 공동 scoring)을 넣는다. 같은 compute에서 샘플을 2배로 늘리는 것보다 좋았다는 보고가 있다.
3. **[높음] 제출은 max가 아니라 기대값으로 고른다.**
   - 같은 설정을 2~3회 rerun해 평균을 본다.
   - 최종 2개 중 하나는 "가장 안정적인 기대값 최고" 설정, 다른 하나는 이와 성격이 다른(diversified) 강한 설정으로 한다.
   - ARChitects처럼 public 최고점을 골라 private에서 손해 본 사례(21.67→16.53, 대안은 19.17→19.17)를 피한다.
4. **[중간, 예산 허용 시] 합성 데이터를 추가해 NVARC 모델을 continual SFT한다.**
   - NVARC 합성 퍼즐 10만 개와 생성 스크립트가 공개돼 있다. 여기에 H-ARC/BARC 기반 concept mixing을 더 돌리거나, AGI-2 스타일의 어려운 개념(global restructuring, 다단계 규칙)을 중심으로 추가 생성한다.
   - 4B full FT는 약 860 H100-h가 들어 비현실적이다. LoRA나 짧은 continual FT로 대응한다. 1~2장짜리 H100을 수십 시간 빌리는 정도의 modest 예산을 가정한다.
   - 우승팀이 밝힌 대로 합성 데이터를 늘리는 것이 가장 확실한 개선 방향이었다.
5. **[중간] 보완 solver를 앙상블 후보 풀에 추가한다.**
   - TRM(pretrain 후 test-time FT, 약 2~4h)을 후보 생성기로 넣는다. 2B 모델에서는 +1점이었지만 4B에서는 효과가 없었으니 저순위로 둔다.
   - 싼 DSL이나 program search(대칭, 색 매핑 등 규칙 기반)는 CPU에서 병렬로 돌린다. 검증된 프로그램이 나오면 attempt_1을 우선 채운다. induction과 transduction은 서로 다른 task를 푼다는 2024 교훈을 따른다.
6. **[낮음] 피할 것**
   - 순수 zero-shot LLM이나 에이전트(공개 결과 약 14% 이하)
   - CompressARC 단독(약 1.7)
   - 공개 노트북을 fork해 seed만 돌리며 public max를 쫓는 것(private에서는 추첨)
   - 큰 모델(8B 이상 diffusion 등)을 새로 사전학습하는 것(ARChitects급 컴퓨트 필요)
7. **[운영]**
   - 마감 주에는 큐 지연과 timeout 사례가 있으니, 최종 후보는 **10월 말 이전에 확정·검증**한다.
   - Entry와 Team merger 마감은 10/26이다.
   - GPU quota(L4는 2배 소모)는 로컬 실험보다 제출 검증에 우선 배분한다.

**핵심 한 줄 [추론]**: plateau(약 32%)에 있는 100팀 이상 사이에서 private 순위는 노이즈가 정한다. 은메달을 안정적으로 따려면, 결정론적으로 평가한 뒤 NVARC 베이스라인의 **기대값**을 +3~5%p 올려야 한다. 수단은 re-scoring 개선, self-ensemble, 런타임 활용, 가능하면 합성 데이터 추가 학습이다. 그다음 최종 제출 2개를 기대값 기준으로 고른다.
