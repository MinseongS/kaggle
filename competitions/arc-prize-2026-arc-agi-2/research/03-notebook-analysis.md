# 03. 공개 노트북 분석: NVARC 계열 3종

대상 노트북 경로는 `research/ref-notebooks/`이다.

| 약칭 | 노트북 | 하드웨어 | 비고 |
|---|---|---|---|
| **PP** | `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` | NvidiaL4 (4×L4) | public LB 33.89 (현재 약 10위, gold cut 33.47보다 높음) |
| **T4** | `nihilisticneuralnet/baseline-nvarc-arc-25-winning-solution-for-t4x2` | NvidiaTeslaT4 (2×T4) | NVARC를 T4로 포팅한 버전 |
| **V1** | `luxluxshan/arc2-nvarc-v1` ("NVARC+ v2") | NvidiaL4 (4×L4) | 절반 비용 TTT 2회를 풀링 |

셀 번호는 **0부터 센 ipynb 셀 인덱스**다. PP는 Cell 0이 markdown이고, T4는 markdown 없이 Cell 0부터 코드다. 따라서 같은 코드라도 노트북마다 셀 번호가 다르다.

---

## 0. 의존성 (kernel-metadata.json)

세 노트북의 의존성은 모두 같다.

- **model_sources**: `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`
  - NVARC(ARC Prize 2025 우승팀)의 Qwen3-4B SFT 모델이다.
  - 토큰 ID 0–9는 숫자, 10은 개행, 11은 user, 12는 assistant, 13은 pad, 15는 eos다(PP Cell 5 `ARC_VOCAB`). V1 Cell 0 설명에 따르면 어휘를 16토큰으로 축소한 모델이다.
- **kernel_sources**: `sorokin/pip-install-unsloth-flash-patch` (utility script). `ref-notebooks/pip-install-unsloth-flash-patch/`에 pull해 두었다.
  - Cell 0: `unsloth==2025.9.7`, `unsloth_zoo==2025.9.9`, `numpy==2.2.6` 등을 `/kaggle/working`에 설치한다.
  - Cell 1: flash-attn 2.8.2 wheel(cu128, torch2.8, cp311)을 설치한다.
  - Cell 5–6: `unsloth/models/qwen3.py`에 바이너리 patch를 적용한다. 디코드 경로(`paged_attention_K/V`)의 SDPA와 matmul attention을 `flash_attn_func(Q,K,V)`로 교체하고, `temp_O`의 shape 버그를 고친다.
  - **이 patch는 디코드 attention에서 mask와 causal 처리를 제거한다.** 그래서 batched DFS는 배치 안의 시퀀스 길이가 모두 같아야 한다(padding mask가 없음). 또 Unsloth의 **preallocated paged KV 버퍼가 같은 위치를 덮어쓰는** 동작 덕분에 DFS 형제 노드 탐색이 올바르게 돌아간다. Unsloth 버전이 바뀌면 조용히 깨질 수 있는 취약한 의존성이다.
- **dataset_sources**: 없음.
- **competition_sources**: `arc-prize-2026-arc-agi-2`
- **docker_image**: 세 노트북 모두 `python@sha256:3200…e868`(py3.11, torch 2.8)로 고정되어 있다. 인터넷은 꺼져 있다.

---

## 1. 공통 파이프라인 (NVARC 코어)

세 노트북은 `arc_loader.py`, `arc_decoder.py`, `arc_solver.py`, `starter.py`를 `%%writefile`로 만든 뒤 subprocess로 실행한다.

### 1.1 데이터 포맷
- 그리드는 행마다 숫자열을 쓰고 개행으로 구분한다. 형식은 `<|im_start|>user\n{in}<|im_end|><|im_start|>assistant\n{out}<|im_end|>`이다(loader `QwenFormatter`).
- 30×30 출력은 약 931 토큰이다. `max_seq_length=8192`이다.
- 컨텍스트가 8192 토큰을 넘으면 `cut_to_len`이 **앞쪽 train 예제부터 제거**한다(`from_end=False`).
- 로컬 eval 기준 컨텍스트 길이는 중앙값 약 2.6k 토큰, p90 약 4.7k 토큰이다. 8192를 넘는 task는 2개다.

### 1.2 Test-time training (TTT): 적용함, task마다 LoRA를 리셋
PP Cell 5 기준이다.
- 모든 task 시작 시 LoRA를 초기값(`default_weights`)으로 복원한다. 각 task는 독립적으로 학습된다.
- **LoRA 설정**:
  - `r=256`, `lora_alpha=32`, `use_rslora=True`(scale = 32/√256 = 2), `dropout=0`
  - target: `q,k,v,o,gate,up,down,embed_tokens,lm_head`
  - `random_state=42`
- **학습 설정**:
  - `lr=5e-5`, cosine 스케줄, `warmup_ratio=0.1`, `epochs=1`
  - `bs=1`, `grad_accum=1`, `max_grad_norm=1.0`
  - optimizer `adamw_torch`, bf16. T4 버전만 `adamw_8bit`에 fp16을 쓴다.
  - Trainer `seed=42`
- **학습 데이터**: `puzzle_ds.augment(n=16, shfl_keys=True, seed=1)`
  - 기하 변환 8종(transpose 유무 × rot90 0–3회)에 색 순열 16개를 곱해 **128개 시퀀스**를 만든다.
  - 색 순열 `permute_rnd_all_`은 배경 0을 포함한 10색 전체를 무작위로 섞는다.
  - `shuffle_ex`로 train 예제 순서도 섞는다.
  - 각 시퀀스는 해당 task의 train 쌍 전체이고, **모든 assistant 출력에 loss를 건다**(collator `j%2==1`).
  - test 입력은 학습에 쓰지 않는다.
  - 128 step, 1 epoch.

### 1.3 디코딩: batched DFS(`turbo_dfs`)
- **추론 view**: `puzzle_ds_multi.augment(n=2, seed=2)`로 8 기하 × 2 색순열 = test 출력당 **16 view**를 만든다.
- **배치 구성**: test 출력마다 4 view씩 4배치를 만든다. 조합은 `[id×2, rot180×2]`, `[rot90×2, rot270×2]`, `[T×2, T.rot180×2]`, `[T.rot90×2, T.rot270×2]`이다. 같은 shape 클래스끼리 묶어야 토큰 길이가 같아진다(PP Cell 5 하단).
- **DFS 규칙**:
  - 누적 NLL이 `max_score = -ln(0.2)`보다 작은 경로만 확장한다. 즉 **누적 확률 0.2 이상인 시퀀스만 남기므로 view당 후보는 최대 5개**다.
  - 12개 ARC 토큰만 고려한다.
  - DFS 호출 하나(배치 하나)에 벽시계 **540초** 한도가 있다.
- **역변환**: 각 후보를 `invert_mod`로 원래 좌표계와 색으로 되돌린다.

### 1.4 후보 재채점(augmentation scoring)
- 새 후보 그리드마다 `augment(seed=hash(bk)%1024**2)`로 8 기하 × 색순열 1개 = **8 view**를 만든다.
- 각 view에서 teacher-forced NLL을 계산한다(`calc_scores`, 4개씩 2배치).
- 결과는 `known_scores`에 캐시하므로 같은 그리드는 한 번만 채점한다.

### 1.5 투표와 시도 2개 선택 (`arc_decoder.py`, PP Cell 4 / T4 Cell 3)
- 기본은 `score_kgmon`이다. 점수 = (그 그리드를 만든 view 수, 최대 16) − (augmented NLL 8개의 평균)이다. 같은 view 안에서 같은 그리드는 한 번만 나오므로 앞 항은 사실상 view 투표 수다.
- 대안으로 `score_full_probmul_3`(Σ(3−beam_nll) + mean Σ(3−aug_nll))도 있지만 benchmark 출력에만 쓴다.
- **attempt_1과 attempt_2는 kgmon 순위 1, 2위**다(`fill_submission`).
- 후보가 1개뿐이면 attempt_2는 `[[0]]`, 후보가 없으면 두 시도 모두 `[[0]]`이다.

### 1.6 병렬화와 시간 가드
- `starter.py`가 `mp.spawn`으로 GPU마다 worker 하나를 띄우고 공유 queue에서 task를 꺼낸다. 순서는 **task id 알파벳순**이다.
- Unsloth patch 경합을 피하려고 rank k는 rank k−1의 import 완료 마커를 기다린다.
- 시간 가드는 네 가지다.
  1. `global_end_time`: PP는 12h − 600s(Cell 1), T4는 12h − 1200s(Cell 0).
  2. task 시작 전 `time.time() > end_time`이면 worker를 멈춘다.
  3. **디코드 배치 시작 전에만** `spend_time > 1200`(TTT 시간 포함)이나 end_time 초과를 검사하고, 걸리면 그 task의 남은 배치를 버린다.
  4. DFS 내부에서 540초 창이나 end_time을 넘으면 탐색을 멈춘다.
- **TTT(`trainer.train()`)와 재채점 루프에는 시간 가드가 없다.**
- non-rerun(commit) 모드는 eval set에서 4개 task(`0934a4d8, 36a08778, 981571dc, aa4ec2a5`)만 돌리고, 마지막 셀에서 `benchmark_selection_algos()`와 reload score를 출력한다.

---

## 2. 노트북별 차이

### 2.1 PP: `arc-agi2-lb33-89-minimal-perfpatch`
- **Cell 0 (markdown)** 요지: 원본 `baseline_LB33.89`(NVARC L4 버전)의 셀 9개, 4×L4, 학습 128 / 추론 16 view, LoRA, 임계값, 선택 로직, 시간 예산을 모두 그대로 두었다. 바꾼 것은 logits 핫스팟 두 곳뿐이다.
- **perfpatch 변경점 (Cell 5)**:
  1. `turbo_dfs`
     - 원본은 `logits.float().cpu().log_softmax(-1)`로 vocab 전체를 CPU로 보냈다.
     - 패치는 GPU에서 `logsumexp`를 계산하고 12개 토큰만 `index_select`해서 `.cpu()`로 보낸다.
  2. `calc_scores`
     - `use_cache=True`를 `use_cache=False`로 바꿨다.
     - GPU에서 target 토큰만 gather한 뒤 `logsumexp`로 정규화한다.
     - 원본은 `[B, L, V]` logits 전체를 CPU로 보내 log_softmax를 계산했다.
- **효과는 크지 않을 것으로 본다.**
  - vocab이 16토큰이면 CPU로 옮기는 데이터량 차이는 작다. 스텝마다 `.cpu()` 동기화도 그대로 남아 있다.
  - 실질적 이득은 `calc_scores`의 KV cache 쓰기와 CPU softmax 제거 정도로 보인다.
  - 수치적으로는 "거의" 동일하다(GPU와 CPU fp32 연산 순서 차이). 0.2 임계 부근에서 드물게 결과가 뒤집힐 수는 있다.
  - **LB 33.89는 이 패치 덕분이라기보다 NVARC의 run-to-run 분산(아래 5.3) 안에서 운 좋게 나온 값일 가능성이 높다.**
- **T4 버전 대비 차이** (T4 Cell 4 ↔ PP Cell 5, T4 Cell 5 ↔ PP Cell 6)

  | 항목 | T4 | PP |
  |---|---|---|
  | 가중치 로드 | `load_in_4bit=True`, fp16 (QLoRA) | `load_in_4bit=False`, bf16 전체 |
  | attention | xformers `memory_efficient_attention` 몽키패치 (T4는 flash-attn2 미지원) | Unsloth + flash-attn |
  | optimizer | `adamw_8bit` | `adamw_torch` |
  | 정밀도 | fp16/bf16 off, `half_precision_backend="cpu_amp"`, GradScaler 비활성 | bf16=True |
  | worker 수 | 2 | 4 |
  | `OMP_NUM_THREADS` | 6 | 12 |
  | end buffer | 1200s | 600s |
  | 모델 경로 | `/kaggle/input/qwen3_4b…` | `/kaggle/input/models/sorokin/…` |

  - T4는 스케일러 없이 fp16으로 학습하므로 overflow나 NaN 위험이 있다.
  - T4는 2 GPU × 약 11.7h ÷ 240 task ≈ **task당 350초**밖에 확보하지 못한다. 알파벳순 뒤쪽 task는 처리하지 못할 가능성이 크다. 실전 제출용으로는 부적합하다.

### 2.2 T4: `baseline-nvarc-arc-25-winning-solution-for-t4x2`
- 파이프라인은 1장과 같고 차이는 위 표에 정리했다.
- Cell 4(`arc_solver.worker`)에서 xformers patch를 쓴다. GQA 처리를 위해 K와 V를 `repeat_interleave`하는데, 이 때문에 메모리와 속도 손해가 있다.
- 로컬 실험을 T4에서 해야 한다면 참고할 만하다. 다만 점수와 속도 모두 L4 버전보다 불리하다.

### 2.3 V1: `arc2-nvarc-v1` (NVARC+ v2)
- **Cell 0 (markdown) 요지**
  - 같은 NVARC 노트북을 재실행하면 public LB가 **29.7 / 31.8 / 26.9 / 32.2 / 31.4**로 흔들린다(작성자 본인 5회 측정).
  - 그래서 절반 비용 pass를 서로 다른 seed로 2회 돌리고 후보를 풀링한다.
- **Cell 1**: rerun이면 12h − 600s, commit이면 55분 예산으로 모든 단계를 끝까지 한 번씩 돌린다.
- **Cell 5 `arc_solver.py`**
  - `CFG`로 모든 하이퍼파라미터를 환경변수 override할 수 있다. 전체 목록은 4장에 있다.
  - `resolve_model_dir()`는 모델 경로가 달라져도 찾아낸다.
  - `stable_seed()`: 재채점 augmentation seed를 `hash(bk)`에서 `zlib.crc32`로 바꿨다. **PP와 T4는 PYTHONHASHSEED를 고정하지 않으므로, 재채점 순열이 프로세스와 run마다 바뀌는 비결정성이 있다.**
  - `make_view_batches()`: `n_perm`과 `batch_size`를 일반화했다.
  - task 단위 `try/except`로 OOM 등이 나도 worker가 살아남는다.
- **Cell 6 `starter.py`**
  - `estimated_work()`로 **싼 task를 먼저** 처리한다. 대략 train 토큰×16 + test 토큰×8×출력 수로 비용을 추정한다.
  - `--keys-file`, `--nprocs`, `--order`를 지원한다.
  - 마커 대기는 최대 900초이고, worker가 죽으면 2회까지 재시도한다.
- **Cell 7 (Pass A)**
  - `ARC_N_TRAIN_AUG=8`(TTT 64 시퀀스), `ARC_N_EVAL_AUG=1`(8 view), `ARC_TASK_CAP=800`, `PYTHONHASHSEED=0`, cheap-first 순서다.
  - **markdown은 pass A를 "LB-proven configuration"이라고 부르지만, 실제로는 절반 비용 설정이다.** 따라서 "PP보다 나빠질 수 없다"는 보장은 성립하지 않는다.
- **Cell 8 (Pass B)**
  1. pass A가 도달하지 못한 task를 같은 설정으로 먼저 처리한다(catch-up).
  2. 전체 task를 새 seed로 한 번 더 돈다: `LORA_SEED=137`, `TRAIN_AUG_SEED=17`, `EVAL_AUG_SEED=29`, `SCORE_SEED_OFFSET=7`, cap 700, cheap-first, deadline까지.
  - 남은 시간이 20분 미만이면 해당 pass를 건너뛴다.
  - pass마다 subprocess를 새로 띄우므로 모델 로드 비용(4 GPU × 수 분)이 반복된다.
- **Cell 9 (선택)**
  - A와 B 후보 풀을 합쳐 `score_kgmon`으로 순위를 매긴다. 투표 수는 합산하고, NLL은 각 pass의 어댑터로 계산한 값을 섞어 쓴다.
  - 풀링 top-2에 pass A의 1위가 없으면 `[pooled#1, A#1]`으로 강제한다.
  - attempt_1이 비어 있으면 **test 입력을 그대로 복사**한다(identity fallback). attempt_2가 attempt_1과 같으면 `[[0]]`로 둔다.
  - 스키마를 재검증하고 sha를 출력한다.
- **V1의 약점: 배치 효율 저하**
  - `n_eval_aug=1`이면 `make_view_batches`의 else 분기에서 **2 view짜리 배치 4개**가 만들어진다.
  - DFS는 스텝 지연시간이 병목이다. 배치 크기가 절반이 되어도 배치당 벽시계 시간은 거의 줄지 않는다.
  - 결과적으로 view 수는 절반인데 디코드 시간은 거의 그대로다.
  - 개선안: `{id, rot180, T.rot90, T.rot270}`와 `{rot90, rot270, T, T.rot180}`는 각각 shape 클래스가 같다. 이 4개씩을 한 배치로 묶으면 디코드 호출 수가 절반으로 준다.

---

## 3. GPU와 런타임 예산

- **hidden rerun 규모**: 240 task, 259 출력. 로컬 `arc-agi_test_challenges.json`은 training task 240개를 넣어 둔 placeholder이고, 같은 개수로 맞춰져 있다.
  - public과 private 점수는 **같은 1회 실행**의 결과를 각 부분집합에서 채점한 것으로 추정한다(ARC Prize 관례). 따라서 run 노이즈는 두 점수에 독립적으로 들어간다.
- **PP (4×L4)**: 4 × 약 11.8h ≈ 47 GPU·h ÷ 240 ≈ **task당 약 700초**가 평균 예산이고, task cap은 1200초다.
  - 알파벳순 처리라서 무거운 task가 몰리면 **뒤쪽 task가 통째로 비어** `[[0]]`로 제출될 수 있다.
- **시간 가드의 빈틈 (PP와 T4 공통)**
  1. TTT에 가드가 없다. end_time 직전에 시작한 task의 TTT(수 분)가 600초 버퍼를 넘기면 **전체 12h 초과, 즉 제출 실패** 위험이 있다.
  2. `spend_time`에 TTT 시간이 포함된다. 큰 task는 TTT만으로 1200초에 가까워지고, 그러면 디코드를 한 배치도 못 해 결과가 0개가 된다.
  3. 540초와 1200초라는 **벽시계 기반 절단**은 GPU 부하와 thermal 상태에 따라 결과가 달라지는 비결정성의 원인이다.
  4. PP에는 task 단위 예외 처리가 없다. `torch.multiprocessing.spawn`은 한 프로세스가 비정상 종료하면 **나머지 worker도 종료**시킨다. OOM 한 번에 남은 task 전부를 잃을 수 있다. 셀은 `!python`이라 노트북 자체는 이어서 진행되고, 부분 결과로 submission을 쓴다.
- **V1**
  - 예외 처리, cheap-first 순서, catch-up이 있어 위 위험이 크게 줄었다.
  - 다만 pass B 시작 조건이 "남은 시간 ≥ 20분"이라서, pass B 마지막 task의 TTT가 end_time을 넘길 위험은 여전히 있다.

---

## 4. 튜닝 가능한 knob 전체 목록

| 영역 | knob | 기본값 (PP) | 위치 |
|---|---|---|---|
| 예산 | `global_end_time` 버퍼 | 12h−600s | PP Cell 1 / V1 Cell 1 |
| 예산 | task cap (TTT 포함) | 1200s (V1: A 800, B 700) | PP Cell 5 / V1 `ARC_TASK_CAP` |
| 예산 | DFS 창 | 540s | `ARC_DFS_WINDOW` |
| 병렬 | nprocs, OMP threads, task 순서 | 4, 12, 알파벳 | PP Cell 6–7 / V1 `--order` |
| LoRA | r, alpha, rslora, dropout, target modules, init seed | 256, 32, True, 0, 전체+emb+head, 42 | Cell 5 `peft_params` |
| TTT | lr, epochs, warmup_ratio, scheduler, optim, bs, grad_accum, clip, Trainer seed | 5e-5, 1, 0.1, cosine, adamw_torch, 1, 1, 1.0, 42 | Cell 5 `train_args` |
| TTT 데이터 | `n_train_aug`(×8 기하), aug seed, `shfl_keys`, max_seq_length, cut 방향 | 16, 1, True, 8192, 앞에서 제거 | `augment`, `cut_to_len` |
| 디코드 | `n_eval_aug`(×8), eval seed, `min_prob`, 배치 구성 | 2, 2, 0.2, 4 view | `ARC_N_EVAL_AUG`, `ARC_MIN_PROB`, `ARC_DECODE_BATCH` |
| 재채점 | 재채점 aug 수, seed, 배치 분할 | n=1(8 view), `hash(bk)`, 4+4 | Cell 5 scoring 블록 |
| 선택 | 알고리즘, probmul baseline, n_guesses | kgmon, 3, 2 | `arc_decoder.py` |
| 앙상블 (V1) | pass 수, pass별 seed와 설정, A#1 강제 여부, fallback | 2, SEED_B, True, identity | V1 Cell 7–9 |

---

## 5. 약점, 개선 기회, 리스크 (private 점수 기준)

### 5.1 private 점수를 안정적으로 올릴 개선안 (우선순위순)
1. **적응형 예산 스케줄링**: 가장 큰 리스크를 줄이고 기대 이득도 가장 크다.
   - cheap-first 순서를 쓴다(V1 방식).
   - `cap = 남은시간 × GPU수 / 남은 task 수`로 동적으로 계산한다.
   - TTT에 `max_steps`와 가드를 둔다. 예: `remaining < 추정 TTT+최소디코드`이면 TTT를 축소하거나 건너뛰고 디코드만 한다.
   - "모든 task가 최소 1회 decode(예: 4 view)를 받는 것"을 먼저 보장한다. 남는 시간은 불확실한 task(후보 margin이 작은 task)에 추가 view나 seed로 쓴다.
   - private 240 task 중 비어서 제출되는 task를 없애는 것이 가장 확실한 점수 방어다.
2. **디코드 배치 재구성**: 같은 shape 클래스 4 기하 × 색순열 2 = 8 view를 한 배치로 묶는다.
   - DFS 호출이 출력당 4회에서 2회로 줄어 디코드 시간이 약 2배 빨라질 것으로 예상한다.
   - KV는 8×8k 토큰 기준 약 10GB(Qwen3-4B: 36층 × 8 KV head × 128 dim)다. L4 24GB에서 대부분 task는 가능하다. 긴 task는 4 view로 fallback한다.
   - 절약한 시간은 view 추가, `min_prob` 완화, 2번째 seed에 재투자한다.
3. **seed 앙상블과 풀링 (V1 아이디어 검증)**
   - run 간 분산이 크다(26.9–32.2). 각 run이 서로 다른 약 5개 task를 맞힌다는 뜻이다. 서로 다른 seed의 풀을 투표로 합치면 기대값이 오르고 분산이 준다.
   - 단, 절반 비용 설정 2회가 전체 설정 1회보다 나은지는 로컬 CV로 확인해야 한다.
   - 2번 개선으로 얻은 속도를 쓰면 "전체 설정 + α seed"도 가능해진다.
4. **attempt_2 활용**
   - 후보가 1개뿐이라 attempt_2가 `[[0]]`인 출력이 많을 수 있다.
   - 해당 출력만 `min_prob`를 0.05–0.1로 낮춰 재디코드하거나, 다른 seed pass의 후보로 채운다.
   - 추가 비용이 작고 순수하게 이득이다.
5. **선택 알고리즘 오프라인 튜닝**
   - `/kaggle/inference_outputs` pickle을 dataset으로 저장해 두면 GPU 없이 kgmon, probmul, 가중 조합, 재채점 view 수(8→16) 등을 즉시 비교할 수 있다.
   - 과적합을 피하려면 파라미터 1–2개짜리 규칙만 쓴다.
6. **결정성 확보**
   - `hash(bk)`를 crc32로 바꾼다(V1 방식). `PYTHONHASHSEED=0`을 설정한다.
   - 벽시계 절단을 **노드와 스텝 수 예산**으로 바꾼다. 재현성이 생겨 CV 비교가 가능해진다.
   - bf16과 flash-attn backward의 비결정성은 남지만 크지 않다.
7. **견고성**
   - task 단위 `try/except`와 OOM 복구를 넣는다(V1 방식).
   - 빈 출력에는 identity나 최다 후보를 fallback으로 넣는다.
   - 마지막 제출 셀은 반드시 실행되는 독립 셀로 둔다.
8. (연구 과제) TTT 품질
   - 작은 task에 2 epoch을 적용한다.
   - lr과 r을 조정한다.
   - 배경 0을 고정한 색순열을 쓴다. 다만 NVARC SFT 분포와 어긋날 수 있다.
   - 모두 CV 없이 넣으면 위험하다.

### 5.2 리스크
- **private rerun 타임아웃**: 3장의 TTT 무가드, end buffer 600초, PP의 spawn 연쇄 종료 문제. V1은 pass 3회분 모델 로드가 추가된다.
- **빈 task**: PP는 알파벳순이라 deadline에 걸리는 tail task가 `[[0]]`로 나간다.
- **의존성 취약성**: Unsloth 2025.9.7 전용 바이너리 patch를 쓰고, 인터넷이 꺼진 상태에서 utility script에 의존한다. 이 patch가 없으면 DFS의 KV 덮어쓰기 가정이 깨질 수 있다.
- **비결정성**: `hash()` 기반 seed(PP, T4), 벽시계 절단, GPU 비결정성, worker별 task 배정 시점 차이. 이 때문에 public 점수가 약 ±2.5점 흔들린다.

### 5.3 public 점수의 노이즈
- 같은 코드 5회 실행 결과가 26.9–32.2이고 표준편차는 약 2.2점이다. PP의 33.89는 분포 상단일 가능성이 높다.
- public과 private이 각각 약 120 task라고 보면 1 task ≈ 0.8점이다. **LB 차이 2–3점은 유의하지 않다.**
- gold cut(33.47) 근처에서 경쟁하려면 "기대값 상승"과 "분산 축소"가 핵심이다. 앙상블, 빈 task 제거, 결정성 확보가 여기에 해당한다. public LB를 반복 제출해 좋은 값을 고르는 것은 private에 의미가 없다.

---

## 6. 로컬 CV (eval set) 활용 가능성과 소요 시간

- **데이터 구성**
  - `arc-agi_evaluation_challenges.json`: 120 task, 172 출력, 정답 포함. ARC-AGI-2 public eval이다.
  - `arc-agi_test_challenges.json`: training task 240개로 채운 placeholder다.
- **사용 방법**
  - 세 노트북 모두 commit 모드에서 eval set을 읽는다.
  - 전체 CV를 돌리려면 `starter.py`의 debug key 필터를 수정한다. PP는 Cell 6의 4개 key 목록이고, V1은 `ARC_DEBUG_KEYS` 환경변수나 `--keys-file`이다.
  - 채점은 마지막 셀의 `benchmark_selection_algos()`와 `validate_submission()`이 자동으로 한다(PP Cell 8, V1 Cell 9).
- **주의 1: 데이터 누수 가능성**
  - NVARC SFT 데이터가 ARC-AGI-2 public eval을 합성 seed나 학습 데이터로 썼는지 확인해야 한다.
  - 확인 방법: **TTT를 끈 상태**로 eval을 돌려 점수가 비정상적으로 높은지 본다.
  - 누수가 있으면 절대 점수는 과대평가된다. 추론과 선택 knob의 상대 비교에는 어느 정도 쓸 수 있지만 TTT 관련 비교는 왜곡된다.
- **주의 2: 분산**
  - 120 task에서 run 간 표준편차가 약 2점이다.
  - **같은 task 집합에서 task별로 짝지은(paired) 비교**를 해야 한다.
  - 가능하면 seed 2개 이상을 돌리고, 결정성 패치를 먼저 적용해 노이즈를 줄인다.
- **소요 시간 추정**
  - **4×L4, PP 전체 설정(128 TTT, 16 view)**
    - task당 TTT가 약 330k 토큰(128 × 약 2.6k)이다. L4 bf16 LoRA r256 기준 약 3–5분으로 본다.
    - 디코드와 재채점이 출력당 약 1–3분이다.
    - 합치면 task당 약 5–10분, 120 task ÷ 4 GPU ≈ **2.5–5시간**이다.
    - 상한은 cap 기준 120×1200/4 ≈ 10h다.
  - **4×L4, 절반 설정(V1)**: 약 1.5–3시간.
  - **2×T4 (4bit, fp16, xformers)**
    - GPU당 약 2–3배 느리고 GPU 수는 절반이라 **약 10–20시간**이다. 12h 세션 한 번에 들어가지 않을 수 있다.
    - 40–60 task 층화 부분집합이나 절반 설정을 권장한다. 참고로 T4 baseline도 hidden 240 task에서는 task당 약 350초밖에 쓰지 못한다.
  - 실측으로 보정해야 한다. 먼저 20 task 파일럿을 돌려 task당 시간 로그(`finished {key} in …s`)를 모은다.
- **권장 워크플로**
  1. L4×4에서 결정성 패치를 적용한 PP 설정으로 eval 120 task를 1회 돌린다. 후보 pickle은 Kaggle dataset으로 저장한다.
  2. 선택 알고리즘과 attempt_2 규칙은 저장한 pickle로 CPU에서 오프라인 비교한다.
  3. 배치 재구성, 예산 스케줄러, seed 풀링은 GPU에서 A/B 비교한다. 같은 task에서 짝지어 비교한다.
