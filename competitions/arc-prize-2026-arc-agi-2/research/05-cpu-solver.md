# 05. CPU program-search solver (icecuber) — GPU 파이프라인과 병렬 실행 (2026-09-29)

코드: `cpu_solver/` (빌드 산출물은 `cpu_solver/.gitignore`). 과제별 결과: [05-cpu-solver-eval.json](05-cpu-solver-eval.json). 원본 출력: `outputs/cpu_solver/` (git 제외).

## 1. 결론
- **icecuber**(ARC 2020 1위, `top-quarks/ARC-solution`, **MIT**)를 골랐다. C++ DSL 142개 함수를 depth 3까지 탐색하고 조각을 greedy로 조합한다. 의존성은 없다.
- **로컬 ARC-AGI-2 eval: 120 task 중 2.5 task (2.08%), 172 output 중 3개.** 풀린 것은 `981571dc`, `aa4ec2a5`, `f931b4a8`(2개 중 1개)이다. 앞의 두 개는 NVARC도 푼다(debug key이고, v0/v1 commit 모두 정답). 새로 얻을 수 있는 것은 **`f931b4a8` 0.5 task 하나뿐**이다.
- 공식 수치와도 맞다. arcprize.org에 따르면 icecuber는 ARC-AGI-2 semi-private에서 1.6%다.
- **train-verified(모든 train 쌍을 맞춤) 후보의 false positive가 크다.** eval에서 top-1이 fits_all인 output 6개 중 3개가 오답이었다(50%). training 100 task에서는 36개 중 12개(33%)였다. **fits_all이 아닌 후보는 eval에서 한 번도 맞지 않았다.** 그래서 fits_all 후보만 쓴다.
- 기대 이득: hidden 240 task에서 **0~2 task(0~0.8점)**. 비용은 CPU 몇 시간과 notebook 셀 2개다. GPU 시간을 쓰지 않으니 넣을 가치는 있다. 다만 우선순위는 낮다. NVARC attempt_1을 CPU 답으로 **바꾸면 안 된다.**

## 2. 후보 조사

| 후보 | 라이선스 | ARC-AGI-2 성능 | 판단 |
|---|---|---|---|
| icecuber `top-quarks/ARC-solution` | MIT | semi-private 1.6% (arcprize.org), 로컬 eval 2.08% | **채택**. 빠르고 C++만 있으면 됨 |
| `victorvikram/ARC-icecuber` 등 fork | MIT (원본 동일) | 개선 수치 보고 없음 | 필요 없음 |
| Hodel `arc-dsl` / `re-arc` | MIT | solvers.py가 ARC-1 **training task 전용**으로 손으로 쓴 프로그램 | hidden task에는 쓸 수 없음. 합성 데이터용 |
| 2024/2025 Kaggle "icecuber + LLM" 앙상블 노트북 | 노트북마다 다름 | ARC-1 기준 몇 점 이득. AGI-2 수치는 없음 | 구조(빈 attempt 채우기)만 참고 |
| 5위 Barbadillo search-and-learn (2025) | — | private task **0개** | 제외 |

- ARC-AGI-2는 brute-force 탐색이 통하지 않도록 설계됐다. DSL 탐색으로 큰 이득을 기대하기 어렵다.
- icecuber의 2020 전체 모델(depth 3, flip 2종, depth 4)은 ARC-1 test에서 20.6%였다.

## 3. 로컬 적용 내용 (upstream 대비 패치)
- `src/read.cpp`
  - `std::filesystem`(C++17)으로 교체했다. `readAll`은 임의 디렉터리를 받고, 파일을 **정렬**해 sample index를 고정한다.
- `src/runner.cpp`
  - 항상 eval(정답 없음) 모드로 돈다.
  - 입출력 디렉터리는 `ICECUBER_DATA` / `ICECUBER_OUT` env로 받는다.
- `headers/precompiled_stl.hpp`
  - `<map>`, `<string>` 등 누락된 include를 추가했다(최신 libstdc++/libc++ 대응).
- `Makefile`
  - `-lstdc++fs`와 `-g`를 뺐다.
  - **`-fsigned-char`를 추가했다.** aarch64 Linux에서는 `char`가 unsigned라서 EOF 비교가 깨지고 모든 task가 assert로 죽었다. x86_64에서는 문제없지만 안전을 위해 넣었다.
- `cpu_solver/run_icecuber.py` (stdlib만 사용)
  - Kaggle 형식 JSON을 task별 파일로 풀고, (task, test) 단위로 `./run idx pass`를 병렬 실행한다.
  - 실행마다 timeout, RSS 한도, 전체 RSS 한도, 전역 deadline, nice를 적용한다. Linux에서는 `RLIMIT_AS`도 건다.
  - 끝나면 후보를 합쳐 `{task: [{cands: [{grid, score, fits_all, passes, n_passes}]}]}` 형태로 쓴다.
  - `--passes ""`로 실행하면 탐색 없이 merge만 한다.
- icecuber score는 `맞춘 train 쌍 수 − 0.01·(depth + 조각수·1e-3)`이다. 따라서 `fits_all = score > n_train − 0.5`로 판정한다. flip pass는 score를 (2−1e-5)로 나누므로 같은 기준이 그대로 맞다.
- 빌드 확인
  - macOS arm64 (Apple clang 21)
  - Docker `gcc:11` (Linux aarch64): 빌드 12초, 3개 task end-to-end 실행 성공(nice 15, RLIMIT_AS)

## 4. 결과

### 4.1 ARC-AGI-2 eval (120 task / 172 output)
환경: MacBook arm64, 10코어/16GB, worker 6개. 다른 부하(Docker VM 등)가 있어서 CPU 사용률이 약 430%였다. 시간은 보수적으로(길게) 잡힌 값이다.

| pass | 의미 | wall (6 workers) | CPU 합 | 실행당 p50 / p90 / max | peak RSS p50 / max | 누적 점수 |
|---|---|---|---|---|---|---|
| 2 | depth 2 | 19s | ~100s | 0.4 / – / 12.7s | 34 / 109MB | 1.0 task |
| 3 | depth 3 | 661s | 2976s | 15.7 / 40.7 / 268s | 232 / 838MB | **2.5 task** |
| 23 | depth 3 + flip(id 6) | 1881s | 8894s | 44.6 / 118.8 / 716s | 350 / 1608MB | 2.5 (추가 0) |
| 33 | depth 3 + flip(id 7) | 2430s | 9478s | 50.4 / 163 / 900s(TLE 1) | 308 / 1589MB | 2.5 (추가 0) |
| 4 | depth 4 | 로컬에서는 불가 | — | 쉬운 74 task도 240s 안에 하나도 안 끝남 | ~1.1GB 이상 | 측정 못 함 |

- 3 pass 합계는 wall 1h23m, CPU 약 5.9 CPU-h다. Linux에서 `981571dc` depth 3는 272s, RSS 2.9GB였다(mac의 RSS는 메모리 압축 때문에 작게 보임).
- 후보 수: 169/172 output에 후보가 1개 이상 있다. top-1 정답 3개, top-2 정답 3개, top-3 정답 3개다.

### 4.2 train-verified 후보의 false positive
| split | top-1 fits_all output | 그중 top-1 정답 | top-2 안에 정답 | FP(top-1 오답) |
|---|---|---|---|---|
| AGI-2 eval (depth 3 + flip) | 6 | 3 | 3 | **50%** |
| AGI-2 training 100 task 무작위 (seed 0, depth 3) | 36 | 24 | 28 | **33%** |

- eval의 오답 fits_all: `135a2760`, `8f3a5a89`(3개 후보가 모두 fits인데 모두 오답), `f931b4a8` test0(출력 크기부터 틀림).
- **flip 합의 신호**: 정답 fits 3개 중 2개(`981571dc`, `aa4ec2a5`)는 flip pass에서도 같은 답이 나왔다. 오답 3개 중 flip에서도 나온 것은 `8f3a5a89` 1개뿐이다. `n_passes ≥ 2`를 요구하면 정밀도 2/3, 재현율 2/3이다. 표본이 너무 작아서 결정은 보류한다(`merge_cpu.py --require-flip-agree` 옵션).
- training 100 task 결과: 34.0% (task 34개, output 36/109). ARC-1 계열이 섞여 있어서 정상 범위이고, 빌드와 파서는 제대로 동작한다.

## 5. NVARC 후보와 합치는 규칙 (제안, NVARC eval-run 후보로 검증 예정)
FP가 33~50%이고 NVARC top-1이 이미 정답인 경우가 많다. 그래서 **NVARC attempt_1은 건드리지 않는다.** (`cpu_solver/merge_cpu.py`)
1. CPU 쪽은 `fits_all` 후보 중 score가 가장 높은 1개만 쓴다.
2. NVARC attempt_1이 fallback이면 CPU 답을 attempt_1에 넣는다. fallback은 `[[0]]`이거나 test input이 그대로 들어간 경우(= NVARC 후보 없음)다.
3. CPU 답이 이미 attempt_1/2 중 하나와 같으면 아무것도 하지 않는다.
4. NVARC attempt_2가 비었거나 fallback이거나 attempt_1과 같으면 CPU 답을 attempt_2에 넣는다.
5. (튜닝 대상) NVARC 2번째 후보의 kgmon score가 `--weak2`보다 낮으면 CPU 답이 attempt_2를 대체한다.
   - 예: DFS view 1개에서만 나온 후보. `kgmon = 찾은 view 수 − 평균 aug NLL`이다.
   - NVARC eval-run pickle에서 "attempt_2가 맞은 비율"을 kgmon 구간별로 보고 threshold를 정한다.
   - CPU fits_all의 정답률 약 50~67%보다 낮은 구간만 대체한다.
- 사용 안 함: "NVARC top-vote margin이 작으면 CPU 답을 attempt_1로" 올리는 규칙. margin이 작아도 NVARC top-1은 attempt_2 자리에서 살아남는다. 하지만 CPU FP가 50%라서 이득이 거의 없고, 기존 attempt_2를 잃는 손해가 생긴다. 데이터로 확인하기 전에는 넣지 않는다.
- 로컬 확인: v1-commit1 submission(debug 4 task만 실제 추론)에 적용하니 3.0 → 3.5다(a1_fill 4, dup 2). 후보가 없던 `f931b4a8`에서 0.5를 얻었다.

## 6. Kaggle 통합 설계 (아직 push 안 함)
- **빌드: notebook 안에서 소스로 빌드한다.**
  - `cpu_solver/kaggle_cell.py`가 C++ 소스, driver, merge 스크립트를 tar.xz+base64로 묶는다(약 63KB).
  - 이것을 셀 하나에 넣어 `/tmp/cpu_solver`에 풀고 `make -j8`로 빌드한다(gcc 11 기준 약 12초).
  - 별도 dataset이나 인터넷이 필요 없다.
  - Kaggle 이미지는 Linux x86_64 + g++(Ubuntu 계열)다. 첫 Save 실행에서 `g++ --version`, `nproc`, `free -g`를 로그로 남겨 확인한다.
  - 빌드가 실패하면 CPU solver만 건너뛰고 GPU 파이프라인은 그대로 간다.
  - 대안은 x86_64 바이너리를 private dataset으로 올리는 것이다. 로컬이 arm64라 Docker `--platform linux/amd64`로 빌드해야 한다. 소스 빌드가 되면 필요 없다.
- **실행**
  - GPU 셀(`starter.py`) **직전**에 `subprocess.Popen(..., start_new_session=True)`로 detach한다.
  - 기본 설정: `--workers 16 --nice 15 --passes 3:900:0,23:1200:0,33:1200:0 --mem-mb 6000 --total-mem-mb 64000 --deadline global_end_time-1800`
  - 모든 자식 프로세스는 `OMP_NUM_THREADS=1`로 돌고 nice 15라서 GPU worker의 CPU 수요(토크나이즈, DataLoader, `.cpu()` 동기화)에 양보한다.
  - 작업 디렉터리는 `/tmp`다. `/kaggle/working`에 쓰면 출력 커밋이 느려지거나 에러가 난다.
  - 예상 비용: hidden 240 task에서 depth 3 + flip 2종이 약 12 CPU-h, 16 worker로 약 1h wall이다. 남는 CPU로 depth 4 pass를 추가할 수 있다(`4:1800:8`; depth 3 대비 약 20배 CPU, 메모리 수 GB). 다만 eval에서 depth 3 → flip의 추가 이득이 0이었으니 depth 4의 기대 이득도 작다.
- **vCPU**: 4×L4는 GCP `g2-standard-48`(48 vCPU, 192GB) 급으로 보인다. Kaggle 공지와 검색 결과에서 "4 L4 → 48 vCPU, 192GB"라는 언급이 있지만 **실측 확인은 아직이다.** 첫 Save 로그로 확인한 뒤 worker 수를 `nproc − 4×(GPU worker당 필요 코어)` 정도로 정한다.
- **merge**
  - `make_submission.py`가 끝난 뒤 merge 셀이 실행된다.
  - 순서: 아직 돌고 있으면 process group을 kill한다 → `run_icecuber.py --passes ""`로 지금까지 나온 답을 merge한다 → `merge_cpu.py`가 `submission.json`을 덮어쓴다.
  - 두 셀 모두 try/except로 감싸서, 실패해도 원래 submission이 그대로 남게 해야 한다(통합할 때 반영).
- 출력 형식: §3의 `cpu.json`. 규칙 튜닝을 위해 commit(eval) 모드에서는 `cpu.json`을 `/kaggle/working`에 복사해 두면 좋다.

## 7. 다음 단계
1. NVARC eval 전체 run의 후보 pickle을 확보한다. 그다음 (a) `f931b4a8` 외에 CPU만 푸는 task가 있는지, (b) `--weak2` threshold를 확인한다.
2. Kaggle Save 1회로 `nproc`/`free`/`g++`를 확인하고, CPU solver가 GPU 처리 시간을 늦추지 않는지(task당 시간 로그 비교) 본다.
3. 이득이 0.5 task 수준에 그친다면 CPU solver는 attempt_2 fallback 용도로만 두고 GPU 쪽 개선에 집중한다.
