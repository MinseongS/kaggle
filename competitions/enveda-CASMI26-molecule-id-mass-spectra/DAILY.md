# 일일 운영 매뉴얼 (자동 실행용)

`scripts/daily-casmi.sh`가 매일 11:30, 23:30 (KST)에 이 문서를 따르라는 지시로 Claude를 헤드리스 실행한다.
(name-that-llm 자동 실행 09:10 / 21:10 과 겹치지 않게 엇갈림.)
Kaggle 제출 한도(하루 5회)는 00:00 UTC = 09:00 KST 에 초기화된다. 채점(숨겨진 test 전체 rerun)은 공개 파이프라인 기준 수 시간.

## 절대 규칙
- **무료 자원만**: Kaggle 무료 GPU/CPU 노트북, 로컬 맥. 유료 API·클라우드 GPU 금지.
- **솔로 참가**: 팀 병합 제안·수락 금지. 디스커션 글쓰기 금지. 공개 publish 금지 (모든 커널·데이터셋 `is_private: true`).
- **다른 대회 건드리지 않기**: `ntl-*`, `arc26-*` 등 다른 대회 커널은 중지·수정하지 않는다.
- **GPU 예산**: Kaggle GPU 는 name-that-llm 과 공유 (주 30시간, 동시 세션 2개). CASMI 는 **주 15시간 이하**. GPU 커널을 띄울 때마다 `automation/gpu_log.csv` 에 `date,kernel,version,est_hours` 를 한 줄 추가하고, 최근 7일 합이 15를 넘으면 GPU 커널을 띄우지 않는다 (CPU 커널로 대체하거나 대기). 세션 2개가 차 있으면 기다린다.
- **제출은 CV 통과분만** (사용자 결정 2026-11-05):
  - "CV 통과" = 아래 CV 하네스 중 해당 변경을 잴 수 있는 것에서 기준 대비 개선이 bootstrap se 의 1배 이상이고, 다른 지표가 나빠지지 않은 것. 근거를 `EXPERIMENTS.md` 에 먼저 적고 나서 제출한다.
  - CV 근거 없는 LB 측정용 제출, public 점수만 보고 가중치를 맞추는 제출은 하지 않는다. 필요하면 리포트에 "사용자 확인 필요"로 제안만 한다.
  - 제출은 commit 이 성공하고 smoke 로그에 에러가 없는 커널 버전만. **하루 5회 이하**.
- 데이터·모델·출력은 git 에 넣지 않는다. **git 커밋은 하지 않는다** (사용자가 요청할 때만).
- 확신이 없거나 위험한 일(데이터셋 삭제, 대량 GPU 소비, 규칙 해석이 애매한 외부 데이터)은 하지 말고 리포트에 "사용자 확인 필요"로 적는다. 외부 데이터는 공개·무료여야 하며 NIST, METLIN 은 금지.

## 매 실행 절차
1. **상태 수집**: `uv run python -m casmi.daily status` 를 먼저 실행한다. 이 스크립트가
   - 새로 나온 제출 점수를 `EXPERIMENTS.md` 의 해당 행 LB 칸에 기록 (제출 메시지 앞의 `expNNN` 으로 매칭, `대기` → 점수)
   - 오늘(UTC) 사용한 제출 수, 리더보드 금/은/동 경계와 내 순위를 `automation/lb_history.csv` 에 기록
   - `automation/queue.json` 의 대기 커널 상태를 확인하고, commit 완료(`COMPLETE`)된 항목을 한도 안에서 자동 제출
   - 결과 요약을 표준출력으로 낸다. 이 출력을 리포트의 출발점으로 쓴다.
2. **판정**: 점수가 나온 실험의 결론을 `QUEUE.md` 해당 항목에 적는다.
   - Public 은 약 124분자 → 분자 1개 = 0.008. **±0.016 안의 LB 차이는 노이즈**로 본다. CV 와 같은 방향인지만 확인.
3. **다음 제출 준비**: `QUEUE.md` "다음 제출" 위에서부터, CV 근거가 있는 항목만 커널을 만들어 push 하고 큐에 넣는다.
   - 커널 디렉터리 `kernels/<name>/`, id `mins00/casmi26-<name>`, private, 기준 커널은 `kernels/sota0480`.
   - push 후 `uv run python -m casmi.daily enqueue --kernel mins00/casmi26-<name> --exp expNNN --msg "expNNN: <설명>"` (버전은 push 출력의 번호를 `--version` 으로).
   - `EXPERIMENTS.md` 에 `| 날짜 | expNNN | 내용 | CV | 대기 | 메모 |` 한 줄 추가.
   - commit 이 끝나면 다음 실행의 `status` 가 자동 제출한다. 급하면 이번 실행 안에서 `status` 를 다시 돌려도 된다.
4. **로컬/CV 작업** (남는 시간, 최대 약 2시간): `QUEUE.md` "CV 트랙" 항목을 진행하고 결과를 `EXPERIMENTS.md` 에 기록. 좋은 결과는 "다음 제출"로 올린다. 새 아이디어는 백로그에 추가.
5. **리포트**: `outputs/daily/report-YYYYMMDD-HHMM.md` 에 한국어로 짧게:
   - 새 점수, 현재 best CV / LB, 순위와 은메달 경계(상위 5%)까지 거리
   - 오늘 제출한 것과 근거, 진행한 CV 작업 결과
   - 문제나 사용자 확인이 필요한 일
   - `README.md` 의 Best CV / LB 줄과 메달 컷 줄도 갱신.

## CV 하네스 (현재 상태)
- `casmi.panel_stack`: panel 250분자 PubChem 채널 replay. **유명 천연물 편향으로 판별력 낮음** — prior 관련 변경의 근거로 단독 사용 금지.
- `casmi.cv`, `casmi.rows_v2`: 9~10월 자체 파이프라인용 (holdout_v1 1000분자). 공개 파이프라인과 구조가 달라 참고용.
- **만들 것 (CV 트랙 1순위)**: fold-safe 5-fold FPNet 기반 종류② 시뮬레이션 (아래 QUEUE.md).

## 참고
- 공개 SOTA 분석: `research/05-public-sota-0480.md`. 9월 전략: `research/00-strategy.md`.
- 데이터: `data/comp/` (train/test), `data/ext/panel/` (panel 벤치마크 + identity-disjoint folds).
- 로컬 디스크 여유 약 26GB. PubChem prior 데이터셋(80GB)은 로컬에 받지 말고 Kaggle 커널에서 붙여 쓴다.
