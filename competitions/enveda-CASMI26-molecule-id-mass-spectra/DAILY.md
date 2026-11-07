# 일일 운영 매뉴얼 (자동 실행용)

**Orca 자동화** "enveda-CASMI26 일일 운영" (id `ecee5fe1-8447-437e-9dcd-077d584b1159`, cron `30 11,23 * * *` Asia/Seoul, kaggle 워크스페이스, 매번 새 세션)이 이 문서를 따르라는 지시로 Claude 를 실행한다. 사용자는 Orca 앱에서 실행 기록과 세션을 모니터링한다.
(name-that-llm Orca 자동화 09:10 / 21:10 과 겹치지 않게 엇갈림. 예전 launchd 방식 `scripts/daily-casmi.sh` 는 비상용으로만 남김, plist 는 `~/Library/LaunchAgents/disabled/`.)
실행 확인: `orca automations runs --id ecee5fe1-8447-437e-9dcd-077d584b1159 --json`, 즉시 실행: `orca automations run <id>`.
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
0. **지시 확인**: `DIRECTIVES.md` 를 읽는다. "받은 편지함" 에 새 지시가 있으면 절차·QUEUE·도구에 반영하고 "반영됨" 으로 옮긴다 (반영 위치를 적는다). 이 파일의 지시는 이 매뉴얼의 절차보다 우선한다 (절대 규칙은 예외).
   - `insights/PROTOCOL.md` "읽기" 절을 따른다 (유형: science).
1. **상태 수집**: `uv run python -m casmi.daily status` 를 먼저 실행한다. 이 스크립트가
   - 새로 나온 제출 점수를 `EXPERIMENTS.md` 의 해당 행 LB 칸에 기록 (제출 메시지 앞의 `expNNN` 으로 매칭, `대기` → 점수)
   - 오늘(UTC) 사용한 제출 수, 리더보드 금/은/동 경계와 내 순위를 `automation/lb_history.csv` 에 기록
   - `automation/queue.json` 의 대기 커널 상태를 확인하고, commit 완료(`COMPLETE`)된 항목을 한도 안에서 자동 제출
   - 결과 요약을 표준출력으로 낸다. 이 출력을 리포트의 출발점으로 쓴다.
2. **판정**: 점수가 나온 실험의 결론을 `QUEUE.md` 해당 항목에 적는다.
   - Public 은 약 124분자 → 분자 1개 = 0.008. **±0.016 안의 LB 차이는 노이즈**로 본다. CV 와 같은 방향인지만 확인.
3. **다음 제출 준비**: `HYPOTHESES.md` 4절 우선순위의 가설 검증분을 먼저, 그다음 `QUEUE.md` "다음 제출" 위에서부터, CV 근거가 있는 항목만 커널을 만들어 push 하고 큐에 넣는다.
   - 커널 디렉터리 `kernels/<name>/`, id `mins00/casmi26-<name>`, private, 기준 커널은 `kernels/sota0480`.
   - push 후 `uv run python -m casmi.daily enqueue --kernel mins00/casmi26-<name> --exp expNNN --msg "expNNN: <설명>"` (버전은 push 출력의 번호를 `--version` 으로).
   - `EXPERIMENTS.md` 에 `| 날짜 | expNNN | 내용 | CV | 대기 | 메모 |` 한 줄 추가.
   - commit 이 끝나면 다음 실행의 `status` 가 자동 제출한다. 급하면 이번 실행 안에서 `status` 를 다시 돌려도 된다.
4. **로컬/CV 작업** (남는 시간, 최대 약 2시간): `HYPOTHESES.md` 4절 순서대로 다음 가설을 진행하고 (판정은 가설의 채택/기각 기준, 결과는 HYPOTHESES 5절 검증 로그에도 기록), 그다음 `QUEUE.md` "CV 트랙" 항목을 진행하고 결과를 `EXPERIMENTS.md` 에 기록. 좋은 결과는 "다음 제출"로 올린다. 새 아이디어는 백로그에 추가.
5. **메커니즘 점검** (매 실행, 필수): `uv run python -m casmi.daily health` 를 실행한다.
   - 출력의 🟥 항목마다 원인을 한 줄로 진단하고, 이번 실행에서 고칠 수 있으면 고친다. 예: 제출이 48시간 넘게 없음 → CV 트랙이 막힌 지점을 찾아 QUEUE 를 쪼개거나 순서를 바꿈. 같은 커널이 두 번 실패 → 원인 기록 후 우회. CV 와 LB 방향이 자주 어긋남 → CV 하네스를 의심하고 QUEUE 맨 위에 하네스 점검을 올림.
   - 고칠 대상은 절차(이 문서의 "매 실행 절차"), `QUEUE.md`, `casmi.daily` 같은 도구다. 바꾼 것은 `automation/CHANGELOG.md` 에 `날짜 | 무엇을 | 왜 (어떤 health 신호)` 로 남긴다.
   - **절대 규칙은 고치지 않는다**. 규칙이 개선을 막는다고 판단되면 리포트에 "사용자 확인 필요: 규칙 변경 제안" 으로 근거와 함께 올린다.
   - health 지표 자체가 실제 문제를 못 잡고 있다고 보이면 지표를 추가·수정한다 (`casmi/daily.py` 의 `cmd_health`).
6. **주간 회고** (마지막 회고 후 7일이 지났으면, health 출력에 표시됨): `outputs/daily/retro-YYYYMMDD.md` 에
   - 지난 7일의 LB·순위·best CV 추이, 제출 수와 CV 통과율, CV↔LB 방향 일치율
   - 효과 있었던 것 / 없었던 것, 막힌 곳, 다음 7일 계획 (QUEUE 재정렬)
   - 메커니즘 자체의 개선점 (절차·도구·지표). 반영한 것은 CHANGELOG 에.
7. **리포트**: `outputs/daily/report-YYYYMMDD-HHMM.md` 에 한국어로 짧게:
   - 새 점수, 현재 best CV / LB, 순위와 은메달 경계(상위 5%)까지 거리
   - 오늘 제출한 것과 근거, 진행한 CV 작업 결과
   - 메커니즘 점검 결과 (🟥 항목과 조치, CHANGELOG 에 추가한 것)
   - 문제나 사용자 확인이 필요한 일
   - `README.md` 의 Best CV / LB 줄과 메달 컷 줄도 갱신.
   - `insights/PROTOCOL.md` "쓰기" 절: 해당 사항이 있을 때만 `insights/inbox/enveda-CASMI26-molecule-id-mass-spectra.md` 에 한 줄.

## CV 하네스 (현재 상태)
- `casmi.panel_stack`: panel 250분자 PubChem 채널 replay. **유명 천연물 편향으로 판별력 낮음** — prior 관련 변경의 근거로 단독 사용 금지.
- `casmi.cv`, `casmi.rows_v2`: 9~10월 자체 파이프라인용 (holdout_v1 1000분자). 공개 파이프라인과 구조가 달라 참고용.
- **만들 것 (CV 트랙 1순위)**: fold-safe 5-fold FPNet 기반 종류② 시뮬레이션 (아래 QUEUE.md).

## 참고
- 공개 SOTA 분석: `research/05-public-sota-0480.md`. 9월 전략: `research/00-strategy.md`.
- 데이터: `data/comp/` (train/test), `data/ext/panel/` (panel 벤치마크 + identity-disjoint folds).
- 로컬 디스크 여유 약 26GB. PubChem prior 데이터셋(80GB)은 로컬에 받지 말고 Kaggle 커널에서 붙여 쓴다.
