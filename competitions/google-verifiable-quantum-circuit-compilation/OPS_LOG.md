# 운영 로그 (감독 세션과 대화 세션이 기록)

## 2026-11-05 대화 세션
- 상태: LB 24.56271, 664/2372, 은 컷 24.80641 (-0.244). 공개 24.79 노트북 등장으로 은 컷이 하루에 +0.22
- 발견: 수집 출력 7.5GB 중 dqi.kmb 동일 파일 10부, 디스크 여유 12GB
- 조치: harvest 에 하드링크 중복 제거·스냅샷 정리 추가 (7.5→3.1GB). healthcheck, 감독 세션, DIRECTIVES.md 신설
- 사람 필요: `claude setup-token` 토큰을 ~/.config/vqcc/claude_oauth_token 에 저장 (11/7 전)
- 추가 조치: 사용자 요청으로 launchd → Orca 자동화 전환 (수집·제출 07:00/19:00, 밤 연구 01:00, 아침 감독 08:30, Asia/Seoul, 이 workspace 에서 실행). launchd plist 는 `.disabled` 로 이름 변경. healthcheck 가 Orca 실행 기록을 읽도록 수정

## 2026-11-06 07:17 (수집·제출 실행)
- 발견: 07:11 자동 제출 실패 — `400 FAILED_PRECONDITION: Submission files must be named "submission.zip"`. harvest 가 `outputs/harvest/best.zip` 으로 제출하고 있었음 (첫 자동 제출이라 이제야 드러남)
- 조치: `src/harvest.py` build_zip 이 `outputs/harvest/best_zip/submission.zip` 을 만들도록 수정, 재실행 → 07:16 제출 성공 (local GM 24.85790, invsqrt 21.246 / dqi 65.107 / dfthcsos 11.104), 채점 대기
- 남은 문제: healthcheck 의 Orca 자동화 조회 실패 3건 (이 세션에서 orca 조회 불가 여부 확인 필요), Claude 토큰 만료 11/7~11/10
- 추가 조치: healthcheck 의 Orca 조회 실패는 daily_harvest.sh PATH 에 `/Applications/Orca.app/Contents/Resources/bin` 이 없어서였음 → PATH 에 추가
- 08:00 확인: 제출 LB 24.85789 (로컬 24.85790 과 일치), 순위 757→119/2410, 은 컷과 동점

## 2026-11-06 감독
- 상태: 132/2412, best LB 24.85789 (로컬 24.85790 과 일치), 은 컷 24.87121 (-0.0133, 오전 중에도 24.86787→24.87121 로 상승), 오늘(UTC) 제출 1/5. 08:33 수동 수집 결과 공개 포트폴리오 24.85794 (+0.00004, 제출 기준 미달 → 미제출). 공개 흡수만으로는 은 컷에 못 미침
- 발견:
  - 밤 연구 Orca 상태 `dispatch_failed`("Orca stopped watching", 01:11)는 오탐. 세션은 01:00~02:25 실제로 일함 (r001~r012, loader v1 PASS). 다만 **5시간 중 1.4시간만 쓰고 멈춤**, 세션은 지금도 idle 상태로 터미널에 남아 있음
  - 연구 세션이 남긴 백그라운드 대기 루프(`until grep prune_*.log ...; sleep 10`, PID 58650)가 조건 불충족으로 6시간 넘게 돌고 있었음
  - healthcheck: 밤 연구를 launchd 시절 `research_logs/` 로만 판단해 "실행 기록 없음"으로 표시. 이미 해소된 07:11 제출 실패를 계속 오류로 표시
  - harvest 가 submission.zip 없는 노트북을 조용히 건너뜀 (예: leoprovorov "Full-XOR Chemistry", general_receipt.json 만 있음)
  - 공개 chem 최고가 자체 loader 컴파일러("C43 native compiler", T 15181→15092)로 갱신됐는데 RESEARCH.md 표는 11-05 값
  - 낭비: 제출 1/5 사용. 개선 후보가 없어서이지 파이프라인 문제는 아님. 디스크 여유 36GB 정상
- 조치:
  - 대기 루프 kill
  - `scripts/research_prompt.md`: 05:30 까지 계속 일할 것, 막히면 다음 항목으로 갈 것, 대기 루프에 timeout 필수, 2시간 진전 없으면 방향 전환
  - `scripts/healthcheck.py`: 00:30~07:00 파일 수정 시각으로 밤 연구 실제 활동 시간 측정 (3시간 미만이면 경고, 활동이 있으면 dispatch_failed 에 주석), 마지막 `submit OK` 이후의 실패만 집계
  - `src/harvest.py`: submission.zip 이 없는 노트북을 로그에 남김 (`--no-submit --skip-download` 로 동작 확인)
  - `RESEARCH.md`: 공개 최고 표 갱신, 큐 -1 (은 컷까지 과제 하나 비용 ~0.16% 감소면 충분, 작은 개선 먼저), 2a (C43 native compiler 노트북 소스를 `kaggle kernels pull` 로 받아 읽기만 → loader v2 지름길)
  - `README.md` best CV/LB 갱신
- 사람 필요:
  - **Claude 로그인 갱신 (11/7~11/10 만료 예정)** — 만료되면 모든 Orca 자동화가 멈춤
  - Orca 가 밤 연구 run 을 11분 만에 "stopped watching" 처리한 원인 확인 (Orca 쪽 문제로 보임, 결과 보고가 실행 기록에 안 남음). 끝난 연구/감독 세션 터미널이 idle 로 계속 쌓이므로 가끔 닫아 줄 것

## 2026-11-07 감독
- 상태: 315/2507, best LB 24.94093 (11-06 19:10 제출), 은 컷 24.96481 (−0.0239), 오늘(UTC 11-06) 제출 3/5 (08:38 KST 감독 제출 1회 포함, 채점 대기)
- 발견:
  - **07:14 자동 제출이 로컬 24.94136 → LB 24.87108 (−0.070).** 역산하면 LB 에서 dqi 의 R = 1.2013887 이고, 이는 102,400샷 중 정확히 2샷 실패일 때(R 1.2013881)와 일치한다. 원인 회로는 manwithacat "the real 150k was the friends we made" dqi (cac29de9). 빈 salt 와 로컬 salt 3개(a1, s1, 기존 1개)에서는 실패 0이었으니 드물게 실패하는 회로이고, 비공개 salt 에서만 드러났다. invsqrt 비율은 직전 제출과 같고, dfthcsos 는 salt 를 바꿔도 비용이 같았다(T 14966).
  - harvest 의 제출 기준선이 LB 가 아니라 로컬 GM(24.94136)이었다. 그래서 LB 로 확인된 더 나은 조합이 다시 제출될 수 없었다. 또 prune 이 직전 LB 확인 제출의 출처(alexchilton 6b73, yoyobar 4903, heliosli 09e2)를 지워서 그 조합을 복구할 수 없었다.
  - 밤 연구는 01:02~03:22 만 일함. 큐가 모두 "재학습급" 이라며 스스로 종료 → 같은 자리(공개 위 국소 개선 포화 확인)를 맴돎.
- 조치:
  - `src/harvest.py`:
    - (1) dqi 최종 후보를 비용 순으로 salt 4개까지 늘려 검증하고 가장 나쁜 비용을 쓴다 (`harden`, 후보 최대 4개).
    - (2) `lb_exclude.json`: 자동 제출의 LB 가 로컬보다 0.01 이상 낮으면 그 제출의 dqi 를 자동 제외한다. cac29de9 가 이 경로로 제외됨을 확인했다.
    - (3) 제출 기준선 = max(LB 최고, 채점 대기 중 제출의 로컬 GM).
    - (4) 같은 파일 조합은 다시 제출하지 않는다 (history 에 `keys` 기록, 07:14 항목은 소급 입력).
    - (5) 최근 제출 3개의 출처는 prune 하지 않는다.
  - `--no-submit --skip-download` 로 동작 확인: prvsiyan dqi (edcde2ae) salt 4개 모두 실패 0. 포트폴리오 24.94111 > LB 최고 24.94093 → `--skip-download` 로 제출 OK (08:38).
  - `scripts/healthcheck.py`: lb_exclude 로 처리된 LB-로컬 차이는 경고에서 뺀다.
  - `RESEARCH.md`: 큐 −3 신설 (A: invsqrt 처음부터 재설계 — 비공개 22.68 보고, 공개 21.25. B: chem QAT 를 로컬 M4 MPS 에서 밤새 백그라운드). dqi salt 규칙 추가.
  - `scripts/research_prompt.md`: "큰 작업뿐" 은 멈출 이유가 아님. 첫 단계부터 시작하고 긴 학습은 시작 직후 `nohup timeout` 으로 띄운다. 끝에 다음 밤 첫 행동 3개를 적는다.
  - `README.md`: best CV/LB 갱신.
- 사람 필요:
  - **Claude 로그인 갱신 (11/7~11/10 만료 예정, `claude auth login`)** — 만료되면 Orca 자동화(수집·연구·감독)가 모두 멈춤. 11-05 에 요청한 `~/.config/vqcc/claude_oauth_token` 도 아직 없음
  - 08:38 제출의 LB 가 24.9411 근처로 나오는지 확인 (healthcheck 가 자동 표시)
