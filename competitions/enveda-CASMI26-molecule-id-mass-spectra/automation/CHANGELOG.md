# 메커니즘 변경 기록

날짜 | 무엇을 | 왜 (어떤 health 신호)
---|---|---
2026-11-05 | DIRECTIVES.md, `casmi.daily health`, DAILY.md 절차 0·5·6 추가 | 사용자 지시: 지시 누적 → 실험·제출 → 파이프라인 개선 루프를 항상 점검·개선
2026-11-05 | daily-casmi.sh 실패 알림(macOS 알림 + failures.log), health 로그 검사에 인증 오류 패턴 추가 | ntl 자동화에서 토큰 만료로 조용히 실패한 사례를 반영 (같은 토큰 공유)
2026-11-05 | launchd → Orca 자동화 `ecee5fe1` (30 11,23 * * *, kaggle 워크스페이스, fresh session) | 사용자 지시: Orca 에서 모니터링. 실패 알림은 Orca 실행 기록으로 대체
2026-11-05 | `casmi.daily health` "자동 실행" 에 Orca 실행 기록(최근 26시간 상태별 수, failed/skipped/missed → 🟥) 추가 | Orca 로 옮긴 뒤 launchd 로그·failures.log 는 더 이상 쓰이지 않아 실행 실패를 못 잡음 (첫 Orca 실행에서 확인)
2026-11-05 | QUEUE CV 트랙 1 에 다음 실행 할 일(출력 받기 → `casmi.sim2_eval`) 명시, `casmi.sim2_eval` 작성 | 제출 근거가 될 CV 하네스가 없어 "다음 제출" 이 막혀 있음 (health 상 아직 🟩 이지만 11/15 중간 점검 전 병목)
2026-11-06 | `casmi.sim2_eval` 기준을 공개 0.480 채널 규칙 전체(ls+lp+src+NP+LOTUS+syn, λ0.15)로 수정. `automation/gpu_log.csv` 신설 (exp014 소급 3h 추정 포함) | 기준이 ls+lp 만이라 공개 규칙과 달랐음. DAILY.md 가 요구하는 GPU 로그 파일이 없어 health GPU 지표가 실제 사용을 못 셌음
2026-11-06 | QUEUE 다음 제출 0번에 exp015/016 판정 절차, CV 트랙 1 을 완료 처리하고 후속 (LambdaRank, sim2 v2 z 스케일) 명시 | sim2 하네스 완성으로 "다음 제출" 병목 해소
2026-11-07 | `casmi.daily health` 에 "CV↔LB" 지표 추가 (EXPERIMENTS 의 expNNN LB 를 14일 안 이전 최고 LB 와 비교, 최근 3개 중 2개 이상 아래면 🟥). 기존 "최근 LB" 는 항상 🟩 라 방향 불일치를 못 잡았음 | exp015/016 이 CV 와 반대로 LB 하락했는데 health 🟥 0 이었음
2026-11-07 | QUEUE 맨 위에 "CV 하네스 점검" (h1 출력 diff, h2 최종 리스트 시뮬레이션, h3 분포) 추가, PubChem 채널 가중치 계열 제출 중단 | health 🟥 CV↔LB (3/3)
