# Experiments

| date | exp | change | CV | LB | note |
|---|---|---|---|---|---|
| 2026-09-29 | v0 | NVARC perfpatch + 결정성(crc32, PYTHONHASHSEED=0) + 과제별 try/except + TTT 마감 가드 + cheap-first + identity fallback | debug 4개: 2.5/4 | **28.89** (약 1000등) | 공개 fork 분포 27~33 의 아래쪽. commit run 은 4×L4 대기열에서 약 13h 대기 후 23분 실행. 과제당 556~1380s. 제출 ref 56652265 |
| 2026-09-29 | v1 | v0 + 남는 시간에 vote margin 낮은 과제부터 새 seed 로 재풀이 (2차 패스), 투표 합산 | debug 4개: 3.0/4 | **29.72** | 2차 패스 정상 동작 (debug 에선 4개 모두 재풀이). 과제당 시간은 1차와 비슷 (534~1231s). 제출 ref 56657059 |
| 2026-09-30 | v2 | v1 + 8-view same-shape decode batches. 입력 경로를 /kaggle/input 탐색으로 찾게 수정 | pending | pending | 9/29 첫 commit(v2, v2-eval)은 하드코딩된 대회 경로 FileNotFoundError 로 추론 0건 (v1 은 같은 경로로 정상). 9/30 재push: v2 ver2, v2-eval ver2 (eval 120 task, 6h) |
