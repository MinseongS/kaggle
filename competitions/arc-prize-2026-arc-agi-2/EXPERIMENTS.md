# Experiments

| date | exp | change | CV | LB | note |
|---|---|---|---|---|---|
| 2026-09-29 | v0 | NVARC perfpatch + 결정성(crc32, PYTHONHASHSEED=0) + 과제별 try/except + TTT 마감 가드 + cheap-first + identity fallback | debug 4개: 2.5/4 | pending | commit run 은 4×L4 대기열에서 약 13h 대기 후 23분 실행. 과제당 556~1380s. 제출 ref 56652265 |
| 2026-09-29 | v1 | v0 + 남는 시간에 vote margin 낮은 과제부터 새 seed 로 재풀이 (2차 패스), 투표 합산 | debug 4개: 3.0/4 | pending | 2차 패스 정상 동작 (debug 에선 4개 모두 재풀이). 과제당 시간은 1차와 비슷 (534~1231s). 제출 ref 56657059 |
