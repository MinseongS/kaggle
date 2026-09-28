# Experiments

| date | exp | change | CV | LB | note |
|---|---|---|---|---|---|
| 2026-09-29 | v0 | NVARC perfpatch + 결정성(crc32, PYTHONHASHSEED=0) + 과제별 try/except + TTT 마감 가드 + cheap-first + identity fallback | debug 4개: 2.5/4 | pending | commit run 은 4×L4 대기열에서 약 13h 대기 후 23분 실행. 과제당 556~1380s. 제출 ref 56652265 |
