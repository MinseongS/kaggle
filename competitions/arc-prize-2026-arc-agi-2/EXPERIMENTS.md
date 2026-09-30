# Experiments

| date | exp | change | CV | LB | note |
|---|---|---|---|---|---|
| 2026-09-29 | v0 | NVARC perfpatch + 결정성(crc32, PYTHONHASHSEED=0) + 과제별 try/except + TTT 마감 가드 + cheap-first + identity fallback | debug 4개: 2.5/4 | **28.89** (약 1000등) | 공개 fork 분포 27~33 의 아래쪽. commit run 은 4×L4 대기열에서 약 13h 대기 후 23분 실행. 과제당 556~1380s. 제출 ref 56652265 |
| 2026-09-29 | v1 | v0 + 남는 시간에 vote margin 낮은 과제부터 새 seed 로 재풀이 (2차 패스), 투표 합산 | debug 4개: 3.0/4 | **29.72** | 2차 패스 정상 동작 (debug 에선 4개 모두 재풀이). 과제당 시간은 1차와 비슷 (534~1231s). 제출 ref 56657059 |
| 2026-09-30 | v2 | v1 + 8-view same-shape decode batches. 입력 경로를 /kaggle/input 탐색으로 찾게 수정 | pending | pending | 9/29 첫 commit(v2, v2-eval)은 하드코딩된 대회 경로 FileNotFoundError 로 추론 0건 (v1 은 같은 경로로 정상). 9/30 재push: v2 ver2, v2-eval ver2 (eval 120 task, 6h) |
| 2026-09-30 | v2-eval | eval commit 에서 /kaggle/inference_outputs 를 inference_outputs.tar 로 저장 (오프라인 선택 규칙 튜닝용, src/rescore.py) | pending | — | ver2(대기열)는 취소, ver3 push. GPU batch 세션 동시 2개 제한 있음 |
| 2026-09-30 | v3 | v2 + icecuber CPU 탐색 (GPU 셀 직전 detach, nice 15, 16 worker, depth3+flip 2종) → merge_cpu 규칙으로 빈/약한 attempt 채움. 셀 전부 try/except | pending | — | 로컬 e2e 확인 (depth2 20s, 빈 submission → 1.0). v2-eval ver3 취소 후 v3-eval ver1 push: submission_nvarc.json + inference_outputs.tar 도 남기므로 v2 NVARC-only 측정 겸함 |
