"""kmx 스크래치 큐비트의 수명 구간 (R/첫 사용 ~ HMR 또는 다음 R 전 마지막 사용) 을 구하고 최대 겹침 계산.
Usage: uv run python src/qubit_intervals.py <file.kmx> <first_scratch> [--dump out.npz]"""
import re, sys, numpy as np
f = sys.argv[1]; FS = int(sys.argv[2])
qre = re.compile(rb"\bq(\d+)")
start = {}; last = {}; intervals = []
with open(f, "rb") as fh:
    for i, l in enumerate(fh):
        if l.startswith((b"REGISTER", b"APPEND", b"INFO")): continue
        op = l.split(b" ", 1)[0]
        head = l.split(b" if ")[0]
        qs = [int(m) for m in qre.findall(head)]
        for q in qs:
            if q < FS: continue
            if op == b"R":
                if q in start: intervals.append((start[q], last[q], q))
                start[q] = i; last[q] = i
                continue
            if q not in start: start[q] = i
            last[q] = i
            if op == b"HMR":
                intervals.append((start[q], i, q)); del start[q]; del last[q]
for q in start: intervals.append((start[q], last[q], q))
iv = np.array(intervals)
ev = np.concatenate([np.stack([iv[:, 0], np.ones(len(iv), int)], 1), np.stack([iv[:, 1] + 1, -np.ones(len(iv), int)], 1)])
ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]  # 같은 시점: 종료(-1) 먼저
cur = np.cumsum(ev[:, 1])
mx = cur.max(); where = ev[np.argmax(cur), 0]
print("intervals", len(iv), "distinct scratch", len(set(iv[:, 2].tolist())), "max overlap", mx, "first at line", where, "count of maximal points", int((cur == mx).sum()))
if "--dump" in sys.argv: np.savez(sys.argv[sys.argv.index("--dump") + 1], iv=iv)
