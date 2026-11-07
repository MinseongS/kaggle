"""kmx 텍스트에서 큐비트별 첫/마지막 사용 줄, 사용 횟수, 구간 수(BIT_STORE/HMR 등 해제 기준이 아닌 단순 접근) 집계.
Usage: uv run python src/qubit_spans.py <file.kmx> <out.npz>"""
import re, sys, numpy as np
qre = re.compile(rb"\bq(\d+)")
first = {}; last = {}; cnt = {}
tags = []
with open(sys.argv[1], "rb") as f:
    for i, l in enumerate(f):
        if l.startswith(b"INFO_TAG"):
            tags.append((i, l.strip().decode())); continue
        if l.startswith((b"REGISTER", b"APPEND")): continue
        for m in qre.findall(l):
            q = int(m)
            if q not in first: first[q] = i
            last[q] = i; cnt[q] = cnt.get(q, 0) + 1
qs = sorted(first)
np.savez(sys.argv[2], q=np.array(qs), first=np.array([first[q] for q in qs]), last=np.array([last[q] for q in qs]), cnt=np.array([cnt[q] for q in qs]), nlines=i + 1)
import json; json.dump(tags, open(sys.argv[2] + ".tags.json", "w"))
print("qubits", len(qs), "lines", i + 1)
