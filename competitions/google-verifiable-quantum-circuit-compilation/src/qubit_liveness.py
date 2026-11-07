"""scratch qubit 의 보수적 수명 구간(R/HMR 로 |0> 이 된 시점 ~ 다음 사용 시작) 을 계산해 동시 활성 최대치를 구한다.
최대치 < 현재 scratch 수 이면 qubit 재배치(구간 그래프 색칠)로 Q 를 줄일 수 있다.
가정: R q = |0> 로 리셋, HMR q b = 측정 후 |0> 로 리셋. 그 외 연산은 qubit 을 사용(live)한다.
Usage: python3 src/qubit_liveness.py <file.kmx> <register names to exclude, comma>"""
import re, sys, collections
f = sys.argv[1]; excl_regs = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set()
qre = re.compile(r"\bq(\d+)")
reg_q = collections.defaultdict(list); cur_reg = []
ops = []
excl = set()
# first pass: registers
with open(f) as fh:
    pending = []
    for line in fh:
        if line.startswith("APPEND_TO_REGISTER"):
            p = line.split(); pending.append((p[1], p[2]))
        elif line.startswith("REGISTER"):
            p = line.split(); name = p[2].strip('"')
            rid = p[1]
            if name in excl_regs:
                for q, r in pending:
                    if r == rid and q.startswith("q"): excl.add(int(q[1:]))
            pending = [x for x in pending if x[1] != rid]
# second pass: intervals
start = {}       # qubit -> index where current lifetime began (first non-reset use)
free = {}        # qubit -> True if currently known |0>
events = []      # (start, end) intervals
nq = 0; idx = 0
last_use = {}
with open(f) as fh:
    for line in fh:
        if line.startswith(("REGISTER", "APPEND", "INFO")): continue
        idx += 1
        p = line.split(); op = p[0]
        qs = [int(x) for x in qre.findall(line.split(" if ")[0])]
        for q in qs: nq = max(nq, q + 1)
        if op in ("R", "HMR"):
            q = qs[0]
            if q in excl: continue
            if q in start:  # close lifetime
                events.append((start.pop(q), idx, q))
            free[q] = True
            continue
        for q in qs:
            if q in excl: continue
            if q not in start:
                start[q] = idx  # lifetime begins (initially |0> or after reset)
            last_use[q] = idx
for q, s in start.items():
    events.append((s, idx + 1, q))
pts = []
for s, e, q in events:
    pts.append((s, 1)); pts.append((e, -1))
pts.sort(key=lambda x: (x[0], x[1]))
c = m = 0
for t, d in pts:
    c += d; m = max(m, c)
scratch = sorted({q for _, _, q in events})
print(f"qubits total {nq}, excluded {len(excl)}, scratch used {len(scratch)}, max simultaneously live (conservative) {m}")
