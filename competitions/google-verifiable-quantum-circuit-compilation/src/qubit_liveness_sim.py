"""시뮬레이션 기반 scratch 생존 분석: 회로를 조각내 실행하며 조각 경계마다 scratch 값을 읽는다.
조각 안에서 qubit 은 (경계에서 nonzero) 또는 (조각 내에서 접근됨) 이면 live 로 본다 (조각 내 일시 사용 포함, 보수적).
Usage: uv run python src/qubit_liveness_sim.py <file.kmx> <task: invsqrt> [--chunk 40] [--shots 2048]"""
import argparse, re, sys
from pathlib import Path
import numpy as np, kickmix as km
ap = argparse.ArgumentParser(); ap.add_argument("kmx"); ap.add_argument("task"); ap.add_argument("--chunk", type=int, default=40); ap.add_argument("--shots", type=int, default=2048); ap.add_argument("--refine", type=int, default=0, help="이전 .live.npy 에서 live>=refine 인 조각을 50연산 단위로 세분")
a = ap.parse_args()
lines = [l.rstrip("\n") for l in open(a.kmx)]
decl = [l for l in lines if l.startswith(("REGISTER", "APPEND"))]
body = [l for l in lines if l and not l.startswith(("REGISTER", "APPEND", "INFO"))]
full = km.Circuit("\n".join(lines))
regs = full.register_data
nq = full.num_qubits
io = set()
for name, arr in regs.items():
    for e in arr:
        if isinstance(e, km.q): io.add(int(str(e)[1:]) if str(e).startswith("q") else None)
scratch = [q for q in range(nq) if q not in io]
print("qubits", nq, "io", len(io), "scratch", len(scratch), flush=True)
qre = re.compile(r"\bq(\d+)")
# chunks at condition depth 0
chunks, cur, depth = [], [], 0
for l in body:
    cur.append(l)
    op = l.split()[0]
    if op == "PUSH_CONDITION": depth += 1
    if op == "POP_CONDITION": depth -= 1
    if depth == 0 and len(cur) >= a.chunk:
        chunks.append(cur); cur = []
if cur: chunks.append(cur)
sim = km.Simulator(batch_size=a.shots, debug_print=False, counting_operations=False, seed=1)
sim.clear_for_shot()
if a.task == "invsqrt":
    rng = np.random.default_rng(0)
    MAXX = 3 * (2**10 - 2) ** 2
    xs = np.concatenate([np.arange(1, 513), rng.integers(1, MAXX + 1, a.shots - 1024), np.arange(MAXX - 511, MAXX + 1)]).astype(np.uint64)
    xr = km.array(list(regs["x"]))
    sim.write_across_shots(xr, ((xs[None, :] >> np.arange(22, dtype=np.uint64)[:, None]) & 1) == 1)
if a.task == "dqi":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data/verifiers"))
    from dqi.verify_submission_dqi import ProblemSpecDQI, optimal_dqi_weights, FieldTables, sample_shots, to_bits, validate_registers
    spec = ProblemSpecDQI.benchmark_instance()
    syn_q, mb_b, anc = validate_registers(full, spec)
    wp = optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    syn, mb, ph = sample_shots(spec, FieldTables(km.GF2Field(spec.degree)), np.random.default_rng(0), a.shots, wp)
    sim.use_same_registers_as(full)
    sim.write_across_shots(km.array(syn_q), to_bits(syn, spec.degree))
    sim.write_across_shots(km.array(mb_b), to_bits(mb, spec.degree))
S = km.array([km.q(q) for q in scratch])
live_counts = []; per_chunk = []
prev_nz = np.zeros(len(scratch), bool)
pos = {q: i for i, q in enumerate(scratch)}
prevlive = np.load(Path(a.kmx).with_suffix(".live.npy")).sum(1) if a.refine else None
def sub(ch, n=50):
    out, cur, dep = [], [], 0
    for l in ch:
        cur.append(l); op = l.split()[0]
        if op == "PUSH_CONDITION": dep += 1
        if op == "POP_CONDITION": dep -= 1
        if dep == 0 and len(cur) >= n: out.append(cur); cur = []
    if cur: out.append(cur)
    return out
expanded = []
for ci, ch in enumerate(chunks):
    if prevlive is not None and prevlive[ci] >= a.refine: expanded.extend(sub(ch))
    else: expanded.append(ch)
chunks = expanded
for ch in chunks:
    touched = np.zeros(len(scratch), bool)
    for l in ch:
        for qq in qre.findall(l.split(" if ")[0]):
            qq = int(qq)
            if qq in pos: touched[pos[qq]] = True
    sim.do(km.Circuit("\n".join(ch)))
    nz = sim.read_across_shots(S).any(axis=1)
    live = prev_nz | nz | touched
    per_chunk.append(live); live_counts.append(int(live.sum()))
    prev_nz = nz
lc = np.array(live_counts)
print("chunks", len(chunks), "max live", lc.max(), "at chunk", int(lc.argmax()), "chunks at max:", int((lc == lc.max()).sum()))
print("histogram of live count (top):", np.unique(lc, return_counts=True)[0][-6:], np.unique(lc, return_counts=True)[1][-6:])
if not a.refine: np.save(Path(a.kmx).with_suffix(".live.npy"), np.array(per_chunk))
P = np.array(per_chunk); L = np.array([len(c) for c in chunks])
print("per-qubit live op-fraction (lowest 8):", sorted(zip(((P * L[:, None]).sum(0) / L.sum()).round(3), scratch))[:8])
