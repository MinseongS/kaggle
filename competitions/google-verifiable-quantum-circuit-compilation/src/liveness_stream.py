"""스트리밍 scratch 생존 분석 (큰 dqi kmx 용). 지정 줄 구간에서는 작은 조각, 밖에서는 큰 조각으로 시뮬레이션.
조각 live = (경계에서 nonzero) 또는 (조각 안에서 접근). 결과 npz: 조각 시작줄, 끝줄, live 행렬.
Usage: uv run python src/liveness_stream.py <dqi.kmx> <out.npz> --lo L --hi H [--fine 500] [--coarse 50000] [--shots 256] [--first-scratch 840]"""
import argparse, re, sys
from pathlib import Path
import numpy as np, kickmix as km
ap = argparse.ArgumentParser(); ap.add_argument("kmx"); ap.add_argument("out")
ap.add_argument("--lo", type=int, default=0); ap.add_argument("--hi", type=int, default=10**12)
ap.add_argument("--fine", type=int, default=500); ap.add_argument("--coarse", type=int, default=50000)
ap.add_argument("--shots", type=int, default=256); ap.add_argument("--first-scratch", type=int, default=840)
a = ap.parse_args()
decl = []
with open(a.kmx) as f:
    for l in f:
        if l.startswith(("REGISTER", "APPEND")): decl.append(l.rstrip("\n"))
        else: break
hdr = km.Circuit("\n".join(decl))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data/verifiers"))
from dqi.verify_submission_dqi import ProblemSpecDQI, optimal_dqi_weights, FieldTables, sample_shots, to_bits
spec = ProblemSpecDQI.benchmark_instance()
regs = hdr.register_data
syn_q = [e for i in range(spec.n) for e in regs[f"S{i}"]]
mb_b = [e for i in range(spec.m) for e in regs[f"E{i}"]]
sim = km.Simulator(batch_size=a.shots, debug_print=False, counting_operations=False, seed=1)
sim.clear_for_shot(); sim.use_same_registers_as(hdr)
wp = optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
syn, mb, ph = sample_shots(spec, FieldTables(km.GF2Field(spec.degree)), np.random.default_rng(0), a.shots, wp)
sim.write_across_shots(km.array(syn_q), to_bits(syn, spec.degree))
sim.write_across_shots(km.array(mb_b), to_bits(mb, spec.degree))
ap2 = None
NQ = int(__import__('os').environ.get('NQ', '892'))
scratch = list(range(a.first_scratch, NQ + 0))
S = None
qre = re.compile(r"\bq(\d+)")
starts, ends, lives = [], [], []
prev_nz = None
cur, depth, start = [], 0, None
def flush(lines, s, e):
    global prev_nz, S
    sim.do(km.Circuit("\n".join(lines)))
    sc = [km.q(q) for q in scratch]
    nz = sim.read_across_shots(km.array(sc)).any(axis=1)
    touched = np.zeros(len(scratch), bool)
    for l in lines:
        for qq in qre.findall(l.split(" if ")[0]):
            qq = int(qq) - a.first_scratch
            if 0 <= qq < len(scratch): touched[qq] = True
    if prev_nz is None: prev_nz = np.zeros(len(scratch), bool)
    live = prev_nz | nz | touched
    prev_nz = nz
    starts.append(s); ends.append(e); lives.append(live)
with open(a.kmx) as f:
    for i, l in enumerate(f):
        if l.startswith(("REGISTER", "APPEND")): continue
        if start is None: start = i
        cur.append(l.rstrip("\n")); op = l.split(" ", 1)[0]
        if op == "PUSH_CONDITION": depth += 1
        elif op == "POP_CONDITION": depth -= 1
        lim = a.fine if a.lo <= i < a.hi else a.coarse
        if depth == 0 and (len(cur) >= lim or i + 1 == a.lo or i + 1 == a.hi):
            flush(cur, start, i); cur = []; start = None
            if len(starts) % 200 == 0: print(i, int(lives[-1].sum()), flush=True)
if cur: flush(cur, start, i)
L = np.array(lives)
np.savez(a.out, starts=np.array(starts), ends=np.array(ends), live=L)
lc = L.sum(1)
print("chunks", len(L), "max live", lc.max(), "count at max", int((lc == lc.max()).sum()))
