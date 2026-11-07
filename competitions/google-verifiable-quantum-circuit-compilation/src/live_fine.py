"""dqi kmx: [lo, hi) 줄 구간을 한 줄씩 시뮬레이션하며 스크래치 중 (어느 shot 에서든) 0 이 아닌 큐비트 수의 최대를 측정.
Usage: uv run python src/live_fine.py <dqi.kmx> <lo> <hi> [--shots 256] [--nq 892]"""
import sys, argparse, time
from pathlib import Path
import numpy as np, kickmix as km
ap = argparse.ArgumentParser(); ap.add_argument("kmx"); ap.add_argument("lo", type=int); ap.add_argument("hi", type=int)
ap.add_argument("--shots", type=int, default=256); ap.add_argument("--nq", type=int, default=892); ap.add_argument("--step", type=int, default=1)
a = ap.parse_args()
decl = []
with open(a.kmx) as fh:
    for l in fh:
        if l.startswith(("REGISTER", "APPEND")): decl.append(l.rstrip("\n"))
        else: break
hdr = km.Circuit("\n".join(decl))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data/verifiers"))
from dqi.verify_submission_dqi import ProblemSpecDQI, optimal_dqi_weights, FieldTables, sample_shots, to_bits
spec = ProblemSpecDQI.benchmark_instance(); regs = hdr.register_data
syn_q = [e for i in range(spec.n) for e in regs[f"S{i}"]]; mb_b = [e for i in range(spec.m) for e in regs[f"E{i}"]]
sim = km.Simulator(batch_size=a.shots, debug_print=False, counting_operations=False, seed=1)
sim.clear_for_shot(); sim.use_same_registers_as(hdr)
wp = optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
syn, mb, ph = sample_shots(spec, FieldTables(km.GF2Field(spec.degree)), np.random.default_rng(0), a.shots, wp)
sim.write_across_shots(km.array(syn_q), to_bits(syn, spec.degree)); sim.write_across_shots(km.array(mb_b), to_bits(mb, spec.degree))
S = km.array([km.q(q) for q in range(840, a.nq)])
t0 = time.time()
pre = []; body = []
with open(a.kmx) as fh:
    for i, l in enumerate(fh):
        if l.startswith(("REGISTER", "APPEND")): continue
        if i < a.lo:
            pre.append(l.rstrip("\n"))
            if len(pre) >= 200000: sim.do(km.Circuit("\n".join(pre))); pre = []
        elif i < a.hi: body.append(l.rstrip("\n"))
        else: break
if pre: sim.do(km.Circuit("\n".join(pre)))
print("prefix done", time.time() - t0, flush=True)
mx = 0; hist = {}; cur = []; depth = 0; at = None
for j, l in enumerate(body):
    cur.append(l); op = l.split(" ", 1)[0]
    if op == "PUSH_CONDITION": depth += 1
    elif op == "POP_CONDITION": depth -= 1
    if depth or len(cur) < a.step: continue
    sim.do(km.Circuit("\n".join(cur))); cur = []
    n = int(sim.read_across_shots(S).any(axis=1).sum())
    hist[n] = hist.get(n, 0) + 1
    if n > mx: mx = n; at = a.lo + j
print("max nonzero scratch", mx, "at line", at, "hist top", sorted(hist.items())[-4:], time.time() - t0)
