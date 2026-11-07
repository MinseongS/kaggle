"""joint 국소 탐색: 블록별 낭비 벡터(extra) 집합을 바꿔가며 parity 결손 최소화.
시작: 기존 plan pkl (extra) 에서 지정 블록의 벡터를 빼서 목표 z 로 맞추거나, 무작위.
Usage: uv run python src/joint_local.py <npz> <k> <start.pkl|none> <z0..z4> <iters> <seed> <out.pkl>"""
import sys, time, random, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from loader_tensor import out_bits, delta, mixed_tensor, gf2rank
from joint_planner import State, parity_select, NO
from joint_search2 import parity_max, propose, all_cands

npz, k, start, zarg, iters, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
W = np.load(npz)["angle_words"]
T, dc, c = mixed_tensor(out_bits(delta(W, k)))
UNC = gf2rank(T[:, :, 0]) == 15
rng = random.Random(seed)
zs = list(map(int, zarg.split(",")))
mono = [[1, 3, 7], [2, 6, 14], [4, 5, 13], [8, 9, 10], [11, 12, 15]]
base = State(T, mono, uncoded=UNC)
if start != "none":
    P = pickle.load(open(start, "rb")); mono = P["mono"]; base = State(T, mono, uncoded=UNC)
    extra = [list(e) for e in P["extra"]]
else:
    extra = [[] for _ in mono]
# 목표 z 로 맞춤 (초과분은 무작위 제거, 부족분은 propose)
def build(extra):
    st = State(T, mono, uncoded=UNC)
    ok = True
    for i, ex in enumerate(extra):
        for z in ex:
            if not st.add_hyperplane(i, z): ok = False
    return st, ok
for i in range(len(mono)):
    while len(extra[i]) > zs[i]: extra[i].pop(rng.randrange(len(extra[i])))
st, _ = build(extra)
for i in range(len(mono)):
    while len(extra[i]) < zs[i]:
        z = propose(st, i, rng)
        if st.add_hyperplane(i, z): extra[i].append(z)
st, ok = build(extra)
mx, n, nc = parity_max(st, rng)
cur = mx - n / 2
print(f"start: dims {[st.dimS(i) for i in range(st.nb)]} sum {n} max {mx} slack {cur} cands {nc}", flush=True)
t0 = time.time()
for it in range(iters):
    if cur == 0: break
    blks = [i for i in range(len(mono)) if extra[i]]
    i = rng.choice(blks); r = rng.randrange(len(extra[i]))
    old = extra[i][r]
    # 다른 벡터들로 상태 구성 후 새 벡터 제안
    tmp = [list(e) for e in extra]; tmp[i].pop(r)
    st2, _ = build(tmp)
    z = propose(st2, i, rng)
    if not st2.add_hyperplane(i, z): continue
    mx, n, nc = parity_max(st2, rng)
    sc = mx - n / 2
    if sc >= cur:
        tmp[i].append(z); extra = tmp
        if sc > cur: print(f"  it {it}: slack {sc} max {mx} cands {nc} ({time.time()-t0:.0f}s)", flush=True)
        cur = sc; st = st2
if cur == 0:
    cands = all_cands(st)
    sel, _, _ = parity_select(st, cands, rng, log=lambda x: None)
    pickle.dump(dict(mono=mono, Y=st.Y, extra=extra, sel=sel, k=k), open(out, "wb"))
    print("PERFECT", len(sel), "products saved", out, flush=True)
else:
    pickle.dump(dict(mono=mono, Y=st.Y, extra=extra, sel=None, k=k), open(out + ".partial", "wb"))
    print("final slack", cur, flush=True)
