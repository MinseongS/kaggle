"""joint_planner 탐색 드라이버: 블록별 z (낭비 차원) 를 정하고 초평면을 후보수 휴리스틱으로 키운 뒤
정확한 matroid parity 최대치를 측정. 완전 (max == Σdim/2) 이면 해를 저장.
Usage: uv run python src/joint_search.py <npz> <k> <z0,z1,z2,z3,z4> <seeds> [--samples 30] [--out plan.pkl]"""
import sys, time, random, pickle, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from loader_tensor import out_bits, delta, mixed_tensor
from joint_planner import State, grow, parity_select, gf_rank, NO
ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("k", type=int); ap.add_argument("z"); ap.add_argument("seeds", type=int)
ap.add_argument("--samples", type=int, default=30); ap.add_argument("--out"); ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--blocks", default="1,3,7/2,6,14/4,5,13/8,9,10/11,12,15")
a = ap.parse_args()
W = np.load(a.npz)["angle_words"]
T, dc, c = mixed_tensor(out_bits(delta(W, a.k)))
mono = [list(map(int, b.split(","))) for b in a.blocks.split("/")]
zs = list(map(int, a.z.split(",")))
best = None
for s in range(a.seed0, a.seed0 + a.seeds):
    t = time.time(); rng = random.Random(s)
    st = State(T, mono)
    for step in range(max(zs)):
        for i in range(len(mono)):
            if zs[i] > step: grow(st, i, a.samples, rng, log=lambda x: None)
    dims = [st.dimS(i) for i in range(len(mono))]
    cands = [(i, ui, j, uj, f) for i in range(len(mono)) for j in range(i, len(mono)) for (ui, uj, f) in st.pair_cands(i, j)]
    # 빠른 최대치만 (삭제법 없이)
    from joint_planner import _np
    sel = None
    import joint_planner as jp
    # parity_select 는 삭제까지 하므로, 먼저 rank 만
    nb = len(mono); offs = []; n = 0; Sb = []
    from block_planner import Basis, nullspace
    for i in range(nb):
        S = nullspace(st.Y[i], 3 * NO); b = Basis()
        for kk, v in enumerate(S): b.add(v, 1 << kk)
        Sb.append(b); offs.append(n); n += len(S)
    def coord(i, u, f):
        g = 0
        for r in range(3):
            if (u >> r) & 1: g |= f << (r * NO)
        rem, combo = Sb[i].reduce(g); assert rem == 0
        return combo << offs[i]
    M = _np.zeros((n, n), _np.int64)
    def bl(v):
        o = []
        while v:
            b_ = v & -v; o.append(b_.bit_length() - 1); v ^= b_
        return _np.array(o)
    for (i, ui, j, uj, f) in cands:
        x = rng.randrange(1, 65536); ia, ib = bl(coord(i, ui, f)), bl(coord(j, uj, f))
        M[_np.ix_(ia, ib)] ^= x; M[_np.ix_(ib, ia)] ^= x
    mx = gf_rank(M) // 2
    print(f"seed {s}: dims {dims} sum {sum(dims)} need {sum(dims)//2} cands {len(cands)} max {mx} ({time.time()-t:.1f}s)", flush=True)
    if 2 * mx == sum(dims):
        sel, mx2, nn = parity_select(st, cands, rng, log=lambda x: None)
        print(f"  PERFECT: {len(sel)} products", flush=True)
        if a.out:
            pickle.dump(dict(mono=mono, Y=st.Y, extra=st.extra, sel=sel, k=a.k), open(a.out, "wb"))
            print("  saved", a.out)
        break
