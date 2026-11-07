"""joint 탐색 v2: 초평면(낭비 방향) 을 정확한 parity 여유 (max - Σdim/2) 로 탐욕 선택.
Usage: uv run python src/joint_search2.py <npz> <k> <z0..z4> [--samples 16] [--seed 0] [--out plan.pkl]"""
import sys, time, random, pickle, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from loader_tensor import out_bits, delta, mixed_tensor
from joint_planner import State, gf_rank, parity_select, random_nonjunk, NO, _np
from block_planner import Basis, nullspace, span_elems

def all_cands(st):
    nb = st.nb
    return [(i, ui, j, uj, f) for i in range(nb) for j in range(i, nb) for (ui, uj, f) in st.pair_cands(i, j)]

def parity_max(st, rng, cands=None):
    cands = all_cands(st) if cands is None else cands
    nb = st.nb; offs = []; n = 0; Sb = []
    for i in range(nb):
        S = nullspace(st.Y[i], 3 * NO); b = Basis()
        for kk, v in enumerate(S): b.add(v, 1 << kk)
        Sb.append(b); offs.append(n); n += len(S)
    M = _np.zeros((n, n), _np.int64)
    for (i, ui, j, uj, f) in cands:
        x = rng.randrange(1, 65536)
        ia = []; ib = []
        for (blk, u, out) in ((i, ui, ia), (j, uj, ib)):
            g = 0
            for r in range(3):
                if (u >> r) & 1: g |= f << (r * NO)
            rem, combo = Sb[blk].reduce(g)
            combo <<= offs[blk]
            while combo:
                b_ = combo & -combo; out.append(b_.bit_length() - 1); combo ^= b_
        ia = _np.array(ia); ib = _np.array(ib)
        M[_np.ix_(ia, ib)] ^= x; M[_np.ix_(ib, ia)] ^= x
    return gf_rank(M) // 2, n, len(cands)

def snapshot(st, i):
    return (list(st.Y[i]), list(st.extra[i]), {u: st.cons[i, u] for u in range(1, 8)})
def restore(st, i, snap):
    st.Y[i], st.extra[i] = list(snap[0]), list(snap[1])
    for u in range(1, 8): st.cons[i, u] = snap[2][u]

def propose(st, i, rng):
    """후보 초평면 벡터: 절반은 무작위, 절반은 다른 블록 j 의 K_{u'} 원소 f 로 만든 u⊗f."""
    if rng.random() < 0.5:
        return random_nonjunk(rng, st.J[i])
    j = rng.randrange(st.nb); uj = rng.randrange(1, 8); ui = rng.randrange(1, 8)
    ns = nullspace(st.cons[j, uj], NO)
    if not ns: return random_nonjunk(rng, st.J[i])
    f = 0
    for v in ns:
        if rng.random() < 0.5: f ^= v
    if f == 0: f = ns[0]
    g = 0
    for r in range(3):
        if (ui >> r) & 1: g |= f << (r * NO)
    if g & st.J[i]: return random_nonjunk(rng, st.J[i])
    return g

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("k", type=int); ap.add_argument("z")
    ap.add_argument("--samples", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out")
    ap.add_argument("--blocks", default="1,3,7/2,6,14/4,5,13/8,9,10/11,12,15"); ap.add_argument("--uncoded", action="store_true")
    a = ap.parse_args()
    W = np.load(a.npz)["angle_words"]
    T, dc, c = mixed_tensor(out_bits(delta(W, a.k)))
    mono = [list(map(int, b.split(","))) for b in a.blocks.split("/")]
    zs = list(map(int, a.z.split(",")))
    rng = random.Random(a.seed); t0 = time.time()
    st = State(T, mono, uncoded=a.uncoded)
    mx, n, nc = parity_max(st, rng)
    print(f"start dims {[st.dimS(i) for i in range(st.nb)]} max {mx} need {n//2} slack {mx - n/2}", flush=True)
    steps = [(i) for s in range(max(zs)) for i in range(st.nb) if zs[i] > s]
    for i in steps:
        snap = snapshot(st, i); best = None
        for _ in range(a.samples):
            z = propose(st, i, rng)
            if not st.add_hyperplane(i, z): continue
            mx, n, nc = parity_max(st, rng)
            sc = mx - n / 2
            if best is None or sc > best[0]: best = (sc, z, mx, n, nc)
            restore(st, i, snap)
        st.add_hyperplane(i, best[1])
        print(f"  block {i}: dims {[st.dimS(q) for q in range(st.nb)]} max {best[2]} need {best[3]/2} slack {best[0]} cands {best[4]} ({time.time()-t0:.0f}s)", flush=True)
    mx, n, nc = parity_max(st, rng)
    print(f"final dims {[st.dimS(i) for i in range(st.nb)]} sum {n} max {mx} need {n/2}", flush=True)
    if 2 * mx == n and a.out:
        cands = all_cands(st)
        sel, _, _ = parity_select(st, cands, rng, log=lambda x: None)
        pickle.dump(dict(mono=mono, Y=st.Y, extra=st.extra, sel=sel, k=a.k), open(a.out, "wb"))
        print("PERFECT saved", a.out, len(sel), "products", flush=True)
