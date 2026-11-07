"""chem loader 공동 계획기: 35 출력 (bit0 포함) 을 블록 쌍 곱으로 한 번에 덮는다 (자체 구현).

블록 i 의 목표 S_i ⊇ W_i 를 W_i 보다 z_i 차원 크게 잡는다 (S_i^⊥ = Y_i ⊂ W_i^⊥, 정크 좌표 J_i 는 항상 포함).
J_i 를 포함하는 초평면을 하나 고를 때마다 모든 u 의 K_u = {f : u⊗f ∈ S_i} 가 1 차원씩 커진다.
어떤 초평면을 고를지는 다른 블록과의 짝 후보 수가 최대가 되도록 표본 탐색.
곱 = 두 생성자 (블록 i 의 u⊗f, 블록 j 의 u'⊗f; i==j 허용) 이고, 블록마다 생성자들이 S_i 의 기저가 되면 완료.
곱 수 = Σ dim S_i / 2.
"""
from __future__ import annotations
import itertools, random
from block_planner import Basis, nullspace, span_elems

NO = 35


def junk_mask(block_monos, singletons=(1, 2, 4, 8)):
    """bit0 출력 좌표 중 T 가 구조적으로 0 인 곳 (단일변수가 아닌 행)."""
    m = 0
    for r, mono in enumerate(block_monos):
        if mono not in singletons:
            for j in range(5): m |= 1 << (r * NO + 7 * j)
    return m


def block_cols(T, block_monos, absorb=True):
    """블록 열 벡터 76개. h=75 (high 단항식 76) 는 m>=4 행이 don't care: absorb 이면 그 자유 좌표를
    다른 75열의 결합과 일치하도록 골라 열을 span 안으로 넣는다 (가능할 때)."""
    cols = []
    for h in range(76):
        v = 0
        for r, mono in enumerate(block_monos):
            li = mono - 1
            if h == 75 and mono >= 4: continue
            for o in range(NO):
                if T[li, h, o]: v |= 1 << (r * NO + o)
        cols.append(v)
    if absorb:
        fixed = 0
        for r, mono in enumerate(block_monos):
            if mono < 4: fixed |= ((1 << NO) - 1) << (r * NO)
        if fixed:
            b = Basis()
            for k, cv in enumerate(cols[:75]): b.add(cv & fixed, 1 << k)
            rem, combo = b.reduce(cols[75] & fixed)
            if rem == 0:
                nv = 0; k = 0
                while combo:
                    if combo & 1: nv ^= cols[k]
                    combo >>= 1; k += 1
                assert nv & fixed == cols[75] & fixed
                cols[75] = nv
        else:
            cols[75] = 0
    return cols


def block_W(T, block_monos):
    b = Basis(); [b.add(c) for c in block_cols(T, block_monos)]
    return b


def kcons(Y, u):
    mask = (1 << NO) - 1
    out = []
    for y in Y:
        c = 0
        for r in range(3):
            if (u >> r) & 1: c ^= (y >> (r * NO)) & mask
        out.append(c)
    return out


class State:
    def __init__(self, T, mono_blocks, uncoded=False):
        self.mono = mono_blocks
        self.nb = len(mono_blocks)
        self.W = [block_W(T, bl) for bl in mono_blocks]
        self.J = [0 if uncoded else junk_mask(bl) for bl in mono_blocks]
        self.Y = [nullspace([pv for pv, _ in w.piv.values()], 3 * NO) for w in self.W]
        self.extra = [[] for _ in mono_blocks]
        self.refresh_cons()

    def refresh_cons(self, which=None):
        if not hasattr(self, "cons"): self.cons = {}
        for i in (range(self.nb) if which is None else [which]):
            for u in range(1, 8):
                self.cons[i, u] = kcons(self.Y[i], u)

    def dimS(self, i):
        return 3 * NO - len(self.Y[i])

    def pair_cands(self, i, j, cap=None):
        lst = []
        for ui in range(1, 8):
            for uj in range(1, 8):
                if i == j and uj <= ui: continue
                ns = nullspace(self.cons[i, ui] + self.cons[j, uj], NO)
                if len(ns) > 7: ns = ns[:7]
                for f in span_elems(ns):
                    lst.append((ui, uj, f))
                    if cap and len(lst) >= cap: return lst
        return lst

    def count_cands(self, i):
        return sum(len(self.pair_cands(min(i, j), max(i, j))) for j in range(self.nb))

    def add_hyperplane(self, i, z):
        """S_i 에 벡터 z (정크 좌표 0) 추가 = Y_i 를 z^⊥ 로 제한."""
        assert z & self.J[i] == 0
        Y = self.Y[i]
        # Y ∩ z^⊥: z 와 내적 1 인 기저 하나를 피벗으로 나머지에서 소거
        piv = None; newY = []
        for y in Y:
            if bin(y & z).count("1") & 1:
                if piv is None: piv = y; continue
                newY.append(y ^ piv)
            else: newY.append(y)
        if piv is None: return False  # z ∈ S_i already
        self.Y[i] = newY; self.extra[i].append(z)
        self.refresh_cons(i)
        return True


def random_nonjunk(rng, J):
    v = 0
    for x in range(3 * NO):
        if not (J >> x) & 1 and rng.random() < 0.5: v |= 1 << x
    return v


def grow(st, i, samples, rng, log=print):
    """블록 i 에 초평면 하나 추가: 표본 중 후보 수 최대."""
    best = None
    Ysave = list(st.Y[i]); csave = {u: st.cons[i, u] for u in range(1, 8)}; esave = list(st.extra[i])
    for _ in range(samples):
        z = random_nonjunk(rng, st.J[i])
        if not st.add_hyperplane(i, z): continue
        sc = st.count_cands(i)
        if best is None or sc > best[0]: best = (sc, z)
        st.Y[i] = list(Ysave); st.extra[i] = list(esave)
        for u in range(1, 8): st.cons[i, u] = csave[u]
    st.add_hyperplane(i, best[1])
    log(f"    grow block {i}: dimS {st.dimS(i)} cands {best[0]}")


def pair_search(st, tries, rng, log=print):
    nb = st.nb
    dims = [st.dimS(i) for i in range(nb)]
    pairs = [(i, j) for i in range(nb) for j in range(i, nb)]
    cand = {p: st.pair_cands(*p) for p in pairs}
    log("  cands " + str({p: len(v) for p, v in cand.items()}) + f" dims {dims} sum {sum(dims)}")
    def gvec(u, f):
        v = 0
        for r in range(3):
            if (u >> r) & 1: v |= f << (r * NO)
        return v
    allc = [(i, j, c) for (i, j) in pairs for c in cand[i, j]]
    best = None
    for t in range(tries):
        rng.shuffle(allc)
        # 희소 블록 우선: 각 블록의 잔여 필요/후보 비율로 정렬 (약간의 무작위)
        B = [Basis() for _ in range(nb)]; chosen = []
        for sweep in range(3):
            for i, j, (ui, uj, f) in allc:
                if i == j:
                    if len(B[i]) + 2 > dims[i]: continue
                    gi, gj = gvec(ui, f), gvec(uj, f)
                    tb = Basis(); tb.piv = dict(B[i].piv)
                    if tb.add(gi) and tb.add(gj):
                        B[i].add(gi); B[i].add(gj); chosen.append((i, ui, j, uj, f))
                    continue
                if len(B[i]) >= dims[i] or len(B[j]) >= dims[j]: continue
                gi, gj = gvec(ui, f), gvec(uj, f)
                if B[i].reduce(gi)[0] and B[j].reduce(gj)[0]:
                    B[i].add(gi); B[j].add(gj); chosen.append((i, ui, j, uj, f))
        deficit = sum(d - len(b) for d, b in zip(dims, B))
        if best is None or deficit < best[0]:
            best = (deficit, chosen)
            log(f"    try {t}: products {len(chosen)} deficit {deficit}")
        if deficit == 0: break
    return best, dims


# ---------- exact linear matroid parity (Lovász 행렬, GF(2^16) 무작위 대입) ----------
import numpy as _np
_POLY = 0x1002D  # x^16 + x^5 + x^3 + x^2 + 1 (primitive)
_EXP = _np.zeros(2 * 65535, _np.int64); _LOG = _np.zeros(65536, _np.int64)
_x = 1
for _i in range(65535):
    _EXP[_i] = _x; _LOG[_x] = _i
    _x <<= 1
    if _x & 0x10000: _x ^= _POLY
_EXP[65535:] = _EXP[:65535]


def _gmul_vec(f, row):
    """스칼라 f (≠0) × 벡터 row (GF(2^16))"""
    out = _np.zeros_like(row)
    nz = row != 0
    out[nz] = _EXP[_LOG[f] + _LOG[row[nz]]]
    return out


def gf_rank(M):
    M = M.copy(); n, m = M.shape; r = 0
    for c in range(m):
        if r == n: break
        piv = _np.nonzero(M[r:, c])[0]
        if len(piv) == 0: continue
        p = r + piv[0]
        if p != r: M[[r, p]] = M[[p, r]]
        inv = _EXP[65535 - _LOG[M[r, c]]]
        M[r] = _gmul_vec(inv, M[r])
        col = M[:, c].copy(); col[r] = 0
        rows = _np.nonzero(col)[0]
        if len(rows):
            pr = M[r]; nzp = pr != 0
            lp = _LOG[pr[nzp]]
            lf = _LOG[col[rows]]
            prod = _EXP[lf[:, None] + lp[None, :]]
            sub = M[_np.ix_(rows, _np.nonzero(nzp)[0])]
            M[_np.ix_(rows, _np.nonzero(nzp)[0])] = sub ^ prod
        r += 1
    return r


def parity_select(st, cands, rng, log=print):
    """cands: [(i, ui, j, uj, f)]. 블록 S_i 좌표로 두 벡터를 만들고 Lovász 행렬 rank 로 최대 parity, 삭제법으로 해."""
    nb = st.nb
    # S_i 기저와 좌표 (combo 비트)
    offs = []; Sb = []; n = 0
    for i in range(nb):
        S = nullspace(st.Y[i], 3 * NO)
        b = Basis()
        for k, v in enumerate(S): b.add(v, 1 << k)
        Sb.append(b); offs.append(n); n += len(S)
    def coord(i, u, f):
        g = 0
        for r in range(3):
            if (u >> r) & 1: g |= f << (r * NO)
        rem, combo = Sb[i].reduce(g)
        assert rem == 0
        return combo << offs[i]
    vecs = [(coord(i, ui, f), coord(j, uj, f)) for (i, ui, j, uj, f) in cands]
    xs = [rng.randrange(1, 65536) for _ in cands]
    def bits(v):
        out = []
        while v:
            b = v & -v; out.append(b.bit_length() - 1); v ^= b
        return out
    E = []
    for (a, b) in vecs:
        ia, ib = _np.array(bits(a)), _np.array(bits(b))
        E.append((ia, ib))
    def add(M, p, x):
        ia, ib = E[p]
        M[_np.ix_(ia, ib)] ^= x; M[_np.ix_(ib, ia)] ^= x
    M = _np.zeros((n, n), _np.int64)
    for p in range(len(cands)): add(M, p, xs[p])
    r0 = gf_rank(M)
    log(f"  parity: n {n}, candidates {len(cands)}, max products {r0 // 2}")
    alive = list(range(len(cands)))
    order = list(range(len(cands))); rng.shuffle(order)
    for p in order:
        add(M, p, xs[p])
        if gf_rank(M) == r0: alive.remove(p)
        else: add(M, p, xs[p])
    return [cands[p] for p in alive], r0 // 2, n
