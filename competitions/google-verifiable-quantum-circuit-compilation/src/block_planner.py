"""chem angle loader 곱 계획기 (블록 쌍 방식, 자체 구현).

문제: 회로 k 의 델타 D (1220 주소 x 40 비트) 의 ANF 중 mixed 항 (low 4비트 단항식 L x high 단항식 H) 을
곱 (aL + aH)(bL + bH) 의 합으로 덮는다. 곱 하나의 mixed 기여 = fanout ⊗ (aL⊗bH + bL⊗aH).
low 15차원을 3차원 블록 5개로 나누면, 블록 i 의 생성자 u⊗f (u∈B_i, f∈출력) 들의 span 이 W_i
(= 각 high 열 T_h 의 블록 i 부분들의 span) 를 포함하면 된다. 곱 하나는 블록 i 의 생성자 하나와
블록 j 의 생성자 하나 (같은 f) 를 동시에 제공한다. f 는 K_u^(i) ∩ K_u'^(j) 에서 고른다.

출력 (하드) = 각 단어 bit1..6 (30개). bit0 (코딩된 경우 low 단항식 행이 단일 변수뿐) 은 별도 곱.
bit7 은 회로에서 bit6 과 helper(gauge) 로 처리 (emitter 담당).
"""
from __future__ import annotations
import itertools, random
import numpy as np


# ---------- GF(2) bitset linear algebra ----------
class Basis:
    """증분 가우스 소거 (int 비트셋). add() 는 독립이면 True."""
    def __init__(self):
        self.piv = {}  # pivot bit -> (vec, combo)
    def reduce(self, v, combo=0):
        while v:
            p = v.bit_length() - 1
            if p not in self.piv: return v, combo
            pv, pc = self.piv[p]; v ^= pv; combo ^= pc
        return 0, combo
    def add(self, v, combo=0):
        r, c = self.reduce(v, combo)
        if r == 0: return False
        self.piv[r.bit_length() - 1] = (r, c); return True
    def __len__(self): return len(self.piv)


def nullspace(rows, n):
    """rows: list of n-bit ints. 반환: {x : <row,x>=0 ∀row} 의 기저."""
    piv = {}
    for v in rows:
        while v:
            j = v.bit_length() - 1
            if j not in piv: piv[j] = v; break
            v ^= piv[j]
    # 완전 기약형
    keys = sorted(piv)
    for j in keys:
        for k in keys:
            if k != j and (piv[k] >> j) & 1: piv[k] ^= piv[j]
    out = []
    for j in range(n):
        if j in piv: continue
        v = 1 << j
        for k, row in piv.items():
            if (row >> j) & 1: v |= 1 << k
        out.append(v)
    return out


def span_elems(basis):
    out = [0]
    for b in basis: out += [x ^ b for x in out]
    return out[1:]


def popc(x): return bin(x).count("1")


# ---------- planner ----------
def plan(T, hard, blocks, edges_mult=None, seed=0, tries=200, log=print, extra=None, intra=False, ncols=76, boost=0):
    """T: (15, 76, n) uint8 mixed 텐서 (dc 열 h=75 는 low>=4 행 자유, 호출자가 넘긴 값 무시 가능).
    hard: 하드 출력 인덱스 리스트 (길이 no). blocks: 5개의 low 행 인덱스(0..14) 3-튜플.
    반환: products = [(li_block, u_i_rows(list of low idx), lj_block, u_j_rows, f_hard(int over hard)), ...] 와
    블록별 해 (high 계수) 는 solve_high 에서."""
    no = len(hard)
    nb = len(blocks)
    # 블록별 열 벡터: T_h 의 (r, o) -> r*no + o
    W = []; Wperp = []
    for B in blocks:
        cols = []
        for h in range(ncols):
            v = 0
            for r, li in enumerate(B):
                if h == 75 and li >= 3: continue  # don't care (주소 >= 1220) -> 0 으로 둔다
                for oi, o in enumerate(hard):
                    if T[li, h, o]: v |= 1 << (r * no + oi)
            cols.append(v)
        bas = Basis(); [bas.add(c) for c in cols]
        if extra is not None:
            for v in extra[len(W)]: bas.add(v)
        W.append(bas)
        # W^⊥: 기저 벡터들의 nullspace
        perp = nullspace([pv for pv, _ in bas.piv.values()], 3 * no)
        if boost:
            # Y = Σ_u ker(C_u) ⊂ W^⊥ 로 줄이면 모든 u 의 K_u 가 함께 커진다 (S = Y^⊥ ⊇ W, 낭비 = codim)
            mask_ = (1 << no) - 1
            Ks = Basis()
            for u in range(1, 8):
                imgs = []
                for y in perp:
                    cc = 0
                    for r in range(3):
                        if (u >> r) & 1: cc ^= (y >> (r * no)) & mask_
                    imgs.append(cc)
                # λ (len(perp) 비트) 중 Σ λ_k imgs_k = 0 인 것: 행렬 [imgs] 의 왼쪽 영공간
                nb_ = len(perp)
                rows = []  # 각 출력 좌표 o 에 대해 λ 에 대한 제약
                for o in range(no):
                    rv = 0
                    for kk in range(nb_):
                        if (imgs[kk] >> o) & 1: rv |= 1 << kk
                    rows.append(rv)
                for lam in nullspace(rows, nb_):
                    y = 0
                    for kk in range(nb_):
                        if (lam >> kk) & 1: y ^= perp[kk]
                    Ks.add(y)
            ky = [pv for pv, _ in Ks.piv.values()]
            extra_dims = len(perp) - len(ky); log(f"    block {len(W)-1}: perp {len(perp)} sumker {len(ky)} extra {extra_dims}")
            if 0 < extra_dims <= boost:
                perp = ky
                S = Basis(); [S.add(v) for v in nullspace(perp, 3 * no)]
                bas = S
                W[-1] = S
        Wperp.append(perp)
    dims = [len(b) for b in W]
    log(f"  block dims {dims} perp {[len(p) for p in Wperp]}")
    # K_u^(i): f 에 대한 제약 c_{y,u} = Σ_r u_r y[r,:]
    mask = (1 << no) - 1
    K = {}
    for i in range(nb):
        for u in range(1, 8):
            cons = []
            for y in Wperp[i]:
                c = 0
                for r in range(3):
                    if (u >> r) & 1: c ^= (y >> (r * no)) & mask
                cons.append(c)
            K[i, u] = cons  # 제약 행 (f ⊥ 각 행)
    # 후보 곱: 블록 쌍 (i,j), u_i, u_j, f ∈ K∩K
    cand = {}
    pairlist = list(itertools.combinations(range(nb), 2)) + ([(i, i) for i in range(nb)] if intra else [])
    for i, j in pairlist:
        lst = []
        for ui in range(1, 8):
            for uj in range(1, 8):
                if i == j and uj <= ui: continue
                ns = nullspace(K[i, ui] + K[j, uj], no)
                if len(ns) > 6: ns = ns[:6]
                for f in span_elems(ns):
                    lst.append((ui, uj, f))
        cand[i, j] = lst
    log("  candidates per pair " + str({k: len(v) for k, v in cand.items()}))

    def gen_vec(i, u, f):
        v = 0
        for r in range(3):
            if (u >> r) & 1: v |= f << (r * no)
        return v

    rng = random.Random(seed)
    best = None
    pairs = list(cand)
    for t in range(tries):
        B = [Basis() for _ in range(nb)]
        chosen = []
        # 무작위 순서의 후보 전체를 훑으며 두 블록 모두에서 독립이면 채택 (블록 차수 상한 = dims)
        allc = [(i, j, c) for (i, j) in pairs for c in cand[i, j]]
        rng.shuffle(allc)
        # 낮은 잔여 차원 블록을 우선하도록 두 번 훑음
        for sweep in range(2):
            for i, j, (ui, uj, f) in allc:
                if len(B[i]) >= dims[i] or len(B[j]) >= dims[j]: continue
                gi, gj = gen_vec(i, ui, f), gen_vec(j, uj, f)
                if i == j:
                    if len(B[i]) + 2 > dims[i]: continue
                    t = Basis(); t.piv = dict(B[i].piv)
                    if t.add(gi) and t.add(gj):
                        B[i].add(gi); B[i].add(gj); chosen.append((i, ui, j, uj, f))
                    continue
                ri, _ = B[i].reduce(gi); rj, _ = B[j].reduce(gj)
                if ri and rj:
                    B[i].add(gi); B[j].add(gj); chosen.append((i, ui, j, uj, f))
        deficit = sum(d - len(b) for d, b in zip(dims, B))
        if best is None or deficit < best[0]:
            best = (deficit, chosen, [len(b) for b in B])
            log(f"  try {t}: products {len(chosen)} deficit {deficit} per-block {[len(b) for b in B]}")
        if deficit == 0: break
    return best, W, dims
