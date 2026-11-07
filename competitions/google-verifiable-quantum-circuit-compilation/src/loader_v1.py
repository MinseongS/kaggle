"""chem ANGLE_k loader 컴파일러 v1 (정확성 우선 기준선).

구조 (회로 k 마다):
  주소 a = (h, l), l = 하위 nL 비트, h = 상위 비트. 정의역 {a < A} 는 down-set 이라 ANF 계수를 m < A 로만 두는 확장이 유효.
  f_j(a) = XOR_S x_S(l) * g_{j,S}(h),  g_{j,S} = XOR_t c_j[(t<<nL)|S] * mono_t(h)
  1) 상위 단항식 mono_t (deg>=2, t < ceil(A/2^nL)) 를 ancilla 에 CCX 체인으로 계산 (각 1 AND)
  2) S=∅ 항: 단항식/변수/상수를 CX/X 로 출력에 직접 XOR (무료)
  3) S≠∅: x_S 를 임시 ancilla 에 (|S|-1 AND), 누산기 G 에 g 를 CX 로 만들고 CCX(x_S, G -> out_j)  (출력·S 쌍마다 1 AND)
     --pair: 두 S, S' 를 묶어 P = x_S ^ g_{S'}, Q = g_S ^ x_{S'} 로 1 AND 에 두 항 (보정 x_S x_S' 와 g_S g_S' 는 잔차에 반영)
  4) 해제: 임시곱/단항식은 HMR + 조건부 CZ (Toffoli 0)
레지스터 계약: selection = ceil(log2 A) 비트 (q0..), word_0..word_{J-1} 각 brot 비트, 나머지 workspace.

Usage: uv run python src/loader_v1.py <submission.npz> <out_dir> [--nL 4] [--pair]
"""
import argparse, math, shutil
from pathlib import Path
import numpy as np


def mobius(f, n):
    f = f.copy()
    for i in range(n):
        s = 1 << i
        f = f.reshape(-1, 2 * s, f.shape[-1]); f[:, s:] ^= f[:, :s]; f = f.reshape(-1, f.shape[-1])
    return f


def eval_anf(c, n):  # inverse Möbius = Möbius over GF2
    return mobius(c, n)


class KB:
    def __init__(self, nsel, J, brot):
        self.lines = []
        self.nsel, self.J, self.brot = nsel, J, brot
        self.next_q = nsel + J * brot
        self.free = []
        self.max_q = self.next_q
        self.tof = 0

    def header(self):
        h = [f"APPEND_TO_REGISTER q{i} r0" for i in range(self.nsel)] + ['REGISTER r0 "selection"']
        for j in range(self.J):
            base = self.nsel + j * self.brot
            h += [f"APPEND_TO_REGISTER q{base + b} r{j + 1}" for b in range(self.brot)] + [f'REGISTER r{j + 1} "word_{j}"']
        return h

    def alloc(self):
        q = self.free.pop() if self.free else self.next_q
        if q == self.next_q:
            self.next_q += 1; self.max_q = max(self.max_q, self.next_q)
        self.lines.append(f"R q{q}")
        return q

    def ccx(self, a, b, t):
        self.lines.append(f"CCX q{a} q{b} q{t}"); self.tof += 1

    def cx(self, a, t): self.lines.append(f"CX q{a} q{t}")
    def x(self, t): self.lines.append(f"X q{t}")

    def and_free(self, t, a, b):  # t holds a*b; measurement-based uncompute
        self.lines.append(f"HMR q{t} b0"); self.lines.append(f"CZ q{a} q{b} if b0")
        self.free.append(t)


def compile_circuit(D, A, nsel, J, brot, nL, pair):
    """D: (A, J*brot) delta bits. returns (kmx text, toffolis, qubits)."""
    n = nsel
    nout = J * brot
    f = np.zeros((1 << n, nout), dtype=np.uint8); f[:A] = D
    c = mobius(f, n); c[A:] = 0           # down-set extension
    kb = KB(nsel, J, brot)
    out_q = [nsel + i for i in range(nout)]
    nH = n - nL
    hmax = ((A - 1) >> nL) + 1
    # 1) high monomials
    src = {0: None}
    for i in range(nH):
        src[1 << i] = nL + i
    order = sorted(t for t in range(1, hmax) if bin(t).count("1") >= 2)
    parent = {}
    for t in order:
        top = t.bit_length() - 1
        rest = t ^ (1 << top)
        a = src[rest]; b = nL + top
        q = kb.alloc(); kb.ccx(a, b, q); src[t] = q; parent[t] = (a, b)

    def xor_func(coefs, target):  # coefs: bool array over t < hmax
        for t in np.flatnonzero(coefs):
            if t == 0: kb.x(target)
            else: kb.cx(src[t], target)

    # coefficient tensor G[j][S] = vector over t
    def gvec(S):
        idx = (np.arange(hmax) << nL) | S
        return c[idx].T.astype(bool)  # (nout, hmax)

    # 2) S = ∅
    g0 = gvec(0)
    for j in range(nout):
        xor_func(g0[j], out_q[j])

    # low monomials on the fly
    def low_mono(S):
        bits = [i for i in range(nL) if S >> i & 1]
        if len(bits) == 1: return bits[0], []
        temps = []; cur = bits[0]
        for b in bits[1:]:
            t = kb.alloc(); kb.ccx(cur, b, t); temps.append((t, cur, b)); cur = t
        return cur, temps

    def free_low(temps):
        for t, a, b in reversed(temps): kb.and_free(t, a, b)

    Ss = list(range(1, 1 << nL))
    G = {S: gvec(S) for S in Ss}
    if not pair:
        acc = kb.alloc()
        for S in Ss:
            m, temps = low_mono(S)
            cur = np.zeros(hmax, bool)
            for j in range(nout):
                g = G[S][j]
                if not g.any(): continue
                xor_func(g ^ cur, acc); cur = g.copy()
                kb.ccx(m, acc, out_q[j])
            xor_func(cur, acc)
            free_low(temps)
    else:
        assert nL == 4
        # truth tables on h for products of h-functions
        def h_tt(coefs):  # coefs (hmax,) -> values on h < hmax
            v = np.zeros((1 << nH, 1), np.uint8); v[:hmax, 0] = coefs
            return eval_anf(v, nH)[:hmax, 0]
        def h_anf(vals):
            v = np.zeros((1 << nH, 1), np.uint8); v[:hmax, 0] = vals
            cc = eval_anf(v, nH)[:, 0]; cc[hmax:] = 0  # down-set
            return cc[:hmax].astype(bool)
        Gc = {S: G[S].copy() for S in Ss}
        acc1, acc2 = kb.alloc(), kb.alloc()
        sched = [[(1, 2), (4, 8)], [(3, 12), (5, 10), (6, 9)], [(7, 11), (13, 14)], [(15, None)]]
        for phase in sched:
            for S, S2 in phase:
                m1, t1 = low_mono(S)
                m2, t2 = low_mono(S2) if S2 else (None, [])
                for j in range(nout):
                    ga = Gc[S][j]; gb = Gc[S2][j] if S2 else None
                    if S2 is None or not gb.any():
                        if not ga.any(): continue
                        xor_func(ga, acc2); kb.ccx(m1, acc2, out_q[j]); xor_func(ga, acc2)
                        continue
                    if not ga.any():
                        xor_func(gb, acc2); kb.ccx(m2, acc2, out_q[j]); xor_func(gb, acc2)
                        continue
                    # P = m1 ^ gb, Q = ga ^ m2 ; P*Q = m1 ga + m2 gb + m1 m2 + ga gb
                    kb.cx(m1, acc1); xor_func(gb, acc1)
                    kb.cx(m2, acc2); xor_func(ga, acc2)
                    kb.ccx(acc1, acc2, out_q[j])
                    kb.cx(m1, acc1); xor_func(gb, acc1)
                    kb.cx(m2, acc2); xor_func(ga, acc2)
                    # corrections: ga*gb (h only) now, m1*m2 -> term S|S2 later
                    xor_func(h_anf(h_tt(ga) & h_tt(gb)), out_q[j])
                    Gc[S | S2][j, 0] ^= True
                free_low(t2); free_low(t1)
    # 4) free high monomials (children first)
    for t in reversed(order):
        a, b = parent[t]; kb.and_free(src[t], a, b)
    text = "\n".join(kb.header() + kb.lines) + "\n"
    return text, kb.tof, kb.max_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("out")
    ap.add_argument("--nL", type=int, default=4); ap.add_argument("--pair", action="store_true")
    ap.add_argument("--J", type=int, default=5)
    ap.add_argument("--only", type=int, default=None)
    a = ap.parse_args()
    d = np.load(a.npz)
    words, brot = d["angle_words"], int(d["brot"])
    A, nw = words.shape
    J = a.J; K = math.ceil(nw / J)
    nsel = math.ceil(math.log2(A))
    pad = np.zeros((A, K * J), dtype=np.int64); pad[:, :nw] = words
    batches = pad.reshape(A, K, J)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.npz, out / "submission.npz")
    tot = 0
    for k in range(K) if a.only is None else [a.only]:
        prev = batches[:, k - 1] if k else np.zeros((A, J), np.int64)
        dl = batches[:, k] ^ prev
        D = ((dl[:, :, None] >> np.arange(brot)) & 1).reshape(A, J * brot).astype(np.uint8)
        text, tof, nq = compile_circuit(D, A, nsel, J, brot, a.nL, a.pair)
        (out / f"angle_k_{k}.kmx").write_text(text)
        tot += tof
        print(f"k={k} toffoli={tof} qubits={nq}", flush=True)
    print("total per pass", tot)


if __name__ == "__main__":
    main()
