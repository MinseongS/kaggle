"""chem ANGLE_k loader 컴파일러 v2 (자체 구현): 블록 쌍 곱 계획 + pi-gauge helper.

회로 k (k>=1) 구조:
  1. 각 단어 CX(bit6 -> bit7): 입력 batch 의 bit7^bit6 = helper (QUADS[j] 저주소 2차 단항식) 이므로
     top 선이 helper 를 담는다 (정·역방향 모두). 끝에서 같은 CX 로 bit7 = helper ^ 새 bit6 복원.
  2. high 단항식 69개 + low 단항식 (helper 가 아닌 6개) 을 init_and 로 계산.
  3. 곱: (low 형태 + high 형태)·(low 형태 + high 형태) 를 선형결합 in-place CX 로 만들고 CCX 1개 + CX fan-out.
  4. 순수(low-only / high-only) 잔여는 캐시 선에서 CX.
  5. 단항식 measurement 기반 해제 (del_and), 1 의 CX 복원.
k=0: 입력 0 (정방향) / batch0 (역방향). helper 를 top 선에서 얻을 수 없으므로 low 11개 모두 계산하고
     bit7 을 일반 출력 (bit6 fan-out + helper CX) 으로 다룬다. (qubit 131 > 128 문제: 아직 미지원)
Usage: uv run python src/loader_v2.py <submission.npz> <k> <out.kmb> [--tries N] [--seed S]
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np
import kickmix as km
sys.path.insert(0, str(Path(__file__).parent))
from loader_tensor import anf_bits, delta, A, gf2rank
from block_planner import plan, Basis

QUADS = (3, 5, 6, 9, 10)
HIGH = [h << 4 for h in range(1, 77)]                 # high 단항식 마스크 (singleton 포함)
HIGH_AND = [m for m in HIGH if bin(m).count("1") >= 2]  # 69
LOW = list(range(1, 16))
LOW_AND = [m for m in LOW if bin(m).count("1") >= 2]    # 11


def truth(m):
    xs = np.arange(A)
    return ((xs & m) == m).astype(np.uint8)


def bits_of(x):
    while x:
        b = x & -x; yield b.bit_length() - 1; x ^= b


def build_products(T, hard, blocks, tries, seed, log):
    best, Wb, dims = plan(T, hard, blocks, seed=seed, tries=tries, log=log)
    deficit, chosen, _ = best
    if deficit:
        raise RuntimeError(f"planner deficit {deficit}")
    no = len(hard)
    # 블록별 생성자 기저 (combo = 생성자 번호) 로 각 h 의 T_h 블록 부분을 분해
    gens = [[] for _ in blocks]  # (product idx, side)
    bases = [Basis() for _ in blocks]
    def gvec(u, f):
        v = 0
        for r in range(3):
            if (u >> r) & 1: v |= f << (r * no)
        return v
    for p, (i, ui, j, uj, f) in enumerate(chosen):
        for blk, u, side in ((i, ui, "a"), (j, uj, "b")):
            idx = len(gens[blk]); gens[blk].append((p, side))
            assert bases[blk].add(gvec(u, f), 1 << idx)
    aH = [0] * len(chosen); bH = [0] * len(chosen)
    for bi, B in enumerate(blocks):
        for h in range(76):
            v = 0
            for r, li in enumerate(B):
                if h == 75 and li >= 3: continue
                for oi, o in enumerate(hard):
                    if T[li, h, o]: v |= 1 << (r * no + oi)
            rem, combo = bases[bi].reduce(v)
            assert rem == 0, "T_h not in span"
            for g in bits_of(combo):
                p, side = gens[bi][g]
                # a-side 생성자 (블록 i 의 ui) 계수 -> bH, b-side -> aH
                if side == "a": bH[p] ^= 1 << h
                else: aH[p] ^= 1 << h
    prods = []
    for p, (i, ui, j, uj, f) in enumerate(chosen):
        aL = sum(1 << LOW[blocks[i][r]] for r in range(3) if (ui >> r) & 1)
        bL = sum(1 << LOW[blocks[j][r]] for r in range(3) if (uj >> r) & 1)
        fan = sum(1 << hard[oi] for oi in bits_of(f))
        prods.append(dict(aL=aL, aH=aH[p], bL=bL, bH=bH[p], fan=fan))
    return prods


def group_products(T, outs, rows, log):
    """출력 outs 각각 독립: 살아있는 low 행 rows 를 둘씩 묶어 곱 (bit0 코딩 그룹 등)."""
    prods = []
    for o in outs:
        live = [li for li in rows if T[li, :, o].any()]
        for t in range(0, len(live), 2):
            r1 = live[t]
            hrow = lambda li: sum(1 << h for h in range(76) if T[li, h, o] and not (h == 75 and li >= 3))
            if t + 1 < len(live):
                r2 = live[t + 1]
                # mixed = x^{r1}⊗bH + x^{r2}⊗aH
                prods.append(dict(aL=1 << LOW[r1], aH=hrow(r2), bL=1 << LOW[r2], bH=hrow(r1), fan=1 << o))
            else:
                prods.append(dict(aL=1 << LOW[r1], aH=0, bL=0, bH=hrow(r1), fan=1 << o))
    return prods


def form_truth(L, H):
    t = np.zeros(A, np.uint8)
    for m in bits_of(L): t ^= truth(m)
    for h in bits_of(H): t ^= truth((h + 1) << 4)
    return t


def compile_k(words, k, tries=300, seed=0, blocks=None, log=print):
    J = 5
    d = delta(words, k)
    bits8 = ((d[:, :, None] >> np.arange(8)) & 1).astype(np.uint8)  # (A,5,8)
    if k >= 1:
        prev, cur = words[:, (k - 1) * J:k * J], words[:, k * J:(k + 1) * J]
        for j in range(J):
            gp = ((prev[:, j] >> 7) ^ (prev[:, j] >> 6)) & 1
            gc = ((cur[:, j] >> 7) ^ (cur[:, j] >> 6)) & 1
            assert np.array_equal(gp, truth(QUADS[j])) and np.array_equal(gc, truth(QUADS[j])), f"gauge helper missing slot {j}"
    nout = 35
    outbits = bits8[:, :, :7].reshape(A, 35)  # o = 7j + bit
    c = anf_bits(outbits)
    T = np.zeros((15, 76, nout), np.uint8)
    for li, L in enumerate(LOW):
        for hi in range(76):
            m = (hi + 1) * 16 + L
            if m < A: T[li, hi] = c[m]
    hard = [7 * j + b for j in range(5) for b in range(1, 7)]
    bit0 = [7 * j for j in range(5)]
    if blocks is None:
        blocks = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14)]
    prods = build_products(T, hard, blocks, tries, seed, log)
    nh = len(prods)
    prods += group_products(T, bit0, list(range(15)), log)
    log(f"  products hard {nh} + bit0 {len(prods) - nh} = {len(prods)}")
    # 잔여 (pure) 계산
    resid = outbits.copy()
    for p in prods:
        t = form_truth(p["aL"], p["aH"]) & form_truth(p["bL"], p["bH"])
        for o in bits_of(p["fan"]): resid[:, o] ^= t
    rc = anf_bits(resid)
    pure = {}
    for m in range(A):
        nz = np.nonzero(rc[m])[0]
        if len(nz) == 0: continue
        assert m < 16 or m % 16 == 0, f"non-pure residual monomial {m}"
        pure[m] = [int(o) for o in nz]
    return prods, pure


def gauge_flags(words, k, J=5):
    """슬롯별 (입력 batch gauge 가 helper 인가, 출력 batch gauge 가 helper 인가). gauge 는 helper 아니면 0 이어야 함."""
    fl = []
    for j in range(J):
        out = []
        for kk in (k - 1, k):
            w = words[:, kk * J + j]
            g = ((w >> 7) ^ (w >> 6)) & 1
            if np.array_equal(g, truth(QUADS[j])): out.append(True)
            elif not g.any(): out.append(False)
            else: raise AssertionError(f"slot {j} batch {kk}: gauge neither helper nor 0")
        fl.append(tuple(out))
    return fl


def emit(prods, pure, k, flags=None):
    b = km.CircuitBuilder()
    sel = b.create_quantum_register(11, "selection")
    words = [b.create_quantum_register(8, f"word_{j}") for j in range(5)]
    tgt = lambda o: words[o // 7][o % 7]
    cache = {1 << i: sel[i] for i in range(11)}
    if flags is None: flags = [(True, True)] * 5
    toggles = []
    if k >= 1:
        for w in words: b.cx(w[6], w[7])
        free = [m for j, m in enumerate(QUADS) if flags[j] == (True, True)]
        for j, m in enumerate(QUADS):
            if flags[j] == (True, True): cache[m] = words[j][7]
            elif flags[j][0] != flags[j][1]: toggles.append((j, m))
        low_and = [m for m in LOW_AND if m not in free]
        zero_tops = [words[j][7] for j in range(5) if flags[j] == (False, False)]  # 정·역 모두 0 인 선
    else:
        raise NotImplementedError("k=0")
    hq = b.alloc_qubits(len(HIGH_AND))
    n_in_tops = min(len(zero_tops), len(low_and))
    lq = list(zero_tops[:n_in_tops]) + list(b.alloc_qubits(len(low_and) - n_in_tops))
    def split(m):
        t = 1 << (m.bit_length() - 1); return m ^ t, t
    for m, q in zip(HIGH_AND, hq):
        x, y = split(m); b.init_and(cache[x], cache[y], q); cache[m] = q
    for idx, (m, q) in enumerate(zip(low_and, lq)):
        x, y = split(m)
        if idx < n_in_tops: b.ccx(cache[x], cache[y], q)  # 단어 bit7 선 (0 보장): R 없이
        else: b.init_and(cache[x], cache[y], q)
        cache[m] = q
    def wires(L, H):
        s = set()
        for m in bits_of(L): s ^= {cache[1 << m]} if False else set()
        out = []
        for m in bits_of(L): out.append(cache[m])
        for h in bits_of(H): out.append(cache[(h + 1) << 4])
        return out
    qid = lambda q: int(str(q)[1:])
    for p in prods:
        aa = {qid(q): q for q in wires(p["aL"], p["aH"])}
        bb = {qid(q): q for q in wires(p["bL"], p["bH"])}
        assert aa and bb
        steps = []
        pa = min(aa)
        steps += [(w, pa) for w in sorted(aa) if w != pa]
        if pa in bb:
            rest = (set(aa) ^ set(bb)) - {pa}
            assert rest, "a == b"
            pb = min(rest)
            steps += [(w, pb) for w in sorted(rest) if w != pb] + [(pa, pb)]
        else:
            pb = min(bb)
            steps += [(w, pb) for w in sorted(bb) if w != pb]
        for s, t in steps: b.cx(km.q(s), km.q(t))
        fan = list(bits_of(p["fan"]))
        t0 = tgt(fan[0])
        for o in fan[1:]: b.cx(t0, tgt(o))
        b.ccx(km.q(pa), km.q(pb), t0)
        for o in fan[1:]: b.cx(t0, tgt(o))
        for s, t in reversed(steps): b.cx(km.q(s), km.q(t))
    for m, outs in pure.items():
        for o in outs:
            if m == 0: b.x(tgt(o))
            else: b.cx(cache[m], tgt(o))
    for j, m in toggles:  # gauge 가 helper↔0 로 바뀌는 슬롯: top 에 helper 를 XOR
        b.cx(cache[m], words[j][7])
    for m, q in reversed(list(zip(low_and, lq))):
        x, y = split(m); b.del_and(cache[x], cache[y], q)
    for m, q in reversed(list(zip(HIGH_AND, hq))):
        x, y = split(m); b.del_and(cache[x], cache[y], q)
    b.free(km.array(lq[n_in_tops:])) if len(lq) > n_in_tops else None; b.free(hq)
    if k >= 1:
        for w in words: b.cx(w[6], w[7])
    return b.finish_circuit()


def check(words, k, circ, shots=2):
    """정·역방향 전 주소 시뮬레이션 (작업공간·위상·단어)."""
    J = 5
    nv = A
    sim = km.Simulator(batch_size=nv * shots, debug_print=False, seed=123, counting_operations=True)
    sim.use_same_registers_as(circ)
    regs = circ.register_data
    sel = km.array(list(regs["selection"])); ws = [km.array(list(regs[f"word_{j}"])) for j in range(J)]
    addr = np.tile(np.arange(nv), shots)
    ok = True
    for direction, (src, dst) in enumerate([((k - 1), k), (k, k - 1)]):
        sim.clear_for_shot()
        sim.write_across_shots(sel, ((addr[None, :] >> np.arange(11)[:, None]) & 1).astype(bool))
        for j in range(J):
            v = words[addr, src * J + j] if src >= 0 else np.zeros(len(addr), np.int64)
            sim.write_across_shots(ws[j], ((v[None, :] >> np.arange(8)[:, None]) & 1).astype(bool))
        sim.do(circ)
        for j in range(J):
            got = (np.int64(1) << np.arange(8)) @ sim.read_across_shots(ws[j]).astype(np.int64)
            exp = words[addr, dst * J + j] if dst >= 0 else np.zeros(len(addr), np.int64)
            if not np.array_equal(got, exp):
                print(f"  dir {direction} word {j} mismatch rows {(got != exp).sum()}"); ok = False
        allq = set(range(circ.num_qubits)); pub = {int(str(q)[1:]) for q in list(sel)} | {int(str(q)[1:]) for w in ws for q in list(w)}
        scr = km.array([km.q(q) for q in sorted(allq - pub)])
        if len(scr) and sim.read_across_shots(scr).any():
            print("  dirty workspace"); ok = False
    st = sim.operation_statistics()
    return ok, st.num_toffoli_like / (nv * shots * 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("k", type=int); ap.add_argument("out")
    ap.add_argument("--tries", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    W = np.load(a.npz)["angle_words"]
    prods, pure = compile_k(W, a.k, a.tries, a.seed)
    c = emit(prods, pure, a.k)
    st = c.operation_statistics()
    print("static toffoli", st.num_toffoli_like, "qubits", c.num_qubits)
    ok, tof = check(W, a.k, c)
    print("check", ok, "mean toffoli/pass", tof)
    c.to_file(a.out, format="kmb" if a.out.endswith(".kmb") else "kmx")


def compile_joint(words, k, plan):
    """joint_planner 결과 (mono 블록, sel 곱) 로 35 출력 전체를 곱으로 덮는다."""
    from joint_planner import NO
    J = 5
    d = delta(words, k)
    bits8 = ((d[:, :, None] >> np.arange(8)) & 1).astype(np.uint8)
    if k >= 1: gauge_flags(words, k)  # 검사만
    outbits = bits8[:, :, :7].reshape(A, 35)
    c = anf_bits(outbits)
    mono = plan["mono"]; sel = plan["sel"]
    gens = [[] for _ in mono]; bases = [Basis() for _ in mono]
    def gvec(u, f):
        v = 0
        for r in range(3):
            if (u >> r) & 1: v |= f << (r * NO)
        return v
    for p, (i, ui, j, uj, f) in enumerate(sel):
        for blk, u, side in ((i, ui, "a"), (j, uj, "b")):
            idx = len(gens[blk]); gens[blk].append((p, side))
            assert bases[blk].add(gvec(u, f), 1 << idx), "dependent generator"
    aH = [0] * len(sel); bH = [0] * len(sel)
    from joint_planner import block_cols
    Tt = np.zeros((15, 76, 35), np.uint8)
    for li in range(15):
        for hi in range(76):
            mm = (hi + 1) * 16 + li + 1
            if mm < A: Tt[li, hi] = c[mm]
    for bi, bl in enumerate(mono):
        colsb = block_cols(Tt, bl)
        for h in range(76):
            v = colsb[h]
            rem, combo = bases[bi].reduce(v)
            assert rem == 0, f"T_h not in S block {bi} h {h}"
            for g in bits_of(combo):
                p, side = gens[bi][g]
                if side == "a": bH[p] ^= 1 << h
                else: aH[p] ^= 1 << h
    prods = []
    for p, (i, ui, j, uj, f) in enumerate(sel):
        aL = sum(1 << mono[i][r] for r in range(3) if (ui >> r) & 1)
        bL = sum(1 << mono[j][r] for r in range(3) if (uj >> r) & 1)
        prods.append(dict(aL=aL, aH=aH[p], bL=bL, bH=bH[p], fan=f))
    resid = outbits.copy()
    for p in prods:
        t = form_truth(p["aL"], p["aH"]) & form_truth(p["bL"], p["bH"])
        for o in bits_of(p["fan"]): resid[:, o] ^= t
    rc = anf_bits(resid)
    pure = {}
    for m in range(A):
        nz = np.nonzero(rc[m])[0]
        if len(nz) == 0: continue
        assert m < 16 or m % 16 == 0, f"non-pure residual monomial {m}"
        pure[m] = [int(o) for o in nz]
    return prods, pure


# ---------- k=0 (입력 0 / 역방향 batch0, helper 없음) ----------
K0_BLOCKS = [[3, 7, 11], [5, 13, 15], [6, 14, 1], [9, 10, 2], [12, 4, 8]]  # 블록 내부에서만 AND 의존
K0_PHASES = [[0, 2, 3, 4], [0, 1, 4], [1, 2, 3, 4]]  # 단계별 살아있는 블록 (AND 보조선 ≤ 8)


def compile_joint_k0(words, plan):
    """k=0: 35출력 (bit0..6) 계획 + bit7 = bit6 ⊕ helper (fanout 확장 + 순수항)."""
    from joint_planner import NO
    J = 5
    d = words[:, :J].copy()
    bits8 = ((d[:, :, None] >> np.arange(8)) & 1).astype(np.uint8)
    for j in range(J):
        g = bits8[:, j, 7] ^ bits8[:, j, 6]
        assert np.array_equal(g, truth(QUADS[j])), "batch0 gauge != helper"
    prods, pure = compile_joint_words(bits8[:, :, :7].reshape(A, 35), plan)
    # 출력 인덱스: 40비트 o = 8j + bit 로 재매핑, bit6 fanout 은 bit7 에도
    def remap(o35):
        j, bit = divmod(o35, 7); return 8 * j + bit
    for p in prods:
        f = 0
        for o in bits_of(p["fan"]):
            o40 = remap(o); f |= 1 << o40
            if o % 7 == 6: f |= 1 << (o40 + 1)
        p["fan40"] = f
    pure40 = {}
    for m, outs in pure.items():
        for o in outs:
            o40 = remap(o); pure40.setdefault(m, []).append(o40)
            if o % 7 == 6: pure40[m].append(o40 + 1)
    for j, m in enumerate(QUADS):
        pure40.setdefault(m, []).append(8 * j + 7)
    for m in list(pure40):
        cnt = {}
        for o in pure40[m]: cnt[o] = cnt.get(o, 0) ^ 1
        pure40[m] = [o for o, v in cnt.items() if v]
        if not pure40[m]: del pure40[m]
    return prods, pure40


def compile_joint_words(outbits, plan):
    """compile_joint 의 핵심 (출력 비트 행렬을 직접 받는 판)."""
    from joint_planner import NO, block_cols
    c = anf_bits(outbits)
    mono = plan["mono"]; sel = plan["sel"]
    gens = [[] for _ in mono]; bases = [Basis() for _ in mono]
    def gvec(u, f):
        v = 0
        for r in range(3):
            if (u >> r) & 1: v |= f << (r * NO)
        return v
    for p, (i, ui, j, uj, f) in enumerate(sel):
        for blk, u, side in ((i, ui, "a"), (j, uj, "b")):
            idx = len(gens[blk]); gens[blk].append((p, side))
            assert bases[blk].add(gvec(u, f), 1 << idx), "dependent generator"
    aH = [0] * len(sel); bH = [0] * len(sel)
    Tt = np.zeros((15, 76, 35), np.uint8)
    for li in range(15):
        for hi in range(76):
            mm = (hi + 1) * 16 + li + 1
            if mm < A: Tt[li, hi] = c[mm]
    for bi, bl in enumerate(mono):
        colsb = block_cols(Tt, bl)
        for h in range(76):
            rem, combo = bases[bi].reduce(colsb[h]); assert rem == 0
            for g in bits_of(combo):
                p, side = gens[bi][g]
                if side == "a": bH[p] ^= 1 << h
                else: aH[p] ^= 1 << h
    prods = []
    for p, (i, ui, j, uj, f) in enumerate(sel):
        aL = sum(1 << mono[i][r] for r in range(3) if (ui >> r) & 1)
        bL = sum(1 << mono[j][r] for r in range(3) if (uj >> r) & 1)
        prods.append(dict(aL=aL, aH=aH[p], bL=bL, bH=bH[p], fan=f, blocks=(i, j)))
    resid = outbits.copy()
    for p in prods:
        t = form_truth(p["aL"], p["aH"]) & form_truth(p["bL"], p["bH"])
        for o in bits_of(p["fan"]): resid[:, o] ^= t
    rc = anf_bits(resid); pure = {}
    for m in range(A):
        nz = np.nonzero(rc[m])[0]
        if len(nz) == 0: continue
        assert m < 16 or m % 16 == 0, f"non-pure residual monomial {m}"
        pure[m] = [int(o) for o in nz]
    return prods, pure


def emit_k0(prods, pure40, mono):
    b = km.CircuitBuilder()
    sel = b.create_quantum_register(11, "selection")
    words = [b.create_quantum_register(8, f"word_{j}") for j in range(5)]
    tgt = lambda o: words[o // 8][o % 8]
    cache = {1 << i: sel[i] for i in range(11)}
    hq = b.alloc_qubits(len(HIGH_AND))
    def split(m):
        t = 1 << (m.bit_length() - 1); return m ^ t, t
    for m, q in zip(HIGH_AND, hq):
        x, y = split(m); b.init_and(cache[x], cache[y], q); cache[m] = q
    qid = lambda q: int(str(q)[1:])
    # 블록별 AND 단항식 (블록 내부 의존: 오름차순으로 계산하면 하위가 먼저)
    def block_ands(i):
        return sorted(m for m in mono[i] if bin(m).count("1") >= 2)
    alive = {}
    def lsplit(m):
        # 같은 블록 (또는 단일변수) 로 분해되는 비트 t 선택
        blk = next(bl for bl in mono if m in bl)
        for t in (8, 4, 2, 1):
            if m & t and (bin(m ^ t).count("1") == 1 or (m ^ t) in blk): return m ^ t, t
        raise AssertionError(f"no in-block split for {m}")
    def bring_up(i):
        for m in block_ands(i):
            if m in alive: continue
            q = b.alloc_qubits(1)[0]
            x, y = lsplit(m)
            # split 결과가 다른 블록 단항식이면 안 됨 (K0_BLOCKS 는 블록 내 의존만)
            b.init_and(cache[x], cache[y], q); cache[m] = q; alive[m] = q
    def bring_down(i):
        for m in reversed(block_ands(i)):
            if m not in alive: continue
            x, y = lsplit(m); q = alive.pop(m)
            b.del_and(cache[x], cache[y], q); b.free(q); del cache[m]
    def wires(L, H):
        out = [cache[m] for m in bits_of(L)] + [cache[(h + 1) << 4] for h in bits_of(H)]
        return out
    def do_prod(p):
        aa = {qid(q): q for q in wires(p["aL"], p["aH"])}
        bb = {qid(q): q for q in wires(p["bL"], p["bH"])}
        steps = []; pa = min(aa)
        steps += [(w, pa) for w in sorted(aa) if w != pa]
        if pa in bb:
            rest = (set(aa) ^ set(bb)) - {pa}; pb = min(rest)
            steps += [(w, pb) for w in sorted(rest) if w != pb] + [(pa, pb)]
        else:
            pb = min(bb); steps += [(w, pb) for w in sorted(bb) if w != pb]
        for s, t in steps: b.cx(km.q(s), km.q(t))
        fan = list(bits_of(p["fan40"])); t0 = tgt(fan[0])
        for o in fan[1:]: b.cx(t0, tgt(o))
        b.ccx(km.q(pa), km.q(pb), t0)
        for o in fan[1:]: b.cx(t0, tgt(o))
        for s, t in reversed(steps): b.cx(km.q(s), km.q(t))
    done = set(); pure_done = set()
    # 순수항 중 low AND 가 필요 없는 것 (상수, 단일변수, high) 은 바로
    for m, outs in pure40.items():
        if m == 0 or m >= 16 or bin(m).count("1") == 1:
            for o in outs:
                if m == 0: b.x(tgt(o))
                else: b.cx(cache[m], tgt(o))
            pure_done.add(m)
    cur = set()
    for ph in K0_PHASES:
        for i in sorted(cur - set(ph)): bring_down(i)
        for i in ph: bring_up(i)
        cur = set(ph)
        for pi_, p in enumerate(prods):
            if pi_ in done: continue
            i, j = p["blocks"]
            if i in cur and j in cur: do_prod(p); done.add(pi_)
        for m, outs in pure40.items():
            if m in pure_done or m not in cache: continue
            for o in outs: b.cx(cache[m], tgt(o))
            pure_done.add(m)
    for i in sorted(cur): bring_down(i)
    assert len(done) == len(prods), f"unscheduled products {len(prods) - len(done)}"
    assert pure_done == set(pure40), "unscheduled pure terms"
    for m, q in reversed(list(zip(HIGH_AND, hq))):
        x, y = split(m); b.del_and(cache[x], cache[y], q)
    b.free(hq)
    return b.finish_circuit()
