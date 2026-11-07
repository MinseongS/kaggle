"""chem 상대 코딩 + 행 내 보상: 열 j (5..14) 의 LSB 를 '같은 슬롯 이전 배치 열 j-5 의 LSB ⊕ 16행 블록별 아핀함수' 로 맞춘다.
불일치 행은 열 j 를 ±1 바꾸고, 같은 행의 다른 열을 보상 이동 (코딩된/고정 열: ±2 로 LSB 유지, 아직 처리 안 한 열: ±1)
을 탐욕적으로 최대 R 번 적용해 그 행의 Δe2² (정확, E2Model.deltas) 를 최소화한다.
열 0–4 (배치 0) 는 고정 (공개 회로 0 재사용). gauge 경계 (63/64, 191/192) 를 넘는 이동 금지.
one-body 행 (a<76) 은 Δe2 가 없으므로 단순 최소 이동 (보상 없음) — e1 은 별도 확인.
Usage: uv run python src/chem_code_round2.py <submission.npz> <out.npz> [--cols 5-14] [--rounds 2]"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from chem_code_round import E2Model, load_inputs, evaluate_words, GAUGE_BAD

N1 = 76


def ok_move(old, new):
    return (int(old), int(new)) not in GAUGE_BAD and not (old < 64 <= new or new < 64 <= old and False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("out")
    ap.add_argument("--cols", default="5-14"); ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    lo, hi = map(int, a.cols.split("-")); cols = list(range(lo, hi + 1))
    z = np.load(a.npz); inp = load_inputs(a.npz)
    words = z["angle_words"].astype(np.int64).copy(); edges = z["tree_edges"]
    A = words.shape[0]
    m0 = evaluate_words(inp, words, edges, 8)
    print("base", {k: round(v, 6) for k, v in m0.items() if isinstance(v, float)}, flush=True)
    model = E2Model(inp, words, edges)
    xs = np.arange(16)
    pats = np.array([((c0 + bin(x & cm).count("1")) & 1) for c0 in (0, 1) for cm in range(16) for x in xs]).reshape(32, 16)
    frozen = set(range(0, 5))
    t0 = time.time()
    for j in cols:
        cur = model.words
        pending = set(c for c in cols if c > j)  # 아직 코딩 안 한 열: ±1 보상 가능
        # 1) 단순 비용 (보상 없음) 으로 패턴 선택
        cost = np.full((A, 2), np.inf)
        for di, d in enumerate((1, -1)):
            nw = cur.copy(); nw[:, j] = (cur[:, j] + d) % 256
            ok = np.array([(int(cur[r, j]), int(nw[r, j])) not in GAUGE_BAD for r in range(A)])
            r2 = np.nonzero(ok & (np.arange(A) >= N1))[0]
            cost[r2, di] = model.deltas(r2, nw[r2])
            r1 = np.nonzero(ok & (np.arange(A) < N1))[0]
            cost[r1, di] = 1e-9
        best1 = np.minimum(cost[:, 0], cost[:, 1])
        ref = cur[:, j - 5] & 1  # 이전 배치 같은 슬롯 LSB
        flips = []
        for y in range((A + 15) // 16):
            rr = np.arange(16 * y, min(16 * y + 16, A)); x = rr - 16 * y
            want_rel = (cur[rr, j] & 1) ^ ref[rr]
            cs = [best1[rr[p[x] != want_rel]].sum() for p in pats]
            pi = int(np.argmin(cs)); p = pats[pi]
            flips += list(rr[p[x] != want_rel])
        # 2) 각 불일치 행: 열 j 를 ±1 + 보상 탐욕
        newrows = {}
        two = [r for r in flips if r >= N1]; one = [r for r in flips if r < N1]
        for r in one:
            d = 1 if cost[r, 0] <= cost[r, 1] else -1
            newrows[r] = cur[r].copy(); newrows[r][j] = (cur[r, j] + d) % 256
        tot_before = 0.0; tot_after = 0.0
        # 배치 처리: 각 행에 대해 후보 이동 집합을 만들어 deltas 로 평가
        base_rows = []
        for r in two:
            cands = []
            for d in (1, -1):
                v = (cur[r, j] + d) % 256
                if (int(cur[r, j]), int(v)) in GAUGE_BAD: continue
                w = cur[r].copy(); w[j] = v; cands.append(w)
            base_rows.append(cands)
        # 첫 이동: 두 방향 중 정확 비용 최소
        rows_arr = []; cand_arr = []; owner = []
        for k_, (r, cands) in enumerate(zip(two, base_rows)):
            for w in cands: rows_arr.append(r); cand_arr.append(w); owner.append(k_)
        dl = model.deltas(np.array(rows_arr), np.array(cand_arr))
        state = {}
        for idx, k_ in enumerate(owner):
            if k_ not in state or dl[idx] < state[k_][0]: state[k_] = (dl[idx], cand_arr[idx])
        tot_before = sum(v[0] for v in state.values())
        for rnd in range(a.rounds):
            rows_arr = []; cand_arr = []; owner = []
            for k_, r in enumerate(two):
                if k_ not in state: continue
                w0 = state[k_][1]
                for c in range(75):
                    if c == j or c in frozen: continue
                    steps = (1, -1) if c in pending else (2, -2)
                    for d in steps:
                        v = (w0[c] + d) % 256
                        g0 = ((int(w0[c]) >> 7) ^ (int(w0[c]) >> 6)) & 1; g1 = ((int(v) >> 7) ^ (int(v) >> 6)) & 1
                        if g0 != g1: continue
                        w = w0.copy(); w[c] = v
                        rows_arr.append(r); cand_arr.append(w); owner.append(k_)
            if not rows_arr: break
            dl = model.deltas(np.array(rows_arr), np.array(cand_arr))
            improved = 0
            bestc = {}
            for idx, k_ in enumerate(owner):
                if dl[idx] < state[k_][0] and (k_ not in bestc or dl[idx] < bestc[k_][0]): bestc[k_] = (dl[idx], cand_arr[idx])
            for k_, v in bestc.items(): state[k_] = v; improved += 1
            if not improved: break
        tot_after = sum(v[0] for v in state.values())
        for k_, r in enumerate(two):
            if k_ in state: newrows[r] = state[k_][1]
        old = model.e2sq
        for r, w in newrows.items(): model.words[r] = w
        model.refresh()
        print(f"col {j}: flips {len(flips)} (2body {len(two)}), Σ single-row Δ before comp {tot_before:.3e} after {tot_after:.3e}, actual Δe2² {model.e2sq - old:.3e}, e2 {np.sqrt(model.e2sq):.6f} ({time.time()-t0:.0f}s)", flush=True)
    m1 = evaluate_words(inp, model.words, edges, 8)
    print("coded", {k: round(v, 6) for k, v in m1.items() if isinstance(v, float)}, flush=True)
    out = {k: z[k] for k in z.files}; out["angle_words"] = model.words
    np.savez(a.out, **out); print("wrote", a.out)


if __name__ == "__main__":
    main()
