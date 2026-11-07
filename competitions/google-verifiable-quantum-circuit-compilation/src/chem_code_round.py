"""chem angle 단어 LSB 코딩: 지정 열의 각 16행 블록 (같은 high 주소) 에서 LSB 를 low 4비트의 아핀 함수로 만든다.
(공개 coded loader 와 같은 코드: 이러면 그 열이 들어가는 회로의 bit0 델타 mixed 행이 단일변수뿐 -> 곱 30출력 블록법 적용 가능)

불일치 행은 단어를 ±1 (gauge 경계 63/64, 191/192 금지) 바꾸고, 그 비용은 e2^2 의 정확한 단일행 변화량
(Y 의 랭크-1 갱신 공식) 으로 평가. 패턴 32개 중 비용 최소를 고른다 (행 간 상호작용은 열 단위 커밋 후 재계산).
one-body 행 (a<76) 은 e1 로 평가 (단순: 행 변경 후 e1 직접 계산).
Usage: uv run python src/chem_code_round.py <submission.npz> <out.npz> --cols 0-14
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/verifiers"))
from dfthcsos.verify import load_inputs, evaluate_words  # noqa: E402
from dfthcsos.scoreinfra.dfthc_math import reconstruct_u_vectors, h1_thc_reconstruction_error, reconstruct_x  # noqa: E402

GAUGE_BAD = {(63, 64), (64, 63), (191, 192), (192, 191)}


def decode(words_rows, edges):
    v = reconstruct_u_vectors(words_rows / 128.0, edges)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class E2Model:
    def __init__(self, inp, words, edges):
        self.inp, self.edges = inp, edges
        ref = inp.ref; tei = np.asarray(ref["tei"], np.float64); S = inp.S
        N = tei.shape[0]; self.N = N
        R, B, C = inp.w.shape; self.R, self.B, self.C = R, B, C
        T = tei.copy(); di = np.arange(N)
        T[:, :, di, di] += 0.5 * S[:, :, None]; T[di, di, :, :] += 0.5 * S[None, :, :]
        T = T.reshape(N * N, N * N); self.T = 0.5 * (T + T.T); self.Tn2 = float((self.T ** 2).sum())
        self.words = words.copy()
        self.refresh()

    def refresh(self):
        N, R, B, C = self.N, self.R, self.B, self.C
        self.U = decode(self.words[N:], self.edges).reshape(R, B, N)
        X = np.einsum("rbp,rbq,rbc->rcpq", self.U, self.U, self.inp.w)
        self.Y = X.reshape(R * C, N * N)
        self.G = self.Y @ self.Y.T
        self.YT = self.Y @ self.T
        self.e2sq = float((self.G ** 2).sum() - 2 * (self.YT * self.Y).sum() + self.Tn2)

    def deltas(self, rows, newwords):
        """rows: 두-체 행 주소 (>=N) 배열, newwords: (n, 75) 새 단어. 반환 Δe2^2 (n,)"""
        N, R, B, C = self.N, self.R, self.B, self.C
        out = np.empty(len(rows))
        for s in range(0, len(rows), 1024):
            rr = rows[s:s + 1024]; nw = newwords[s:s + 1024]
            u_new = decode(nw, self.edges)
            idx = rr - N; rho = idx // B; bb = idx % B
            u_old = self.U[rho, bb]
            D = (np.einsum("np,nq->npq", u_new, u_new) - np.einsum("np,nq->npq", u_old, u_old)).reshape(len(rr), N * N).T  # (N², n)
            TD = self.T @ D
            Z = self.Y @ D          # (RC, n)
            YTD = self.YT @ D       # (RC, n)
            s_ = (D * D).sum(0); DTD = (D * TD).sum(0)
            for t in range(len(rr)):
                r = rho[t]; a = self.inp.w[r, bb[t]]  # (C,)
                sl = slice(r * C, (r + 1) * C)
                z = Z[:, t]
                Ga = self.G[:, sl] @ a          # (RC,)
                aGz = Ga @ z; aGa = Ga[sl] @ a
                zp = z.copy(); zp[sl] += 0.5 * s_[t] * a
                a2 = a @ a; azp = a @ zp[sl]
                dG2 = 4 * aGz + 2 * s_[t] * aGa + 2 * a2 * (zp @ zp) + 2 * azp ** 2
                dtr = 2 * (a @ YTD[sl, t]) + a2 * DTD[t]
                out[s + t] = dG2 - 2 * dtr
        return out


def e1_of(inp, words, edges):
    N = 76
    u1 = decode(words[:N], edges)
    R, B, C = inp.w.shape
    U = decode(words[N:], edges).reshape(R, B, N)
    X = reconstruct_x(U, inp.w); X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    ref = inp.ref; h1_thc = ref["oei"] - 0.5 * np.einsum("prrq->pq", ref["tei"])
    return h1_thc_reconstruction_error(u1, h1_thc, inp.S, inp.beta_1, ref["electrons"], inp.W, X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("out")
    ap.add_argument("--cols", default="0-14")
    ap.add_argument("--e1w", type=float, default=0.0, help="one-body 비용 가중 (e1^2 변화 * e1w)")
    a = ap.parse_args()
    lo, hi = map(int, a.cols.split("-"))
    cols = list(range(lo, hi + 1))
    z = np.load(a.npz); inp = load_inputs(a.npz)
    words = z["angle_words"].astype(np.int64).copy(); edges = z["tree_edges"]
    A, N = words.shape[0], 76
    m0 = evaluate_words(inp, words, edges, 8)
    print("base", {k: round(v, 6) for k, v in m0.items() if isinstance(v, float)}, flush=True)
    model = E2Model(inp, words, edges)
    print("model e2", np.sqrt(model.e2sq), flush=True)
    xs = np.arange(16)
    pats = [((c0 + bin(x & cm).count("1")) & 1) for c0 in (0, 1) for cm in range(16) for x in xs]
    pats = np.array(pats).reshape(32, 16)
    t0 = time.time()
    for j in cols:
        # 후보 비용: 모든 행에 대해 ±1
        cost = np.full((A, 2), np.inf)
        cur = model.words
        for di, d in enumerate((1, -1)):
            nw_all = cur.copy(); nw_all[:, j] = (cur[:, j] + d) % 256
            ok = np.array([(int(cur[r, j]), int(nw_all[r, j])) not in GAUGE_BAD for r in range(A)])
            rows2 = np.nonzero(ok & (np.arange(A) >= N))[0]
            cost[rows2, di] = model.deltas(rows2, nw_all[rows2])
            rows1 = np.nonzero(ok & (np.arange(A) < N))[0]
            if a.e1w > 0:
                base1 = e1_of(inp, cur, edges) ** 2
                for r in rows1:
                    w2 = cur.copy(); w2[r, j] = nw_all[r, j]
                    cost[r, di] = a.e1w * (e1_of(inp, w2, edges) ** 2 - base1)
            else:
                cost[rows1, di] = 1e-9
        best = np.minimum(cost[:, 0], cost[:, 1]); bestd = np.where(cost[:, 0] <= cost[:, 1], 1, -1)
        newcol = cur[:, j].copy(); tot = 0.0; nflip = 0
        for y in range((A + 15) // 16):
            rr = np.arange(16 * y, min(16 * y + 16, A)); x = rr - 16 * y
            lsb = cur[rr, j] & 1
            cs = []
            for p in pats:
                mis = p[x] != lsb
                cs.append(best[rr[mis]].sum() if mis.any() else 0.0)
            pi = int(np.argmin(cs)); p = pats[pi]
            mis = p[x] != lsb
            if not np.isfinite(cs[pi]): raise RuntimeError(f"infeasible block y={y} col={j}")
            for r in rr[mis]:
                newcol[r] = (cur[r, j] + bestd[r]) % 256
            tot += cs[pi]; nflip += int(mis.sum())
        model.words[:, j] = newcol
        old = model.e2sq
        model.refresh()
        print(f"col {j}: flips {nflip}, predicted Δe2² {tot:.3e}, actual Δe2² {model.e2sq - old:.3e}, e2 {np.sqrt(model.e2sq):.6f} ({time.time()-t0:.0f}s)", flush=True)
    m1 = evaluate_words(inp, model.words, edges, 8)
    print("coded", {k: round(v, 6) for k, v in m1.items() if isinstance(v, float)}, flush=True)
    out = {k: z[k] for k in z.files}; out["angle_words"] = model.words
    np.savez(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
