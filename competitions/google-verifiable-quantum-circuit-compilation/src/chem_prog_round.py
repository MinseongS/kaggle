"""chem 점진 코딩 반올림: 지정 열들의 두-체 행 각도를 연속 변수로 풀어 (w, W 와 함께) λ_eff 최소화 (e2, gap 페널티),
한 열씩 '이전 배치 같은 슬롯 LSB ⊕ 16행 블록 아핀' 코드로 반올림·고정하고 남은 연속 각도로 보상.
one-body 행 (a<76) 은 단순 최소이동 코딩 (e1 은 마지막에 확인).
Usage: uv run python src/chem_prog_round.py <submission.npz> <out.npz> [--cols 5-14] [--iters 40]"""
from __future__ import annotations
import argparse, sys, time, math
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/verifiers"))
from dfthcsos.verify import load_inputs, evaluate_words  # noqa: E402
torch.set_default_dtype(torch.float64)
THR, GAP_FLOOR = 0.02, 3.914898
GAUGE_BAD = {(63, 64), (64, 63), (191, 192), (192, 191)}


def decode_t(theta, edges, N=76):
    """theta: (n, 75) 라디안 (= π·word/128). 반환 (n, N) 단위벡터."""
    n = theta.shape[0]
    cols = [torch.zeros(n) for _ in range(N)]
    cols[0] = torch.ones(n)
    c, s = torch.cos(theta), torch.sin(theta)
    for k, (p, ch) in enumerate(edges):
        pv, cv = cols[p], cols[ch]
        cols[p] = c[:, k] * pv - s[:, k] * cv
        cols[ch] = s[:, k] * pv + c[:, k] * cv
    return torch.stack(cols, 1)


class Problem:
    def __init__(self, npz):
        self.z = np.load(npz); self.inp = inp = load_inputs(npz)
        self.words = self.z["angle_words"].astype(np.int64).copy(); self.edges = [tuple(map(int, e)) for e in self.z["tree_edges"]]
        ref = inp.ref; tei = np.asarray(ref["tei"], np.float64); S = inp.S; N = tei.shape[0]; self.N = N
        R, B, C = inp.w.shape; self.R, self.B, self.C = R, B, C
        T = tei.copy(); di = np.arange(N)
        T[:, :, di, di] += 0.5 * S[:, :, None]; T[di, di, :, :] += 0.5 * S[None, :, :]
        T = T.reshape(N * N, N * N); self.T = torch.tensor(0.5 * (T + T.T)); self.Tn2 = float((self.T ** 2).sum())
        h1_thc = ref["oei"] - 0.5 * np.einsum("prrq->pq", tei)
        self.const_h1 = torch.tensor(h1_thc - 0.5 * ref["electrons"] * S + inp.beta_1 * np.eye(N))
        self.gap_const = inp.energy_upper - ref["ecore"] + ref["electrons"] * inp.beta_1
        self.w = torch.tensor(np.ascontiguousarray(inp.w), requires_grad=True)
        self.W = torch.tensor(np.ascontiguousarray(inp.W), requires_grad=True)

    def metrics(self, theta2):
        N, R, B, C = self.N, self.R, self.B, self.C
        u = decode_t(theta2, self.edges).reshape(R, B, N)
        X = torch.einsum("rbp,rbq,rbc->rcpq", u, u, self.w)
        Y = X.reshape(-1, N * N); G = Y @ Y.T
        e2 = torch.sqrt(torch.clamp((G * G).sum() - 2 * ((Y @ self.T) * Y).sum() + self.Tn2, min=1e-30))
        h1p = self.const_h1 - torch.einsum("rc,rcpq->pq", self.W, X); h1p = 0.5 * (h1p + h1p.T)
        lam = torch.linalg.eigvalsh(h1p); lam_h1 = lam.abs().sum(); wB = self.W + self.w.sum(1)
        Lam = lam_h1 + 0.25 * ((self.w.abs().sum(1) + wB.abs()) ** 2).sum()
        gap = self.gap_const + (lam_h1 - lam.sum()) + 0.5 * (self.W ** 2).sum()
        return e2, gap, Lam, torch.sqrt(gap * (2 * Lam - gap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("out")
    ap.add_argument("--cols", default="5-14"); ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--mu", type=float, default=1e6)
    a = ap.parse_args()
    lo, hi = map(int, a.cols.split("-")); cols = list(range(lo, hi + 1))
    P = Problem(a.npz); N1 = 76; A = P.words.shape[0]
    th_fixed = torch.tensor(P.words[N1:] * math.pi / 128.0)
    free = torch.zeros_like(th_fixed, dtype=torch.bool); free[:, cols] = True
    th_var = th_fixed.clone().requires_grad_(True)
    def theta(): return torch.where(free, th_var, th_fixed)
    with torch.no_grad():
        e2, gap, Lam, le = P.metrics(theta())
    print(f"start e2 {e2.item():.6f} gap {gap.item():.6f} leff {le.item():.5f}", flush=True)
    tgt_e, tgt_g = THR - 1e-5, GAP_FLOOR + 1e-5
    def opt_round(iters):
        params = [th_var, P.w, P.W]
        opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            e2, gap, Lam, le = P.metrics(theta())
            loss = le + a.mu * (torch.relu(e2 - tgt_e) ** 2 + torch.relu(tgt_g - gap) ** 2)
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad(): return [t.item() for t in P.metrics(theta())]
    t0 = time.time()
    e2, gap, Lam, le = opt_round(a.iters)
    print(f"relaxed: e2 {e2:.6f} gap {gap:.6f} leff {le:.5f} ({time.time()-t0:.0f}s)", flush=True)
    xs = np.arange(16)
    pats = np.array([((c0 + bin(x & cm).count("1")) & 1) for c0 in (0, 1) for cm in range(16) for x in xs]).reshape(32, 16)
    words = P.words.copy()
    for j in cols:
        cont = th_var.detach().numpy()[:, j] * 128 / math.pi  # 두-체 행 연속 단어값
        full = np.concatenate([words[:N1, j].astype(np.float64), cont])  # one-body 는 현재 단어
        ref = words[:, j - 5] & 1
        newcol = words[:, j].copy()
        for y in range((A + 15) // 16):
            rr = np.arange(16 * y, min(16 * y + 16, A)); x = rr - 16 * y
            best = None
            for p in pats:
                want = p[x] ^ ref[rr]  # 목표 LSB
                tot = 0.0; ch = []
                for r, wbit in zip(rr, want):
                    v = full[r]; fl = math.floor(v)
                    cands = [fl, fl + 1] if (fl & 1) == wbit else [fl - 1, fl + 1] if False else [fl + 1 if ((fl + 1) & 1) == wbit else fl, fl - 1 if ((fl - 1) & 1) == wbit else fl + 2]
                    cands = [c for c in {fl - 1, fl, fl + 1, fl + 2} if (c & 1) == wbit]
                    # gauge 유지: 원 단어와 같은 bit7^bit6
                    g0 = ((int(words[r, j]) >> 7) ^ (int(words[r, j]) >> 6)) & 1
                    cands = [c for c in cands if ((((c % 256) >> 7) ^ ((c % 256) >> 6)) & 1) == g0] or cands
                    cbest = min(cands, key=lambda c: abs(c - v))
                    tot += (cbest - v) ** 2; ch.append(cbest % 256)
                if best is None or tot < best[0]: best = (tot, ch)
            newcol[rr] = best[1]
        words[:, j] = newcol
        with torch.no_grad():
            th_fixed[:, j] = torch.tensor(newcol[N1:] * math.pi / 128.0)
            free[:, j] = False
        if any(free.any(0)):
            e2, gap, Lam, le = opt_round(a.iters)
        else:
            e2, gap, Lam, le = opt_round(a.iters)  # 가중치만
        print(f"commit col {j}: e2 {e2:.6f} gap {gap:.6f} leff {le:.5f} ({time.time()-t0:.0f}s)", flush=True)
    out = {k: P.z[k] for k in P.z.files}
    out["angle_words"] = words
    out["W"] = P.w.detach().numpy(); out["H2_I"] = (P.W + P.w.sum(1)).detach().numpy()
    np.savez(a.out, **out)
    m = evaluate_words(load_inputs(a.out), words, P.z["tree_edges"], 8)
    print("official", {k: round(v, 6) for k, v in m.items() if isinstance(v, float)}, flush=True)


if __name__ == "__main__":
    main()
