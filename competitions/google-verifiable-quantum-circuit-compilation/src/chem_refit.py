"""화학(dfthcsos): 회로(각도 단어)는 고정하고 submission.npz 의 가중치 w (R,B,C), W (R,C) 만 재최적화해 lambda_eff 를 낮춘다.

T, Q 는 회로와 형상(R,B,C,brot)으로만 정해지므로 그대로이고, lambda_eff 만 바뀐다.
제약: error_1, error_2 <= 20 mHa, E_gap > 3.914898, Lambda > 49.
결과는 공식 evaluate_words 로 재확인한 뒤 npz 로 쓴다 (verifier 전체 실행은 별도).

Usage: uv run python src/chem_refit.py <src_dir_with_submission.npz> <out_npz> [--steps N] [--margin m]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/verifiers"))
from dfthcsos.verify import load_inputs, evaluate_words  # noqa: E402
from dfthcsos.scoreinfra.dfthc_math import reconstruct_u_vectors  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

torch.set_default_dtype(torch.float64)
THR, GAP_FLOOR = 0.02, 3.914898


def build(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    inputs = load_inputs(str(npz_path))
    words, edges, brot = d["angle_words"], d["tree_edges"], int(d["brot"])
    ref = inputs.ref
    N = ref["oei"].shape[0]
    u_all = reconstruct_u_vectors(words / (1 << (brot - 1)), edges)
    u_all = u_all / np.linalg.norm(u_all, axis=1, keepdims=True)
    R, B, C = inputs.w.shape
    v1 = u_all[:N]
    u2 = u_all[N:].reshape(R, B, N)
    tei = np.asarray(ref["tei"], dtype=np.float64)
    S = inputs.S
    h1_thc = ref["oei"] - 0.5 * np.einsum("prrq->pq", tei)
    const_h1 = h1_thc - 0.5 * ref["electrons"] * S + inputs.beta_1 * np.eye(N)
    # target T for error_2 (N^2 x N^2)
    T = tei.copy()
    di = np.arange(N)
    T[:, :, di, di] += 0.5 * S[:, :, None]
    T[di, di, :, :] += 0.5 * S[None, :, :]
    T = T.reshape(N * N, N * N)
    T = 0.5 * (T + T.T)  # X (x) X is symmetric in (pq)<->(rs)
    gap_const = inputs.energy_upper - ref["ecore"] + ref["electrons"] * inputs.beta_1
    return dict(d=d, inputs=inputs, words=words, edges=edges, brot=brot, v1=v1, u2=u2, T=T,
                Tn2=float((T * T).sum()), const_h1=const_h1, gap_const=gap_const, N=N)


class Model:
    def __init__(self, P):
        self.P = P
        self.u2 = torch.tensor(P["u2"])
        self.v1 = torch.tensor(P["v1"])
        self.T = torch.tensor(P["T"])
        self.const_h1 = torch.tensor(P["const_h1"])
        self.perm = None

    def metrics(self, w, W, set_perm=False):
        P, N = self.P, self.P["N"]
        X = torch.einsum("rbp,rbq,rbc->rcpq", self.u2, self.u2, w)  # (R,C,N,N)
        Y = X.reshape(-1, N * N)
        G = Y @ Y.T
        TY = Y @ self.T
        e2sq = (G * G).sum() - 2 * (TY * Y).sum() + P["Tn2"]
        e2 = torch.sqrt(torch.clamp(e2sq, min=1e-30))
        h1p = self.const_h1 - torch.einsum("rc,rcpq->pq", W, X)
        h1p = 0.5 * (h1p + h1p.T)
        lam, vec = torch.linalg.eigh(h1p)
        if set_perm or self.perm is None:
            ov = np.abs(P["v1"] @ vec.detach().numpy())
            ri, ci = linear_sum_assignment(-ov)
            perm = np.empty(N, dtype=int); perm[ri] = ci
            self.perm = torch.tensor(perm)
        lam_rows = lam[self.perm]
        rec = torch.einsum("j,jp,jq->pq", lam_rows, self.v1, self.v1)
        e1 = torch.linalg.norm(rec - h1p)
        lam_h1 = lam.abs().sum()
        wB = W + w.sum(1)
        lam_h2 = 0.25 * ((w.abs().sum(1) + wB.abs()) ** 2).sum()
        Lam = lam_h1 + lam_h2
        gap = P["gap_const"] + (lam_h1 - lam.sum()) + 0.5 * (W ** 2).sum()
        leff = torch.sqrt(gap * (2 * Lam - gap))
        return dict(e1=e1, e2=e2, Lam=Lam, gap=gap, leff=leff)


def official(P, w, W):
    inp = P["inputs"]
    from dataclasses import replace
    inp2 = replace(inp, w=w, W=W)
    return evaluate_words(inp2, P["words"], P["edges"], P["brot"])


def feasible(m, margin):
    return (m["error_1"] <= THR - margin and m["error_2"] <= THR - margin
            and m["E_gap"] > GAP_FLOOR + margin and m["Lambda"] > 49.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=2e-9)
    ap.add_argument("--mu", type=float, default=1e5)
    a = ap.parse_args()
    src = Path(a.src)
    npz = src / "submission.npz" if src.is_dir() else src
    P = build(npz)
    inp = P["inputs"]
    m0 = official(P, inp.w, inp.W)
    print("official start", {k: m0[k] for k in ("error_1", "error_2", "Lambda", "E_gap", "lambda_eff")}, flush=True)
    M = Model(P)
    w = torch.tensor(inp.w, requires_grad=True)
    W = torch.tensor(inp.W, requires_grad=True)
    with torch.no_grad():
        t = M.metrics(w, W, set_perm=True)
    print("torch start", {k: float(v) for k, v in t.items()}, flush=True)
    # targets slightly inside the feasible region
    tgt_e = THR - 2e-6
    tgt_g = GAP_FLOOR + 2e-6
    opt = torch.optim.Adam([w, W], lr=a.lr)
    best = (float(m0["lambda_eff"]), inp.w.copy(), inp.W.copy())
    t0 = time.time()
    for it in range(a.steps):
        opt.zero_grad()
        m = M.metrics(w, W, set_perm=(it % 200 == 0))
        pen = (torch.relu(m["e1"] - tgt_e) ** 2 + torch.relu(m["e2"] - tgt_e) ** 2
               + torch.relu(tgt_g - m["gap"]) ** 2)
        loss = m["leff"] + a.mu * pen
        loss.backward()
        opt.step()
        if it % 20 == 0 or it == a.steps - 1:
            ok = m["e1"] <= THR and m["e2"] <= THR and m["gap"] > GAP_FLOOR
            print(f"{it:5d} leff={m['leff'].item():.6f} Lam={m['Lam'].item():.4f} gap={m['gap'].item():.7f} "
                  f"e1={m['e1'].item():.7f} e2={m['e2'].item():.7f} ok={bool(ok)} {time.time()-t0:.0f}s", flush=True)
        if it % 250 == 249 or it == a.steps - 1:
            wn, Wn = w.detach().numpy().copy(), W.detach().numpy().copy()
            mo = official(P, wn, Wn)
            if feasible(mo, a.margin) and mo["lambda_eff"] < best[0]:
                best = (mo["lambda_eff"], wn, Wn)
                print(f"   official feasible leff={mo['lambda_eff']:.6f} (e1 {mo['error_1']:.9f} e2 {mo['error_2']:.9f} gap {mo['E_gap']:.9f})", flush=True)
                save(P, a.out, wn, Wn)
            else:
                print(f"   official leff={mo['lambda_eff']:.6f} e1 {mo['error_1']:.9f} e2 {mo['error_2']:.9f} gap {mo['E_gap']:.9f} feasible={feasible(mo, a.margin)}", flush=True)
    print("best leff", best[0], "start", m0["lambda_eff"])


def save(P, out, w, W):
    d = P["d"]
    arrs = {k: d[k] for k in d.files}
    arrs["W"] = w  # lowercase w is stored as "W"
    arrs["H2_I"] = W + w.sum(axis=1)  # w_B = W + sum_b w
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrs)


if __name__ == "__main__":
    main()
