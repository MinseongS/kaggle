"""chem_refit 의 Adam 대신 활성 제약(e1,e2,gap) 영공간 투영 경사 + 최소노름 복원 (gradient projection).
Usage: uv run python src/chem_refit_sqp.py <src npz or dir> <out npz> [--iters N] [--step s]"""
import argparse, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from chem_refit import build, Model, official, feasible, save, THR, GAP_FLOOR, np, torch

def grads(M, x, shapes, keys=("leff", "e1", "e2", "gap"), set_perm=False):
    nw = int(np.prod(shapes[0]))
    w = torch.tensor(x[:nw].reshape(shapes[0]), requires_grad=True)
    W = torch.tensor(x[nw:].reshape(shapes[1]), requires_grad=True)
    m = M.metrics(w, W, set_perm=set_perm)
    G = []
    for k in keys:
        gw, gW = torch.autograd.grad(m[k], [w, W], retain_graph=True, allow_unused=True)
        gW = torch.zeros_like(W) if gW is None else gW
        G.append(torch.cat([gw.flatten(), gW.flatten()]).numpy())
    return {k: float(v) for k, v in m.items()}, np.array(G)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--slack", type=float, default=3e-9)
    ap.add_argument("--wmask", default="all", help="all | W | w")
    a = ap.parse_args()
    src = Path(a.src); npz = src / "submission.npz" if src.is_dir() else src
    P = build(npz); M = Model(P); inp = P["inputs"]
    shapes = (inp.w.shape, inp.W.shape)
    x = np.concatenate([inp.w.ravel(), inp.W.ravel()])
    nw = inp.w.size
    mask = np.ones_like(x)
    if a.wmask == "W": mask[:nw] = 0
    if a.wmask == "w": mask[nw:] = 0
    split = lambda x: (x[:nw].reshape(shapes[0]).copy(), x[nw:].reshape(shapes[1]).copy())
    m0 = official(P, *split(x))
    best = m0["lambda_eff"]; print("start", best, flush=True)
    tgt = np.array([THR - a.slack, THR - a.slack, GAP_FLOOR + a.slack])
    step = a.step; t0 = time.time()
    for it in range(a.iters):
        m, G = grads(M, x, shapes, set_perm=(it % 10 == 0))
        g0 = G[0] * mask; A = G[1:] * mask
        # null-space projection of -g0 w.r.t. active constraint gradients
        lam = np.linalg.solve(A @ A.T, A @ g0)
        d = -(g0 - A.T @ lam)
        if np.linalg.norm(d) == 0: break
        xn = x + step * d / np.linalg.norm(d)
        # restoration (equality on all three: e1=e2=tgt, gap=tgt)
        for _ in range(6):
            mn, Gn = grads(M, xn, shapes, keys=("e1", "e2", "gap"))
            c = np.array([mn["e1"], mn["e2"], mn["gap"]]) - tgt
            if np.all(np.abs(c) < 1e-10): break
            An = Gn * mask
            xn = xn - An.T @ np.linalg.solve(An @ An.T, c)
        mo = official(P, *split(xn))
        ok = feasible(mo, 0.0)
        if ok and mo["lambda_eff"] < best:
            x = xn; best = mo["lambda_eff"]; save(P, a.out, *split(x)); step *= 1.3
            tag = "acc"
        else:
            step *= 0.5; tag = "rej"
        print(f"{it:4d} {tag} leff={mo['lambda_eff']:.7f} best={best:.7f} e1={mo['error_1']:.10f} e2={mo['error_2']:.10f} gap={mo['E_gap']:.10f} step={step:.3g} |d|res={np.linalg.norm(d):.3f} {time.time()-t0:.0f}s", flush=True)
        if step < 1e-6: break
    print("final best", best, "start", m0["lambda_eff"])

if __name__ == "__main__":
    main()
