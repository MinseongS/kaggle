"""inv_angle_lp.py 가 만든 캐시(err0, tol, B)로 cutting-plane LP 를 풀고 새 각도로 kmb 를 쓴다.
Usage: uv run python src/inv_lp_solve.py <in.kmb> <out.kmb> [--shrink s]"""
import argparse, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, str(Path(__file__).parent))
from inv_angle_lp import load_lines, circuit_from_lines, decimal_str, ROOT

def solve(err0, tol, B, iters=30, init=4000, bound=None):
    n = B.shape[0]
    marg = tol - np.abs(err0)
    act = set(np.argsort(marg)[:init].tolist())
    Bt = B.T.astype(np.float64)
    d = np.zeros(n); m = None
    for it in range(iters):
        idx = np.array(sorted(act))
        Bi = Bt[idx]
        # variables [d (n), m]; maximize m -> minimize -m
        A1 = np.hstack([Bi, np.ones((len(idx), 1))]); b1 = tol[idx] - err0[idx]
        A2 = np.hstack([-Bi, np.ones((len(idx), 1))]); b2 = tol[idx] + err0[idx]
        c = np.zeros(n + 1); c[-1] = -1
        bnds = [(-bound, bound) if bound else (None, None)] * n + [(None, None)]
        r = linprog(c, A_ub=np.vstack([A1, A2]), b_ub=np.concatenate([b1, b2]), bounds=bnds, method="highs")
        if r.status != 0:
            print("lp fail", r.message); break
        d, m = r.x[:n], r.x[-1]
        e = err0 + Bt @ d
        full = tol - np.abs(e)
        viol = np.flatnonzero(full < m - 1e-6 * abs(m))
        print(f"it {it}: lp m={m:.6g} true min={full.min():.6g} active={len(idx)} viol={len(viol)}", flush=True)
        if len(viol) == 0: break
        act.update(np.argsort(full)[:2000].tolist())
    return d, m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--cache", default=str(ROOT / "outputs/research/inv_lp_cache.npz"))
    ap.add_argument("--shrink", type=float, default=0.98, help="실제로 적용할 비율 (반올림 여유)")
    a = ap.parse_args()
    z = np.load(a.cache)
    err0, tol, B, rot = z["err0"], z["tol"], z["B"], z["rot"]
    print("base min margin", (tol - np.abs(err0)).min())
    d, m = solve(err0, tol, B)
    lines = load_lines(a.src)
    for j, li in enumerate(rot):
        p = lines[li].split()
        du = int(round(d[j]))  # units of 2^-64 turn = 2^-63 half-turn
        v = Fraction(p[2]) + Fraction(du, 2**63)
        lines[li] = f"{p[0]} {p[1]} {decimal_str(v)}"
    e = err0 + B.T.astype(np.float64) @ np.rint(d)
    print("predicted min margin after rounding", (tol - np.abs(e)).min(), "gain x", (tol - np.abs(e)).min() / (tol - np.abs(err0)).min())
    c = circuit_from_lines(lines)
    c.to_file(a.out, format="kmb")
    print("wrote", a.out, "d range", d.min(), d.max())

if __name__ == "__main__":
    main()
