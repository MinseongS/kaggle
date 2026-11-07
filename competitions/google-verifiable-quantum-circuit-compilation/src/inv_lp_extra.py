"""invsqrt: 기존 회전 41개 + 추가 후보 회귀변수 (입력 x 비트에 새 Z_POW) 로 LP 여유 비교.
Usage: uv run python src/inv_lp_extra.py [--cache ...] [--cands 0-21] [--greedy N]"""
import argparse, sys, time, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from inv_lp_solve import solve
ap = argparse.ArgumentParser(); ap.add_argument("--cache", default="outputs/research/inv_lp_cache_y33.npz")
ap.add_argument("--greedy", type=int, default=2)
a = ap.parse_args()
z = np.load(a.cache); err0, tol, B = z["err0"], z["tol"], z["B"]
N = len(err0); xs = np.arange(1, N + 1, dtype=np.int64)
xb = ((xs[None, :] >> np.arange(22)[:, None]) & 1).astype(np.int8)
def cost_C(m, nrot, base_tof=1281.472, other_T=11.5):
    eps = 2 * math.pi * (m * 2.0**-64) / (2 * nrot)
    T = other_T + nrot * (0.56 * math.log2(1 / eps) + 5.3)
    return base_tof + T / 4
import io, contextlib
def lp(Bm):
    with contextlib.redirect_stdout(io.StringIO()):
        d, m = solve(err0, tol, Bm)
    return m
t0 = time.time()
m0 = lp(B); print(f"base m {m0:.4g} C {cost_C(m0, 41):.2f} ({time.time()-t0:.0f}s)", flush=True)
cur = B; nrot = 41; chosen = []
for g in range(a.greedy):
    res = []
    for i in range(22):
        if i in chosen: continue
        m = lp(np.vstack([cur, xb[i:i+1]]))
        res.append((m, i)); print(f"  +x{i}: m {m:.4g} gain x{m/m0:.3f} C {cost_C(m, nrot+1):.2f}", flush=True)
    m, i = max(res)
    chosen.append(i); cur = np.vstack([cur, xb[i:i+1]]); nrot += 1
    print(f"greedy {g}: add x{i}, m {m:.4g}, C {cost_C(m, nrot):.2f} ({time.time()-t0:.0f}s)", flush=True)
