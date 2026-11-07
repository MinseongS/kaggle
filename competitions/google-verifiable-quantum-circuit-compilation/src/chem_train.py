"""starter kit trainer(JAX)로 새 형상 (R,B,C) 인수분해를 학습한다 (공개 best npz 에서 시드).
rho 축소 시 기여(Σ|W|) 작은 rho 부터 버리고, B/C 는 starter 의 load_seed_params 규칙(앞에서 자르기/랜덤 패딩).
Usage: uv run python src/chem_train.py <seed.npz> <R> <B> <C> <epochs> <out.npz> [--lr 1e-4] [--rho 1e-4] [--eps 0.010]"""
import argparse, os, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SK = ROOT / "data/starter_kit/starter_kit"
sys.path.insert(0, str(SK))
import jax
jax.config.update("jax_enable_x64", True)
from starter.task1_factorization import train_df_thc_qaggle as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("seed"); ap.add_argument("R", type=int); ap.add_argument("B", type=int); ap.add_argument("C", type=int)
ap.add_argument("epochs", type=int); ap.add_argument("out")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rho", type=float, default=1e-4)
ap.add_argument("--eps", type=float, default=0.010)
ap.add_argument("--print", type=int, default=500)
a = ap.parse_args()

d = np.load(a.seed, allow_pickle=False)
U, w, H2I = d["U"], d["W"], d["H2_I"]
R0 = U.shape[0]
if a.R < R0:
    score = np.abs(w).sum(axis=(1, 2))  # rho 기여
    keep = np.sort(np.argsort(-score)[:a.R])
    print("keep rho", keep, "scores", np.round(score, 3))
    U, w, H2I = U[keep], w[keep], H2I[keep]
if a.B < U.shape[1]:
    # rho 마다 sum_c |w_b| 가 큰 b 를 남긴다
    sc = np.abs(w).sum(axis=2)
    idx = np.sort(np.argsort(-sc, axis=1)[:, :a.B], axis=1)
    U = np.take_along_axis(U, idx[:, :, None], axis=1)
    w_full = w
    w = np.take_along_axis(w, idx[:, :, None], axis=1)
    H2I = H2I - w_full.sum(1) + w.sum(1)  # W(=H2_I - sum_b w) 유지
    print("kept B", a.B, "dropped weight frac", 1 - np.abs(w).sum() / np.abs(w_full).sum())
tmp = Path(tempfile.mkdtemp()) / "seed.npz"
np.savez(tmp, U=U, W=w, H2_I=H2I, H1_I=d["H1_I"], Hsym_lower=d["Hsym_lower"])
out = Path(a.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
T.train_df_thc(fcidump_file=T.PACKAGED_FCIDUMP, upper_energy=-22140.27335, rank=a.R, bases=a.B, copies=a.C,
               epochs=a.epochs, lr=a.lr, epsilon_dfthc=a.eps, rho=a.rho, seed=0, print_interval=a.print,
               seed_solution_file=str(tmp), output_file=str(out))
