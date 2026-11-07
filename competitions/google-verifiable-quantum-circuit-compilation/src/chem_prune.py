"""공개 양자화 해에서 rho 마다 sum_c|w_b| 작은 b 를 잘라 B 를 줄이고 (C 는 0 패딩으로 늘릴 수 있음), 남은 단어는 고정한 채 w/W 재적합.
Usage: uv run python src/chem_prune.py <B> <C> [--iters 400]"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from chem_quant_eval import PUB, load_inputs, evaluate_words, h1_prime_of
from chem_shape_eval import refit, cost_model, REF_COST
from chem_onebody import optimize_onebody
from chem_refit import THR, GAP_FLOOR
from dataclasses import replace

ap = argparse.ArgumentParser(); ap.add_argument("B", type=int); ap.add_argument("C", type=int)
ap.add_argument("--iters", type=int, default=400); ap.add_argument("--out")
a = ap.parse_args()
d = np.load(PUB); N = 76; brot = int(d["brot"]); edges = d["tree_edges"]
inp = load_inputs(str(PUB))
w, W = inp.w, inp.W
R, B0, C0 = w.shape
words2 = d["angle_words"][N:].reshape(R, B0, 75)
sc = np.abs(w).sum(axis=2)
idx = np.sort(np.argsort(-sc, axis=1)[:, :a.B], axis=1)
w2 = np.take_along_axis(w, idx[:, :, None], axis=1)
words2 = np.take_along_axis(words2, idx[:, :, None], axis=1)
U2 = np.take_along_axis(np.asarray(inp.params["U"]), idx[:, :, None], axis=1)
print("dropped |w| fraction", 1 - np.abs(w2).sum() / np.abs(w).sum())
if a.C > C0:
    rng = np.random.default_rng(0)
    w2 = np.concatenate([w2, 1e-4 * rng.standard_normal((R, a.B, a.C - C0))], axis=2)
    W = np.concatenate([W, np.zeros((R, a.C - C0))], axis=1)
else:
    keep = np.argsort(-np.abs(w2).sum(axis=(0, 1)))[:a.C]; keep.sort()
    w2, W = w2[:, :, keep], W[:, keep]
params = dict(inp.params); params["U"] = U2
inp2 = replace(inp, w=w2, W=W, params=params)
words = np.vstack([d["angle_words"][:N], words2.reshape(-1, 75)])
m = evaluate_words(inp2, words, edges, brot)
print(f"pruned raw: e1={m['error_1']:.5f} e2={m['error_2']:.5f} Lam={m['Lambda']:.3f} gap={m['E_gap']:.5f} leff={m['lambda_eff']:.4f}", flush=True)
inp2 = refit(inp2, words, edges, brot, a.iters)
words[:N] = optimize_onebody(words[:N], h1_prime_of(inp2, words, edges, brot), edges, brot, sweeps=30, log=lambda s: None)
m = evaluate_words(inp2, words, edges, brot)
ok = m["error_1"] <= THR and m["error_2"] <= THR and m["E_gap"] > GAP_FLOOR and m["Lambda"] > 49
T, Q = cost_model(R, a.B, a.C)
print(f"[{R},{a.B},{a.C}] final: e1={m['error_1']:.5f} e2={m['error_2']:.5f} Lam={m['Lambda']:.3f} gap={m['E_gap']:.6f} leff={m['lambda_eff']:.4f} feasible={ok}")
print(f"[{R},{a.B},{a.C}] cost model T={T:.0f} Q={Q} -> predicted ratio {REF_COST/(m['lambda_eff']*T*Q):.3f} (public 11.043)")
