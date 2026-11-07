"""Log-prob blend of saved OOFs, with LB-noise check vs the best single member.

uv run python src/blend.py oof_a.npy:1.0 oof_b.npy:0.5 [--grid]
"""
import argparse
import itertools

import numpy as np

from cvlib import OUT, lb_gap_sd, load_train, report, sinkhorn

ap = argparse.ArgumentParser()
ap.add_argument("members", nargs="+")
ap.add_argument("--grid", action="store_true", help="search weights over 0..2 step 0.25 (by full bacc)")
a = ap.parse_args()

tr = load_train()
y = tr.y.values
names, ws, P = [], [], []
for m in a.members:
    n, w = (m.split(":") + ["1"])[:2]
    names.append(n); ws.append(float(w)); P.append(np.log(np.load(OUT / n) + 1e-9))


def mix(w):
    lp = sum(wi * p for wi, p in zip(w, P)) / max(sum(w), 1e-9)
    e = np.exp(lp - lp.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


if a.grid:
    best = max(itertools.product(np.arange(0, 2.01, 0.25), repeat=len(P)),
               key=lambda w: (sum(w) > 0) and (mix(w).argmax(1) == y).astype(float).mean() * 0
               + __import__("sklearn.metrics", fromlist=["x"]).balanced_accuracy_score(y, mix(w).argmax(1)))
    ws = list(best)
    print("grid best weights:", dict(zip(names, ws)))

for n, p in zip(names, P):
    report(n, tr, np.exp(p))
b = mix(ws)
report("BLEND", tr, b)
report("BLEND+sinkhorn", tr, b, post=lambda p: sinkhorn(p, 1.0))
best_single = max(P, key=lambda p: (p.argmax(1) == y).mean())
print("LB sd of (blend - best single) on ~1200 public rows:", round(lb_gap_sd(b, np.exp(best_single), y), 4))
