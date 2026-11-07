"""Linear head on frozen LM embeddings (from src/embed.py).

uv run python src/m_emb.py --tag Qwen3-1.7B-Base --layers 0,1,2,3 --pool 2500
`--layers` are indices into the saved layer list.
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cvlib import K, L2I, OUT, folds, load_train, per_class_recall, report, sinkhorn

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="Qwen3-1.7B-Base")
ap.add_argument("--layers", default="all")
ap.add_argument("--pool", type=int, default=2500)
ap.add_argument("--pool_w", type=float, default=0.5)
ap.add_argument("--C", type=float, default=0.05)
ap.add_argument("--per_layer", action="store_true", help="quick train-only scan of each layer")
a = ap.parse_args()

tr = load_train()
Etr = np.load(OUT / f"emb_{a.tag}_train.npy").astype(np.float32)
nl = Etr.shape[1]


def run(sel, use_pool, name):
    Xtr = Etr[:, sel].reshape(len(Etr), -1)
    if use_pool:
        Ep = np.load(OUT / f"emb_{a.tag}_pool{a.pool}.npy").astype(np.float32)[:, sel].reshape(-1, Xtr.shape[1])
        yp = pd.read_csv(OUT / f"emb_{a.tag}_pool{a.pool}_ids.csv").model.map(L2I).values
    oof = np.zeros((len(tr), K))
    t0 = time.time()
    for ti, vi in folds(tr):
        X, y, w = Xtr[ti], tr.y.values[ti], np.ones(len(ti))
        if use_pool:
            X = np.vstack([X, Ep]); y = np.concatenate([y, yp]); w = np.concatenate([w, np.full(len(yp), a.pool_w)])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=a.C, max_iter=300).fit(sc.transform(X), y, sample_weight=w)
        oof[vi] = clf.predict_proba(sc.transform(Xtr[vi]))
    print(f"[{time.time() - t0:.0f}s]", end=" ")
    report(name, tr, oof)
    return oof


if a.per_layer:
    for j in range(nl):
        run([j], False, f"{a.tag}_L{j}_train")
else:
    sel = list(range(nl)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    name = f"emb_{a.tag}_L{'-'.join(map(str, sel))}_pool{a.pool if a.pool else 0}_C{a.C}"
    oof = run(sel, a.pool > 0, name)
    report(name + "+sinkhorn", tr, oof, post=lambda p: sinkhorn(p, 1.0))
    print(per_class_recall(tr.y.values, oof).sort_values().to_string())
    np.save(OUT / f"oof_{name}.npy", oof)
