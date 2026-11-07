"""Fit the final embedding head on all train + pool and save for the Kaggle kernel."""
import json, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from cvlib import L2I, LABELS, OUT, load_train
tag, sel, C = "Qwen3-1.7B-Base", [0, 1, 2, 6], 0.02
POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 2500   # uv run python src/fit_emb_head.py 6000
tr = load_train()
Xt = np.load(OUT / f"emb_{tag}_train.npy").astype(np.float32)[:, sel].reshape(len(tr), -1)
Xp = np.load(OUT / f"emb_{tag}_pool{POOL}.npy").astype(np.float32)[:, sel].reshape(-1, Xt.shape[1])
yp = pd.read_csv(OUT / f"emb_{tag}_pool{POOL}_ids.csv").model.map(L2I).values
X = np.vstack([Xt, Xp]); y = np.concatenate([tr.y.values, yp]); w = np.concatenate([np.ones(len(tr)), np.full(len(yp), 0.5)])
sc = StandardScaler().fit(X)
clf = LogisticRegression(C=C, max_iter=500).fit(sc.transform(X), y, sample_weight=w)
d = OUT / ("ds_emb_head" if POOL == 2500 else f"ds_emb_head_p{POOL}"); d.mkdir(exist_ok=True)
np.savez(d / "emb_head.npz", mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32),
         coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32))
json.dump({"model": tag, "hidden_layers": [2, 4, 6, 20], "labels": LABELS, "C": C, "max_len": 384,
           "pool": "mean over tokens, first token dropped", "open_pool_per_label": POOL}, open(d / "emb_head.json", "w"), indent=1)
print("saved", d)
