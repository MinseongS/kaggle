"""e013: family-specialist heads on Qwen3-1.7B embeddings (train + pool6000), mixed into the
within-family conditional of the e011 blend. uv run python src/fam_head.py"""
import time, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from cvlib import FAM, K, L2I, OUT, bacc, folds, load_train, m_cut_score, per_class_recall
tag, sel, POOL = "Qwen3-1.7B-Base", [0, 1, 2, 6], 6000
tr = load_train(); y = tr.y.values
Xt = np.load(OUT / f"emb_{tag}_train.npy").astype(np.float32)[:, sel].reshape(len(tr), -1)
Xp = np.load(OUT / f"emb_{tag}_pool{POOL}.npy").astype(np.float32)[:, sel].reshape(-1, Xt.shape[1])
yp = pd.read_csv(OUT / f"emb_{tag}_pool{POOL}_ids.csv").model.map(L2I).values
A = np.log(np.load(OUT / "oof_tfidf_svc_pool6000_w0.5_C0.3.npy") + 1e-9)
B = np.log(np.load(OUT / f"oof_emb_{tag}_L0-1-2-6_pool6000_C0.02.npy") + 1e-9)
lp = 0.5 * A + 0.5 * B; base = np.exp(lp - lp.max(1, keepdims=True)); base /= base.sum(1, keepdims=True)
fams = [f for f in np.unique(FAM) if (FAM == f).sum() > 1]
for C in [0.02, 0.1]:
    H = {f: np.zeros((len(tr), (FAM == f).sum())) for f in fams}
    t0 = time.time()
    for ti, vi in folds(tr):
        for f in fams:
            ks = np.where(FAM == f)[0]; remap = {k: j for j, k in enumerate(ks)}
            mt = np.isin(y[ti], ks); mp = np.isin(yp, ks)
            X = np.vstack([Xt[ti][mt], Xp[mp]]); yy = np.array([remap[v] for v in np.concatenate([y[ti][mt], yp[mp]])])
            w = np.concatenate([np.ones(mt.sum()), np.full(mp.sum(), 0.5)])
            sc = StandardScaler().fit(X)
            clf = LogisticRegression(C=C, max_iter=300).fit(sc.transform(X), yy, sample_weight=w)
            H[f][vi] = clf.predict_proba(sc.transform(Xt[vi]))
    print(f"C={C} [{time.time()-t0:.0f}s]")
    for a in [0.0, 0.3, 0.5, 0.7, 1.0]:
        p = base.copy()
        for f in fams:
            ks = np.where(FAM == f)[0]; s = p[:, ks].sum(1, keepdims=True)
            q = np.log(p[:, ks] / s + 1e-9) * (1 - a) + np.log(H[f] + 1e-9) * a
            q = np.exp(q - q.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
            p[:, ks] = s * q
        print(f"  a={a} bacc={bacc(y, p):.4f} m2={m_cut_score(tr, p, draws=30):.4f}", flush=True)
        if a == 0.5: np.save(OUT / f"oof_famhead_C{C}_a0.5.npy", p)
