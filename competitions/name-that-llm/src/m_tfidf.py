"""TF-IDF (char + word) + linear model, official train (+ optional Open Pool sample).

uv run python src/m_tfidf.py --pool 0          # train only
uv run python src/m_tfidf.py --pool 2500       # + 2500 pool passages / label
"""
import argparse
import time

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from cvlib import K, OUT, folds, load_pool, load_train, per_class_recall, report, sinkhorn

ap = argparse.ArgumentParser()
ap.add_argument("--pool", type=int, default=2500)
ap.add_argument("--pool_w", type=float, default=0.5, help="sample weight of pool rows (train = 1)")
ap.add_argument("--C", type=float, default=0.3)
ap.add_argument("--temp", type=float, default=0.1, help="softmax temperature on decision values")
ap.add_argument("--max_feat", type=int, default=400_000)
ap.add_argument("--tag", default="")
a = ap.parse_args()

tr = load_train()
pool = load_pool(a.pool) if a.pool > 0 else None
oof = np.zeros((len(tr), K))
t0 = time.time()
for f, (ti, vi) in enumerate(folds(tr)):
    texts = tr.text.iloc[ti].tolist()
    y = tr.y.values[ti]
    w = np.ones(len(ti))
    if pool is not None:
        texts += pool.text.tolist()
        y = np.concatenate([y, pool.y.values])
        w = np.concatenate([w, np.full(len(pool), a.pool_w)])
    vs = [
        TfidfVectorizer(analyzer="char", ngram_range=(1, 5), sublinear_tf=True, lowercase=False,
                        min_df=3, max_features=a.max_feat, dtype=np.float32),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                        min_df=2, max_features=a.max_feat // 2, token_pattern=r"(?u)\b\w+\b|[^\w\s]",
                        dtype=np.float32),
    ]
    Xa = hstack([v.fit_transform(texts) for v in vs]).tocsr()
    Xb = hstack([v.transform(tr.text.iloc[vi]) for v in vs]).tocsr()
    clf = LinearSVC(C=a.C, class_weight="balanced").fit(Xa, y, sample_weight=w)
    d = clf.decision_function(Xb) / a.temp
    e = np.exp(d - d.max(1, keepdims=True))
    oof[vi] = e / e.sum(1, keepdims=True)
    print(f"fold {f} done {time.time() - t0:.0f}s", flush=True)

name = f"tfidf_svc_pool{a.pool}_w{a.pool_w}_C{a.C}{a.tag}"
report(name, tr, oof)
report(name + "+sinkhorn", tr, oof, post=lambda p: sinkhorn(p, 1.0))
print(per_class_recall(tr.y.values, oof).sort_values().to_string())
np.save(OUT / f"oof_{name}.npy", oof)
