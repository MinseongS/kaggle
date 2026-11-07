import pandas as pd, numpy as np, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from scipy.sparse import hstack
from scipy.optimize import linear_sum_assignment
tr = pd.read_csv("data/train.csv")
labels = sorted(tr.model.unique()); y = tr.model.map({m:i for i,m in enumerate(labels)}).values
fam = lambda m: m.split("-")[0].rstrip("0123456789.")
oof = np.zeros((len(tr), len(labels)))
for f,(a,b) in enumerate(GroupKFold(5).split(tr, y, tr.prompt_id)):
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(1,5), sublinear_tf=True, min_df=2, lowercase=False)
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1,2), sublinear_tf=True, min_df=2, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    Xa = hstack([vc.fit_transform(tr.text.iloc[a]), vw.fit_transform(tr.text.iloc[a])]).tocsr()
    Xb = hstack([vc.transform(tr.text.iloc[b]), vw.transform(tr.text.iloc[b])]).tocsr()
    clf = LogisticRegression(C=10, max_iter=3000).fit(Xa, y[a])
    oof[b] = clf.predict_proba(Xb)
pred = oof.argmax(1)
print("balanced acc (argmax):", round(balanced_accuracy_score(y, pred),4))
yf = np.array([fam(labels[i]) for i in y]); pf = np.array([fam(labels[i]) for i in pred])
print("family acc:", round((yf==pf).mean(),4))
# oracle: if prompt groups known, Hungarian per prompt
ph = pred.copy()
for p, idx in tr.groupby("prompt_id").indices.items():
    r, c = linear_sum_assignment(-np.log(oof[idx]+1e-9)); ph[idx[r]] = c
print("balanced acc (Hungarian w/ known prompt groups):", round(balanced_accuracy_score(y, ph),4))
cm = confusion_matrix(y, pred)
rec = pd.Series(cm.diagonal()/cm.sum(1), index=labels).round(2)
print(rec.sort_values().to_string())
cmd = pd.DataFrame(cm, index=labels, columns=[l[:10] for l in labels]); print(cmd.to_string())
np.save("outputs/oof_tfidf_lr.npy", oof)
