# ntl-lite: TF-IDF (char 1-5 + word 1-2) LinearSVC on official train + Open Pool sample, + Sinkhorn.
# Local grouped CV (src/m_tfidf.py --pool 2500): 0.4708 (0.4816 with sinkhorn).
import glob, numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

LABELS = ["claude-fable-5","claude-opus-4-6","claude-opus-4-8","claude-opus-5","claude-sonnet-5",
 "gemini-3.1-pro-preview","gemini-3.6-flash","gemini-3.7-flash","gemini-3.8-flash","glm-5.3","glm-5.3-flash",
 "gpt-5.5","gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra","gpt-6-astra","kimi-k3","qwen3.8-max"]
L2I = {m: i for i, m in enumerate(LABELS)}
comp = glob.glob("/kaggle/input/**/name-that-llm/train.csv", recursive=True)[0].rsplit("/", 1)[0]
tr = pd.read_csv(f"{comp}/train.csv"); te = pd.read_csv(f"{comp}/test.csv")
pp = glob.glob("/kaggle/input/**/ntl_open_pool.parquet", recursive=True)[0]
p = pd.read_parquet(pp)
p = p[~(p.flag_near_train_prompt | p.flag_train_prompt_regen | p.flag_gemini37_routing_risk)]
rng = np.random.default_rng(0)
p = p.assign(_k=np.log(rng.random(len(p))) / p.match_weight.clip(lower=1e-6).values)
p = p.sort_values("_k", ascending=False).groupby("model").head(2500)
print("train", len(tr), "pool", len(p), "test", len(te), flush=True)

texts = tr.text.tolist() + p.text.tolist()
y = np.concatenate([tr.model.map(L2I).values, p.model.map(L2I).values])
w = np.concatenate([np.ones(len(tr)), np.full(len(p), 0.5)])
vs = [TfidfVectorizer(analyzer="char", ngram_range=(1, 5), sublinear_tf=True, lowercase=False, min_df=3,
                      max_features=400_000, dtype=np.float32),
      TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, lowercase=False, min_df=2,
                      max_features=200_000, token_pattern=r"(?u)\b\w+\b|[^\w\s]", dtype=np.float32)]
Xa = hstack([v.fit_transform(texts) for v in vs]).tocsr()
Xt = hstack([v.transform(te.text) for v in vs]).tocsr()
clf = LinearSVC(C=0.3, class_weight="balanced").fit(Xa, y, sample_weight=w)
d = clf.decision_function(Xt) / 0.1
P = np.exp(d - d.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
if len(te) >= 18:  # full sinkhorn toward uniform marginal
    Q = P.copy()
    for _ in range(50):
        Q = Q / Q.sum(0, keepdims=True) * (len(Q) / 18); Q = Q / Q.sum(1, keepdims=True)
    P = Q
sub = pd.DataFrame({"id": te.id, "model": [LABELS[i] for i in P.argmax(1)]})
sub.to_parquet("submission.parquet", index=False)
print(sub.model.value_counts().to_string())
