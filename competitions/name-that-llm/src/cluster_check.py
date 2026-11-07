import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
tr = pd.read_csv("data/train.csv")
X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.05).fit_transform(tr.text)
for k in [150]:
    lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X.toarray())
    print("ARI:", round(adjusted_rand_score(tr.prompt_id, lab),4))
    ct = pd.crosstab(lab, tr.prompt_id)
    print("purity:", round(ct.max(1).sum()/len(tr),4), "| clusters with exactly 1 prompt:", (ct.gt(0).sum(1)==1).mean().round(3))
    # nearest neighbour same-prompt rate
S = (X @ X.T).toarray(); np.fill_diagonal(S, -1)
print("NN same prompt:", (tr.prompt_id.values[S.argmax(1)] == tr.prompt_id.values).mean().round(4))
