"""e014: ozer672 식 부분 sinkhorn (strength s, target_mix rho) 손잡이를 자체 OOF(e011)로 스캔.
m2 컷 + (균형 / Dirichlet(2) 불균형 클래스 서브샘플). 공개 파이프라인과 calibration이 달라 방향 참고용 proxy.
uv run python src/sk_scan.py
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score as bas
from cvlib import OUT, K, load_train

tr = load_train(); y = tr.y.values
A = np.log(np.load(OUT / "oof_tfidf_svc_pool6000_w0.5_C0.3.npy") + 1e-9)
B = np.log(np.load(OUT / "oof_emb_Qwen3-1.7B-Base_L0-1-2-6_pool6000_C0.02.npy") + 1e-9)
lp = (A + B) / 2; P0 = np.exp(lp - lp.max(1, keepdims=True)); P0 /= P0.sum(1, keepdims=True)
prior = np.bincount(y, minlength=K) + 1.0; prior /= prior.sum()

def post(P, s, rho):
    if s == 0: return P
    pm = np.clip(P.mean(0), 1e-6, None); pm /= pm.sum()
    t = (1 - rho) * pm + rho * prior; t /= t.sum()
    Q = P.copy()
    for _ in range(100):
        Q *= (len(Q) * t[None]) / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
        Q /= Q.sum(1, keepdims=True)
    return (1 - s) * np.log(P + 1e-12) + s * np.log(Q + 1e-12)

rng = np.random.default_rng(0)
groups = list(tr.groupby("prompt_id").indices.values())
draws = []
for d in range(30):
    idx = np.concatenate([rng.choice(g, min(2, len(g)), replace=False) for g in groups])
    draws.append(("bal", idx))
    w = rng.dirichlet(np.full(K, 2.0)); keep = rng.random(len(idx)) < (w / w.max())[y[idx]]
    draws.append(("imb", idx[keep]))

S = [0, 0.1, 0.2, 0.3, 0.42, 0.6]; R = [0, 0.2, 0.35, 0.5, 0.8, 1.0]
print("s\\rho " + " ".join(f"{r:>13}" for r in R))
for s in S:
    row = []
    for r in R:
        sc = {"bal": [], "imb": []}
        for k, idx in draws:
            sc[k].append(bas(y[idx], post(P0[idx], s, r).argmax(1)))
        row.append(f"{np.mean(sc['bal']):.4f}/{np.mean(sc['imb']):.4f}")
    print(f"{s:<5} " + " ".join(row), flush=True)
