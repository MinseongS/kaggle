"""CV harness for name-that-llm.

- Official train split by prompt_id (GroupKFold); external pool rows are training-only.
- Scores: full balanced accuracy, family accuracy, and the test-shaped m=2 cut
  (each prompt keeps 2 random answers; any post-processing runs on the cut set).
- lb_gap_sd: expected public-LB sd of the score difference between two OOF predictions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "kimi-k3", "qwen3.8-max",
]
L2I = {m: i for i, m in enumerate(LABELS)}
K = len(LABELS)
N_PUBLIC = 1200  # ~30% of ~4000 hidden rows (estimate)


def family(m: str) -> str:
    return m.split("-")[0].rstrip("0123456789.")


FAM = np.array([family(m) for m in LABELS])


def load_train() -> pd.DataFrame:
    tr = pd.read_csv(DATA / "train.csv")
    tr["y"] = tr.model.map(L2I)
    return tr


def load_pool(per_label: int | None = 2500, seed: int = 0, weighted: bool = True) -> pd.DataFrame:
    """Open Pool, minus anything near train prompts. Sample per label by match_weight."""
    p = pd.read_parquet(DATA / "open_pool" / "ntl_open_pool.parquet")
    p = p[~(p.flag_near_train_prompt | p.flag_train_prompt_regen | p.flag_gemini37_routing_risk)]
    if per_label is not None:
        # weighted sampling without replacement (Efraimidis-Spirakis): top-n of u^(1/w)
        rng = np.random.default_rng(seed)
        w = p.match_weight.clip(lower=1e-6).values if weighted else np.ones(len(p))
        p = p.assign(_key=np.log(rng.random(len(p))) / w)
        p = p.sort_values("_key", ascending=False).groupby("model").head(per_label).drop(columns="_key")
    p = p.reset_index(drop=True)
    p["y"] = p.model.map(L2I)
    return p


def folds(tr: pd.DataFrame, n: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(GroupKFold(n).split(tr, tr.y, tr.prompt_id))


# ---------- post-processing ----------
def sinkhorn(prob: np.ndarray, strength: float = 1.0, iters: int = 50) -> np.ndarray:
    """Rescale columns so predicted class mass is uniform; blend in log space by `strength`."""
    q = prob.copy()
    for _ in range(iters):
        q = q / q.sum(0, keepdims=True) * (len(q) / K)
        q = q / q.sum(1, keepdims=True)
    lp = (1 - strength) * np.log(prob + 1e-12) + strength * np.log(q + 1e-12)
    e = np.exp(lp - lp.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ---------- scoring ----------
def bacc(y: np.ndarray, prob: np.ndarray) -> float:
    return balanced_accuracy_score(y, prob.argmax(1))


def m_cut_score(tr: pd.DataFrame, prob: np.ndarray, m: int = 2, draws: int = 20,
                post=None, seed: int = 0) -> float:
    """Keep m random answers per prompt, apply `post` on the cut set, score. Mean over draws."""
    rng = np.random.default_rng(seed)
    groups = list(tr.groupby("prompt_id").indices.values())
    out = []
    for _ in range(draws):
        idx = np.concatenate([rng.choice(g, min(m, len(g)), replace=False) for g in groups])
        p = prob[idx] if post is None else post(prob[idx])
        out.append(balanced_accuracy_score(tr.y.values[idx], p.argmax(1)))
    return float(np.mean(out))


def report(name: str, tr: pd.DataFrame, prob: np.ndarray, post=None) -> dict:
    y = tr.y.values
    p = prob if post is None else post(prob)
    pred = p.argmax(1)
    r = {
        "name": name,
        "bacc": round(bacc(y, p), 4),
        "family_acc": round(float((FAM[pred] == FAM[y]).mean()), 4),
        "m2": round(m_cut_score(tr, prob, 2, post=post), 4),
    }
    w = tr.text.str.split().str.len()
    for lo, hi in [(0, 60), (61, 120), (121, 180), (181, 10**6)]:
        s = (w >= lo) & (w <= hi)
        r[f"acc_{lo}-{hi if hi < 10**6 else ''}"] = round(float((pred[s] == y[s]).mean()), 3)
    print(r)
    return r


def lb_gap_sd(prob_a: np.ndarray, prob_b: np.ndarray, y: np.ndarray, n_public: int = N_PUBLIC) -> float:
    """sd of (score_a - score_b) on a public set of n_public rows: sqrt(f / n), f = share where exactly one is right."""
    ra, rb = prob_a.argmax(1) == y, prob_b.argmax(1) == y
    f = float((ra ^ rb).mean())
    return float(np.sqrt(f / n_public))


def per_class_recall(y: np.ndarray, prob: np.ndarray) -> pd.Series:
    pred = prob.argmax(1)
    return pd.Series({LABELS[k]: float((pred[y == k] == k).mean()) for k in range(K)}).round(3)
