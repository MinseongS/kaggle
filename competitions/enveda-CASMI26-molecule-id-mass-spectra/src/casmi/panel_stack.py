"""Panel PubChem-channel replay: hand rule vs learned LambdaRank (molecule-level CV).

Uses only the fold-safe FPNet (never saw the 250 panel molecules) so the comparison is honest.
Usage: uv run python -m casmi.panel_stack
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

D = Path(__file__).resolve().parents[2] / "data/ext/panel"
N_MOL = 250


def load() -> pl.DataFrame:
    c = pl.read_parquet(D / "pubchem_candidates.parquet").with_columns(pl.col("molecule").cast(pl.String))
    s = pl.read_parquet(D / "pubchem_signals.parquet").drop("molecule", "tier_row")
    f = pl.read_parquet(D / "panel_fpnet_scores.parquet").drop("molecule", "tier_row", "candidate_row")
    df = pl.concat([c, s, f], how="horizontal_extend")
    df = df.filter(pl.col("fold0_full").is_finite())
    # PubChem channel situation: answer is not in the pool, pool rows are excluded
    df = df.filter(~pl.col("in_pool") | pl.col("is_truth"))
    df = df.with_columns(
        ident=pl.when(pl.col("scorer_key") != "").then(pl.col("scorer_key")).otherwise(pl.col("connectivity"))
    )
    g = "molecule"
    z = lambda c: (pl.col(c) - pl.col(c).mean().over(g)) / (pl.col(c).std().over(g) + 1e-6)
    rk = lambda c: pl.col(c).rank("ordinal", descending=True).over(g).cast(pl.Float32)
    df = df.with_columns(
        z_f0=z("fold0_full"), z_f0p=z("fold0_partial"),
        r_f0=rk("fold0_full"), r_f0p=rk("fold0_partial"),
        gap_f0=pl.col("fold0_full").max().over(g) - pl.col("fold0_full"),
        n_win=pl.len().over(g).cast(pl.Float32),
        n_sf=pl.col("same_formula").sum().over(g).cast(pl.Float32),
        np_bits=pl.col("np_flags").cast(pl.Int32),
        lotus=pl.col("lotus_refs").cast(pl.Float32).log1p(),
        z_prior=z("prior_v10"), r_prior=rk("prior_v10"),
        z_pop=z("pop"),
    )
    # same-formula-relative features (most errors are same-formula isomers)
    gf = ["molecule", "same_formula"]
    df = df.with_columns(
        zsf_f0=(pl.col("fold0_full") - pl.col("fold0_full").mean().over(gf)) / (pl.col("fold0_full").std().over(gf) + 1e-6),
        rsf_f0=pl.col("fold0_full").rank("ordinal", descending=True).over(gf).cast(pl.Float32),
        rsf_prior=pl.col("prior_v10").rank("ordinal", descending=True).over(gf).cast(pl.Float32),
    )
    return df.sort("molecule")


FEATS = ["z_f0", "z_f0p", "r_f0", "r_f0p", "gap_f0", "n_win", "n_sf", "same_formula",
         "log_substances", "log_pubmed", "log_patents", "np_bits", "lotus", "pubchemlite_member",
         "spectral_source", "np_member", "prior_v10", "z_prior", "r_prior", "z_pop",
         "zsf_f0", "rsf_f0", "rsf_prior"]


def mrr_from(df: pl.DataFrame, score: np.ndarray) -> dict:
    """Per-molecule reciprocal rank (one guess per identity); molecules absent -> 0."""
    d = df.select("molecule", "ident", "is_truth").with_columns(s=pl.Series(score))
    d = d.sort(["molecule", "s"], descending=[False, True]).unique(["molecule", "ident"], keep="first", maintain_order=True)
    d = d.with_columns(r=pl.int_range(1, pl.len() + 1).over("molecule"))
    hit = d.filter(pl.col("is_truth")).group_by("molecule").agg(pl.col("r").min())
    rr = {m: (1.0 / r if r <= 25 else 0.0) for m, r in hit.iter_rows()}
    return rr


def summarize(name: str, rr: dict, mols: list[str]) -> np.ndarray:
    v = np.array([rr.get(m, 0.0) for m in mols])
    v = np.concatenate([v, np.zeros(N_MOL - len(mols))])
    print(f"{name:<40s} MRR@25 {v.mean():.4f}  top1 {np.mean(v == 1):.3f}", flush=True)
    return v


def main() -> None:
    df = load()
    mols = df["molecule"].unique().sort().to_list()
    print("rows", df.height, "molecules", len(mols))
    res = {}
    for w in (0.0, 0.15, 0.25, 0.5):
        sc = (df["z_f0"] + w * df["prior_v10"]).to_numpy()
        res[f"rule z_f0 + {w}*prior_v10"] = summarize(f"rule z_f0 + {w}*prior_v10", mrr_from(df, sc), mols)

    rng = np.random.default_rng(0)
    fold_of = dict(zip(mols, rng.permutation(len(mols)) % 5))
    fold = df["molecule"].replace_strict(fold_of, return_dtype=pl.Int8).to_numpy()
    X = df.select(FEATS).cast(pl.Float32).to_numpy()
    y = df["is_truth"].cast(pl.Int8).to_numpy()
    grp_id = df["molecule"].to_numpy()
    for seed_avg in (3,):
        pred = np.zeros(df.height)
        for k in range(5):
            te = fold == k
            # LightGBM caps a query at 10k rows: train on the top 3000 by FP or prior (+ the answer)
            tr = (fold != k) & ((X[:, FEATS.index("r_f0")] <= 1500) | (X[:, FEATS.index("r_prior")] <= 1500) | (y == 1))
            _, gsz = np.unique(grp_id[tr], return_counts=True)
            for s in range(seed_avg):
                m = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   lambdarank_truncation_level=30, random_state=s, verbose=-1)
                m.fit(X[tr], y[tr], group=gsz)
                pred[te] += m.predict(X[te]) / seed_avg
        res["lambdarank"] = summarize("lambdarank (5-fold mol CV)", mrr_from(df, pred), mols)
    base = res["rule z_f0 + 0.15*prior_v10"]
    for k, v in res.items():
        d = v - base
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"  delta vs rule0.15: {k:<35s} {d.mean():+.4f}  se {np.std(bs):.4f}  up {int((d>0).sum())} down {int((d<0).sum())}")
    tr = (X[:, FEATS.index("r_f0")] <= 1500) | (X[:, FEATS.index("r_prior")] <= 1500) | (y == 1)
    m.fit(X[tr], y[tr], group=np.unique(grp_id[tr], return_counts=True)[1])
    imp = sorted(zip(m.booster_.feature_importance("gain"), FEATS), reverse=True)
    print("gain:", [(f, round(g)) for g, f in imp[:12]])


if __name__ == "__main__":
    sys.exit(main())
