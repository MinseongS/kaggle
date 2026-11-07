"""Kind-2 simulation: LambdaRank vs hand rules on the PubChem-channel rows (CV track 1a).

Fold = identity fold of the molecule (mols.fold), train LightGBM LambdaRank on the other folds, score the held-out
fold. Compares against the public 0.480 rule (lam 0.15) and the exp016 rule (lam 0.20, syn 1.0, np 1.5), under both
z scales (z over the exported top rows, z over the whole 10ppm window from mols.fz_mu/fz_sd).

usage: uv run python -m casmi.sim2_rank [--dir data/ext/sim2]
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from casmi.sim2_eval import mrr_of, public_pop

PRIOR_COLS = ['lsid', 'lpmid', 'lsynonyms', 'lmesh', 'lpatent', 'n_depositors', 'n_vendors', 'np', 'src1',
              'llotus', 'formula_isomers']


def feats(df: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    X = pd.DataFrame({'z': z})
    for c in PRIOR_COLS:
        if c == 'src1':
            X[c] = ((df.src_mask.values & 1) != 0).astype(np.float32)
        elif c == 'llotus':
            X[c] = np.log1p(df.lotus_refs.values.astype(np.float32))
        else:
            X[c] = df[c].values.astype(np.float32)
    X['pub_pop'] = public_pop(df)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/ext/sim2')
    a = ap.parse_args()
    mols = pd.read_parquet(f'{a.dir}/sim2_mols.parquet')
    rows = pd.read_parquet(f'{a.dir}/sim2_rows.parquet')
    keep = mols.loc[mols.n_window.notna() & (mols.truth_rows > 0), 'ik14']
    rows = rows[rows.ik14.isin(keep)].sort_values(['ik14', 'row'], kind='stable').reset_index(drop=True)
    m = mols.set_index('ik14')
    fold = m.fold.reindex(rows.ik14).values
    g = rows.groupby('ik14').fz
    zs = {'top': ((rows.fz - g.transform('mean')) / (g.transform('std', ddof=0) + 1e-9)).values,
          'win': ((rows.fz.values - m.fz_mu.reindex(rows.ik14).values) / (m.fz_sd.reindex(rows.ik14).values + 1e-9))}
    truth_np = rows[rows.is_truth].groupby('ik14').np.max() != 0
    src = m.src
    rng = np.random.default_rng(0)
    for zn, z in zs.items():
        pub = public_pop(rows)
        res = {'public lam0.15': mrr_of(rows, z + 0.15 * pub),
               'exp016 rule': mrr_of(rows, z + 0.20 * public_pop(rows, np_w=1.5, syn_w=1.0))}
        X = feats(rows, z)
        y = rows.is_truth.values.astype(int)
        for name, cols, mono in [('lgb z+pub_pop', ['z', 'pub_pop'], [1, 1]),
                                 ('lgb all', list(X.columns), None),
                                 ('lgb no-commercial', ['z', 'pub_pop', 'lsynonyms', 'np', 'src1', 'llotus',
                                                        'formula_isomers', 'lsid', 'lpmid'], None)]:
            pred = np.zeros(len(rows))
            gains = []
            for k in sorted(np.unique(fold)):
                tr, te = fold != k, fold == k
                grp = rows.ik14[tr].groupby(rows.ik14[tr], sort=False).size().values
                params = dict(objective='lambdarank', n_estimators=300, learning_rate=0.05, num_leaves=15,
                              min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              lambdarank_truncation_level=25, verbose=-1, random_state=0)
                if mono:
                    params['monotone_constraints'] = mono
                mdl = lgb.LGBMRanker(**params)
                mdl.fit(X.loc[tr, cols], y[tr], group=grp)
                pred[te] = mdl.predict(X.loc[te, cols])
                gains.append(mdl.booster_.feature_importance('gain'))
            res[name] = mrr_of(rows, pred)
            gi = np.mean(gains, 0)
            print(f'[{zn}] {name} gain top:', sorted(zip(cols, (gi / gi.sum()).round(3)), key=lambda t: -t[1])[:6])
        base = res['public lam0.15']
        subsets = {'all': base.index, 'lib': base.index[src.reindex(base.index).eq('lib').values],
                   'npx': base.index[src.reindex(base.index).eq('npx').values],
                   'truthNP': truth_np.index[truth_np.values], 'truth non-NP': truth_np.index[~truth_np.values]}
        out = []
        for name, r in res.items():
            line = {'rule': name}
            for sn, ix in subsets.items():
                ix = ix[ix.isin(r.index)]
                d = (r[ix] - base[ix]).values
                bs = d[rng.integers(0, len(d), (1000, len(d)))].mean(1).std()
                line[sn] = f'{r[ix].mean():.3f} ({d.mean():+.3f}±{bs:.3f})'
            out.append(line)
        print(f'== z scale: {zn}  n={ {k: len(v) for k, v in subsets.items()} }')
        print(pd.DataFrame(out).to_string(index=False))


if __name__ == '__main__':
    main()
