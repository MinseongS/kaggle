"""Kind-2 simulation evaluation (outputs of kernel mins00/casmi26-sim2-gen in data/ext/sim2/).

Ranks the PubChem-channel rows of each molecule (truth in the tier) by hand rules z(fz) + lam*prior and reports
MRR@25 (InChIKey14 identity) on all / NP-library / np-examples / truth-in-NP-catalogue subsets with a bootstrap se
against the public 0.480 rule (lam 0.15, full pop: ls+lp+src+np+lotus+syn).

usage: uv run python -m casmi.sim2_eval [--dir data/ext/sim2]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def mrr_of(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """Per molecule reciprocal rank (<=25) of the best truth row, duplicates of a candidate key collapsed."""
    d = df.assign(_s=score).sort_values(['ik14', '_s'], ascending=[True, False], kind='stable')
    d = d.drop_duplicates(['ik14', 'cand_ik14'])
    d['_r'] = d.groupby('ik14').cumcount() + 1
    t = d[d.is_truth].groupby('ik14')._r.min()
    return (1.0 / t).where(t <= 25, 0.0)


def public_pop(df: pd.DataFrame, np_w=1.0, syn_w=0.5, lotus_w=0.5, src_w=0.5) -> np.ndarray:
    """Public 0.480 PubChem-channel prior (frozen v17 patch): ls + lp + 0.5*(src_mask&1) + 1.0*np + 0.5*ln(1+lotus) + 0.5*syn.
    lotus_log_prior assumed ln(1+refs) (definition lives in the dataset's aligned_public_priors.py)."""
    f = lambda c: df[c].astype(np.float32).values
    return (f('lsid') + f('lpmid') + src_w * ((df.src_mask.values & 1) != 0) + np_w * (df.np.values != 0)
            + lotus_w * np.log1p(f('lotus_refs')) + syn_w * f('lsynonyms')).astype(np.float32)


def rules(df: pd.DataFrame) -> dict[str, np.ndarray]:
    g = df.groupby('ik14').fz
    zf = ((df.fz - g.transform('mean')) / (g.transform('std', ddof=0) + 1e-9)).values
    pop = (df.lsid.astype(np.float32) + df.lpmid.astype(np.float32)).values
    pub = public_pop(df)
    out = {'fp only': zf, 'z+0.15pop(ls+lp)': zf + 0.15 * pop}
    for lam in (0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5):
        out[f'public lam={lam}'] = zf + lam * pub
    for npw in (0.0, 2.0):
        out[f'public np={npw}'] = zf + 0.15 * public_pop(df, np_w=npw)
    out['public syn=0'] = zf + 0.15 * public_pop(df, syn_w=0.0)
    out['public syn=1'] = zf + 0.15 * public_pop(df, syn_w=1.0)
    out['exp016 (lam0.2 syn1 np1.5)'] = zf + 0.2 * public_pop(df, np_w=1.5, syn_w=1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/ext/sim2')
    a = ap.parse_args()
    mols = pd.read_parquet(f'{a.dir}/sim2_mols.parquet')
    rows = pd.read_parquet(f'{a.dir}/sim2_rows.parquet')
    probed = mols[mols.n_window.notna()]
    print(f'molecules {len(mols)}, probed {len(probed)}, truth in tier {(probed.truth_rows > 0).sum()}, '
          f'median window {probed.n_window.median():.0f}, truth pass1 rank<=2000 '
          f'{((probed.truth_pass1_rank > 0) & (probed.truth_pass1_rank <= 2000)).sum()}')
    rows = rows[rows.ik14.isin(probed.loc[probed.truth_rows > 0, 'ik14'])]
    truth_np = rows[rows.is_truth].groupby('ik14').np.max() != 0
    src = mols.set_index('ik14').src
    res = {k: mrr_of(rows, s) for k, s in rules(rows).items()}
    base = res['public lam=0.15']
    subsets = {'all': base.index, 'lib': base.index[src.reindex(base.index).eq('lib').values],
               'npx': base.index[src.reindex(base.index).eq('npx').values],
               'truthNP': truth_np.index[truth_np.values], 'truth non-NP': truth_np.index[~truth_np.values]}
    # test (timsTOF, visible 400 mols) precursor m/z spans 245-460 -> neutral mass ~ 230-465
    mass = mols.set_index('ik14').mass.reindex(base.index)
    subsets['testmass'] = base.index[mass.between(230, 465).values]
    rng = np.random.default_rng(0)
    out = []
    for name, r in res.items():
        line = {'rule': name}
        for sn, ix in subsets.items():
            ix = ix[ix.isin(r.index)]
            d = (r[ix] - base[ix]).values
            bs = d[rng.integers(0, len(d), (1000, len(d)))].mean(1).std() if len(d) else np.nan
            line[sn] = f'{r[ix].mean():.3f} ({d.mean():+.3f}±{bs:.3f})'
        out.append(line)
    print('n:', {k: len(v) for k, v in subsets.items()})
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == '__main__':
    main()
