"""Ranker v2: leak-free training rows from ALL 1000 holdout_v1 molecules + extra features + HGB/LambdaRank blend.

  uv run python -m casmi.rows_v2 build --fp-dir data/ext/casmi26-fp-models-h1      # -> outputs/ranker_rows_v2/
  DYLD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/torch/lib uv run python -m casmi.rows_v2 eval
  uv run casmi predict --ranker v2 --ranker-rows outputs/ranker_rows_v2 --fp-dir data/ext/casmi26-fp-models-h1

Rows: every holdout_v1 molecule (cv 500 + extra 500; all excluded from the h1 FPNet's spectra and decoys) is
queried as in cv.py -- split A (class-1 sim, M=0; only A-eligible molecules) and split B (class-2 sim, M=1).
Features = the 31 base columns + prior (desc, desc_rel, formula; prior.py) + FP-family block (per-family f.z
z/rank/gap, confident-bit hinge counts, cosine; the cvx.fp_block columns). COCONUT membership is NEVER a feature.

Files in the rows dir:
  rows.npz   X (float32, n x d), Y, M (0=A, 1=B), G (group id = molecule*2 + split), mol (holdout index)
  meta.json  feature names, groups, fp weights used, counts
  keys.pkl   per-row metric keys + molecule info (evaluation only; not needed at predict time)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from . import config
from .features import FEATURE_NAMES, _rank_norm, _z

PRIOR_GROUPS = ('desc', 'desc_rel', 'formula')
FPB_NAMES = [f'{fam}_{k}' for fam in ('ecfp4', 'ecfp6', 'rdk', 'maccs') for k in ('z', 'rank', 'gap')] + \
            ['fpb_miss', 'fpb_extra', 'fpb_hinge_rank', 'fpb_cos_z', 'fpb_cos_rank', 'fpb_log_nvar']
DEFAULT_DIR = config.OUTPUTS / 'ranker_rows_v2'


# --------------------------------------------------------------------------------------------- extra features
def fp_family_block(F, z, blk):
    """= cvx.fp_block for one window. F: (nc, nbits) 0/1 candidate fps, z: FPNet logits (or None)."""
    nc = F.shape[0]
    if z is None:
        return np.zeros((nc, len(FPB_NAMES)), np.float32)
    F = F.astype(np.float32)
    ex = []
    for b in range(4):
        m = blk == b
        x = F[:, m] @ z[m]
        ex += [_z(x), _rank_norm(x), x - x.max()]
    mean = F.mean(0)
    var = (mean > 0) & (mean < 1)
    pr = 1 / (1 + np.exp(-z[var])); Fv = F[:, var]
    nv = max(int(var.sum()), 1)
    miss = ((1 - Fv) * (pr > 0.9)).sum(1); extra = (Fv * (pr < 0.1)).sum(1)
    cos = (Fv @ pr) / (np.linalg.norm(Fv, axis=1) * np.linalg.norm(pr) + 1e-9)
    ex += [miss, extra, _rank_norm(-(miss + extra)), _z(cos), _rank_norm(cos), np.full(nc, np.log(nv))]
    return np.column_stack(ex).astype(np.float32)


def family_of_bits():
    bits = np.load(config.COCO_DIR / 'fp_bits.npy')
    return np.searchsorted([4096, 8192, 10240], bits, 'right')  # 0 ECFP4, 1 ECFP6, 2 RDKit path, 3 MACCS


def extra_features(pool, pidx, target, z, D, blk):
    """Prior + FP-family columns for one candidate window (pool indices pidx). D = prior.describe_many dict."""
    from .prior import prior_features
    smis = [pool.smiles[i] for i in pidx]
    pf, names = prior_features(smis, pool.mass[pidx], target, np.zeros(len(pidx)), D, PRIOR_GROUPS)
    fb = fp_family_block(pool.fps(pidx), z, blk)
    return np.hstack([pf, fb]).astype(np.float32), names + FPB_NAMES


# --------------------------------------------------------------------------------------------- build rows
def build_rows(E, out_dir=DEFAULT_DIR, log_every=25, limit=None):
    from .cv import fixed_holdout, load_query_rows, make_query
    from .engine import Mask
    from .metric import score_key
    from .prior import describe_many
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    L, P = E.L, E.pool
    hold = fixed_holdout('all')[:limit]
    raw = load_query_rows([i for h in hold for i in h['q_rows']])
    t0 = time.time()
    part = out_dir / 'windows.pkl'   # resumable: per-molecule windows (base X, pidx, z)
    W = pickle.load(open(part, 'rb')) if part.exists() else []
    for j in range(len(W), len(hold)):
        h = hold[j]
        q = make_query(L, h, raw)
        w = dict(target=None, z=None, A=None, B=None)
        if q is not None:
            G = np.array(h['G'], np.int64)
            z = E.logits(q)
            mB = Mask(excl_sids=G)
            cB = E.candidates(q, E.lib_sim(q, mB), E.analog_sim(q, mB), z)
            w.update(target=q.target, z=z)
            if cB is not None:
                w['B'] = dict(X=cB['X'], pidx=cB['pidx'])
            if h['eligible_A']:
                mA = Mask(src_sids=G, src_lib=h['src_lib'])
                ov = {int(s): L.rep_excluding(int(s), lambda idx: mA.spectra_bad(L, idx)) for s in G}
                cA = E.candidates(q, E.lib_sim(q, mA), E.analog_sim(q, mA, ov), z)
                if cA is not None:
                    w['A'] = dict(X=cA['X'], pidx=cA['pidx'])
        W.append(w)
        if (j + 1) % log_every == 0 or j + 1 == len(hold):
            el = time.time() - t0
            print(f'  rows {j+1}/{len(hold)}  {el:.0f}s ({el/(j+1-0):.2f}s/mol)', flush=True)
            pickle.dump(W, open(part, 'wb'))
    t_win = time.time() - t0
    # metric keys + descriptors for every candidate
    allp = np.unique(np.concatenate([w[s]['pidx'] for w in W for s in 'AB' if w[s] is not None]))
    from .metric import score_keys_parallel
    keys = dict(zip(allp.tolist(), score_keys_parallel([P.smiles[i] for i in allp], workers=E.cfg.WORKERS)))
    t1 = time.time()
    D = describe_many([P.smiles[i] for i in allp], cache=out_dir / 'desc_cache.pkl')
    t_desc = time.time() - t1
    blk = family_of_bits()
    X, Y, M, Gr, MOL, KEYS = [], [], [], [], [], []
    names = None
    for j, (h, w) in enumerate(zip(hold, W)):
        for si, s in enumerate('AB'):
            c = w[s]
            if c is None:
                continue
            ex, en = extra_features(P, c['pidx'], w['target'], w['z'], D, blk)
            names = names or FEATURE_NAMES + en
            k = [keys[int(i)] for i in c['pidx']]
            X.append(np.hstack([c['X'], ex])); Y.append(np.array([x == h['mkey'] for x in k], np.float32))
            M.append(np.full(len(k), si, np.int8)); Gr.append(np.full(len(k), 2 * j + si, np.int32))
            MOL.append(np.full(len(k), j, np.int32)); KEYS += k
    X = np.vstack(X); Y = np.concatenate(Y); M = np.concatenate(M); Gr = np.concatenate(Gr); MOL = np.concatenate(MOL)
    np.savez_compressed(out_dir / 'rows.npz', X=X, Y=Y, M=M, G=Gr, mol=MOL)
    pickle.dump(dict(keys=KEYS, mols=[dict(src_lib=h['src_lib'], eligible_A=h['eligible_A'], mkey=h['mkey'])
                                      for h in hold]), open(out_dir / 'keys.pkl', 'wb'))
    meta = dict(names=names, n_base=len(FEATURE_NAMES), prior_groups=list(PRIOR_GROUPS), fpblock=True,
                fp_dir=str(config.FP_MODEL_DIR), n_mol=len(hold), n_rows=int(len(Y)), n_A_groups=int(sum(w['A'] is not None for w in W)),
                n_B_groups=int(sum(w['B'] is not None for w in W)), secs_windows=round(t_win), secs_desc=round(t_desc))
    json.dump(meta, open(out_dir / 'meta.json', 'w'), indent=1)
    print(f'rows: {X.shape} -> {out_dir}; {meta}', flush=True)
    return meta


def load_rows(rows_dir=DEFAULT_DIR):
    rows_dir = Path(rows_dir)
    z = np.load(rows_dir / 'rows.npz')
    return {k: z[k] for k in z.files}, json.load(open(rows_dir / 'meta.json'))


# --------------------------------------------------------------------------------------------- ranker
class RankerV2:
    """HGB (Ranker) and/or LightGBM LambdaRank on v2 rows; kind in {'hgb', 'lgb', 'blend'}.
    `cols` = feature columns used ('base' = the 31, 'all' = base + prior + FP-family)."""

    def __init__(self, cfg, kind='blend', cols='all'):
        self.cfg, self.kind, self.cols = cfg, kind, cols
        self.extra = cols == 'all'

    def fit(self, X, Y, M, G):
        from .cvx import LGBRank
        from .ranker import Ranker
        t0 = time.time()
        self.hgb = Ranker(self.cfg).fit(X, Y, M) if self.kind in ('hgb', 'blend') else None
        self.lgb = LGBRank(self.cfg).fit(X, Y, M, G) if self.kind in ('lgb', 'blend') else None
        self.fit_secs = time.time() - t0
        return self

    def predict(self, X):
        if self.kind == 'hgb':
            return self.hgb.predict(X)
        if self.kind == 'lgb':
            return self.lgb.predict(X)
        return 0.5 * (1 - _rank_norm(self.hgb.predict(X))) + 0.5 * (1 - _rank_norm(self.lgb.predict(X)))

    @classmethod
    def from_rows(cls, cfg, rows_dir=DEFAULT_DIR, kind='blend', cols='all'):
        R, meta = load_rows(rows_dir)
        X = R['X'] if cols == 'all' else R['X'][:, :meta['n_base']]
        r = cls(cfg, kind, cols).fit(X, R['Y'], R['M'], R['G'])
        r.names = meta['names'] if cols == 'all' else meta['names'][:meta['n_base']]
        print(f'ranker v2 ({kind}, {cols}: {X.shape[1]} cols) on {len(R["Y"]):,} rows / {meta["n_mol"]} molecules '
              f'({r.fit_secs:.0f}s)', flush=True)
        return r

    # ---- test-time featurization (two-pass: descriptors for all test candidates are computed in one batch)
    def prepare(self, E, windows, cache=None):
        """windows: list of (q, c, z). Computes descriptors for every candidate SMILES once."""
        if not self.extra:
            return
        from .prior import describe_many
        t0 = time.time()
        smis = [E.pool.smiles[i] for _, c, _ in windows if c is not None for i in c['pidx']]
        self.D = describe_many(smis, cache=cache)
        self.blk = family_of_bits()
        self.prep_secs = time.time() - t0
        print(f'  v2 descriptors: {len(set(smis)):,} candidate SMILES ({self.prep_secs:.0f}s)', flush=True)

    def features(self, E, q, c, z):
        if not self.extra:
            return c['X']
        ex, _ = extra_features(E.pool, c['pidx'], q.target, z, self.D, self.blk)
        X = np.hstack([c['X'], ex])
        assert X.shape[1] == len(self.names), (X.shape, len(self.names))
        return X


# --------------------------------------------------------------------------------------------- evaluation
def _rr(score, keys, Y, topn=25):
    order = np.argsort(-score, kind='stable')
    seen, r = set(), 0
    for i in order:
        k = keys[i]
        if k is None or k in seen:
            continue
        seen.add(k); r += 1
        if Y[i] > 0:
            return 1.0 / r
        if r >= topn:
            break
    return 0.0


def evaluate(rows_dir=DEFAULT_DIR, configs=('rank_train', 'hgb31', 'lgb31', 'blend31', 'hgb', 'lgb', 'blend'),
             n_folds=5, seed=0, out_name='cv_rr.npz'):
    from .cvx import LGBRank
    from .ranker import Ranker
    cfg = config.CFG()
    rows_dir = Path(rows_dir)
    R, meta = load_rows(rows_dir)
    K = pickle.load(open(rows_dir / 'keys.pkl', 'rb'))
    keys = np.asarray(K['keys'], dtype=object)
    nm = meta['n_mol']; nb = meta['n_base']
    X, Y, M, G, mol = R['X'], R['Y'], R['M'], R['G'], R['mol']
    fold = np.random.default_rng(seed).permutation(nm) % n_folds
    grp_idx = {g: np.where(G == g)[0] for g in np.unique(G)}
    res = {}
    prev = dict(np.load(rows_dir / out_name)) if (rows_dir / out_name).exists() else {}
    if 'fold' in prev and not np.array_equal(prev['fold'], fold):
        prev = {}
    todo = [c for c in configs if c not in prev]   # resumable: finished configs are kept

    def save():
        np.savez(rows_dir / out_name, **{**prev, **res, 'fold': fold})

    if 'rank_train' in todo:
        t0 = time.time()
        rk = Ranker.from_rank_train(cfg)
        p = rk.predict(X[:, :nb])
        rr = np.full((2, nm), np.nan)
        for g, ix in grp_idx.items():
            rr[g % 2][g // 2] = _rr(p[ix], keys[ix], Y[ix])
        res['rank_train'] = rr; save()
        print(f'  [rank_train] {time.time()-t0:.0f}s', flush=True)
    for cols in ('31', ''):
        fam = [c for c in ('hgb', 'lgb', 'blend') if c + cols in todo]
        if not fam:
            continue
        Xc = X[:, :nb] if cols == '31' else X
        rr = {c: np.full((2, nm), np.nan) for c in fam}
        t0 = time.time()
        need_h = 'hgb' in fam or 'blend' in fam; need_l = 'lgb' in fam or 'blend' in fam
        for f in range(n_folds):
            tr = fold[mol] != f
            h = Ranker(cfg).fit(Xc[tr], Y[tr], M[tr]) if need_h else None
            l = LGBRank(cfg).fit(Xc[tr], Y[tr], M[tr], G[tr]) if need_l else None
            ph = h.predict(Xc) if h else None      # all rows at once (per-call overhead dominates otherwise)
            pl_ = l.predict(Xc) if l else None
            for g, ix in grp_idx.items():
                if fold[g // 2] != f:
                    continue
                for c in fam:
                    if c == 'blend':
                        p = 0.5 * (1 - _rank_norm(ph[ix])) + 0.5 * (1 - _rank_norm(pl_[ix]))
                    else:
                        p = (ph if c == 'hgb' else pl_)[ix]
                    rr[c][g % 2][g // 2] = _rr(p, keys[ix], Y[ix])
            print(f'  [{"/".join(fam)}{cols}] fold {f} done ({time.time()-t0:.0f}s)', flush=True)
        for c in fam:
            res[c + cols] = rr[c]
        save()
    save()
    return report(rows_dir, out_name)


def _stat(rr, mask, elig):
    """weighted = 0.16*mean(A over A-eligible) + 0.45*mean(B); split C (= 0 by construction) contributes 0."""
    A = rr[0][mask & elig]; B = rr[1][mask]
    return 0.16 * np.nanmean(A) + 0.45 * np.nanmean(B), np.nanmean(A), np.nanmean(B)


def report(rows_dir=DEFAULT_DIR, out_name='cv_rr.npz', base='rank_train', n_boot=2000):
    rows_dir = Path(rows_dir)
    Z = dict(np.load(rows_dir / out_name))
    K = pickle.load(open(rows_dir / 'keys.pkl', 'rb'))
    libs = np.array([m['src_lib'] for m in K['mols']]); elig = np.array([m['eligible_A'] for m in K['mols']])
    # A for groups with no candidates / missing window -> 0 (as in cv.py)
    for k in Z:
        if k != 'fold':
            Z[k] = Z[k].copy(); Z[k][0][elig & np.isnan(Z[k][0])] = 0.0; Z[k][1][np.isnan(Z[k][1])] = 0.0
    npm = libs == 'enveda-np-examples'
    rng = np.random.default_rng(1)
    boots = {name: [rng.integers(0, m.sum(), m.sum()) for _ in range(n_boot)] for name, m in (('np', npm), ('all', np.ones_like(npm)))}
    lines = []
    for k, rr in Z.items():
        if k == 'fold':
            continue
        parts = []
        for name, m in (('np', npm), ('all', np.ones_like(npm))):
            w, a, b = _stat(rr, m, elig)
            s = f'{name}: W={w:.4f} A={a:.4f} B={b:.4f}'
            if base in Z and k != base:
                idx = np.where(m)[0]
                d = []
                for bi in boots[name]:
                    ii = idx[bi]; mm = np.zeros_like(m); np.add.at(mm, ii, 1)
                    # weighted statistic with multiplicity weights
                    wA = mm * elig; wB = mm
                    fa = lambda r: np.nansum(r[0] * wA) / wA.sum(); fb = lambda r: np.sum(r[1] * wB) / wB.sum()
                    d.append(0.16 * (fa(rr) - fa(Z[base])) + 0.45 * (fb(rr) - fb(Z[base])))
                d = np.array(d)
                s += f' d={w - _stat(Z[base], m, elig)[0]:+.4f} (se {d.std():.4f}, P>0 {np.mean(d > 0):.2f})'
            parts.append(s)
        lines.append(f'{k:>11} | ' + ' | '.join(parts))
    lines.append(f'n: np={npm.sum()} all={len(npm)} A-eligible np={int((elig & npm).sum())} all={int(elig.sum())}')
    txt = '\n'.join(lines)
    print(txt, flush=True)
    (rows_dir / 'cv_summary.txt').write_text(txt + '\n')
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser('casmi.rows_v2')
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('--out', default=str(DEFAULT_DIR))
    b.add_argument('--fp-dir', default=str(config.EXT / 'casmi26-fp-models-h1')); b.add_argument('--workers', type=int, default=None)
    b.add_argument('--limit', type=int, default=None)
    e = sub.add_parser('eval'); e.add_argument('--rows', default=str(DEFAULT_DIR))
    e.add_argument('--configs', default='rank_train,hgb31,lgb31,blend31,hgb,lgb,blend'); e.add_argument('--fold-seed', type=int, default=0)
    r = sub.add_parser('report'); r.add_argument('--rows', default=str(DEFAULT_DIR))
    a = ap.parse_args(argv)
    if a.cmd == 'build':
        from .cli import _engine
        config.FP_MODEL_DIR = Path(a.fp_dir)
        cfg = config.CFG()
        if a.workers: cfg.WORKERS = a.workers
        E = _engine(cfg)
        try:
            build_rows(E, a.out, limit=a.limit)
        finally:
            if E.frag is not None: E.frag.close()
    elif a.cmd == 'eval':
        evaluate(a.rows, tuple(a.configs.split(',')), seed=a.fold_seed,
                 out_name='cv_rr.npz' if a.fold_seed == 0 else f'cv_rr_s{a.fold_seed}.npz')
    else:
        report(a.rows)


if __name__ == '__main__':
    main()
