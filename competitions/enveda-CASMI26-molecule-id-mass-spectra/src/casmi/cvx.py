"""Ranker-side experiments on CACHED CV features (outputs/cv/<tag>/feats.pkl) -- no channel recomputation.

  uv run python -m casmi.cvx cands --tag exp000             # once: recover candidate pool indices per window
  uv run python -m casmi.cvx run --exp exp002 --groups desc,desc_rel,formula [--ranker hgb|lgb_rank]

Same folds as cv.evaluate (rng(0).permutation % 5), same Ranker config, only 'full' variant. Reports per-split MRR,
np-weighted total (0.16*A + 0.45*B, C=0), fold std, and a paired bootstrap vs a baseline run (per-molecule rr is
saved to outputs/cv/<tag>/cvx_<exp>.npz).

LightGBM needs libomp: run with DYLD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/torch/lib on macOS.
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np

from . import config
from .cv import NP_LIB, WEIGHTS, rr_from_scores

OUT = config.OUTPUTS / 'cv'


# --------------------------------------------------------------------------------------------- candidate recovery
def recover_candidates(tag='exp000'):
    from .candidates import Pool
    from .metric import score_keys_parallel
    cfg = config.CFG()
    recs = pickle.load(open(OUT / tag / 'feats.pkl', 'rb'))
    P = Pool(cfg)
    wins = []
    for r in recs:
        if r['B'] is None:
            wins.append(None); continue
        c = P.window(r['target'], cfg.PPM_WIN)
        if len(c) == 0:
            c = P.window(r['target'], cfg.PPM_FALLBACK)
        wins.append(c)
    allc = np.unique(np.concatenate([w for w in wins if w is not None]))
    keys = score_keys_parallel([P.smiles[i] for i in allc], workers=cfg.WORKERS)
    k_of = dict(zip(allc.tolist(), keys))
    out, bad = [], 0
    for r, w in zip(recs, wins):
        d = {}
        for s in 'ABC':
            p = r[s]
            if p is None:
                d[s] = None; continue
            if s == 'C':
                w_s = np.array([i for i in w if k_of[int(i)] != r['h']['mkey']], np.int64)
            else:
                w_s = w
            if len(w_s) == p['n'] and all(k_of[int(i)] == k for i, k in zip(w_s, p['keys'])):
                d[s] = w_s; continue
            # cap hit / order differs: match by key (first unused pool index with that key)
            bad += 1
            byk = {}
            for i in w_s:
                byk.setdefault(k_of[int(i)], []).append(int(i))
            d[s] = np.array([byk[k].pop(0) if byk.get(k) else -1 for k in p['keys']], np.int64)
        out.append(d)
    meta = dict(pidx=out, smiles=P.smiles, mass=P.mass, src=P.src)
    idx = np.unique(np.concatenate([v for d in out for v in d.values() if v is not None]))
    idx = idx[idx >= 0]
    small = dict(pidx=out, smiles={int(i): P.smiles[i] for i in idx}, mass={int(i): float(P.mass[i]) for i in idx},
                 src={int(i): int(P.src[i]) for i in idx})
    pickle.dump(small, open(OUT / tag / 'cands.pkl', 'wb'))
    print(f'recovered {len(out)} molecules, {bad} windows matched by key, {len(idx)} unique candidates')
    del meta
    return small


# --------------------------------------------------------------------------------------------- feature augmentation
def augment(recs, cands, groups):
    from .prior import describe_many, prior_features
    smi = [cands['smiles'][int(i)] for d in cands['pidx'] for v in d.values() if v is not None for i in v if i >= 0]
    D = describe_many(smi)
    new, names = [], None
    for r, d in zip(recs, cands['pidx']):
        rr = dict(r)
        for s in 'ABC':
            p = r[s]
            if p is None:
                continue
            pid = d[s]
            ok = pid >= 0
            sm = [cands['smiles'][int(i)] if i >= 0 else 'C' for i in pid]
            ms = np.array([cands['mass'][int(i)] if i >= 0 else r['target'] for i in pid])
            co = np.array([cands['src'][int(i)] == 0 if i >= 0 else 0 for i in pid], np.float32)
            Dx = dict(D); Dx.setdefault('C', (np.full(len(D[next(iter(D))][0]), np.nan, np.float32), None))
            ex, names = prior_features(sm, ms, r['target'], co, Dx, groups)
            ex[~ok] = 0.0
            rr[s] = dict(p, X=np.hstack([p['X'], ex]))
        new.append(rr)
    return new, names


def group_rel(recs, cols=(15, 5, 21, 0)):
    """Within-window relative features for key channels: gap to 2nd best, gap to best among OTHER candidates,
    softmax share. (window ~= same-formula group: 10 ppm)."""
    new = []
    for r in recs:
        rr = dict(r)
        for s in 'ABC':
            p = r[s]
            if p is None:
                continue
            X = p['X']; ex = []
            for c in cols:
                x = X[:, c].astype(np.float64)
                o = np.sort(x)[::-1]
                best, second = o[0], (o[1] if len(o) > 1 else o[0])
                other_best = np.where(x >= best, second, best)
                e = np.exp((x - best) * 3.0); sm = e / e.sum()
                ex += [x - other_best, np.full(len(x), best - second), sm]
            rr[s] = dict(p, X=np.hstack([X, np.column_stack(ex).astype(np.float32)]))
        new.append(rr)
    return new



def fp_block(recs, cands, tag='exp000'):
    """FPNet agreement split by fingerprint family (ECFP4 / ECFP6 / RDKit path / MACCS), plus confident-bit hinge
    counts and cosine(f, sigmoid z) over bits that VARY inside the window. Needs zlog.npy (cvx zlog)."""
    from .candidates import Pool
    from .features import _rank_norm, _z
    Z = np.load(OUT / tag / 'zlog.npy')
    bits = np.load(config.COCO_DIR / 'fp_bits.npy')
    blk = np.searchsorted([4096, 8192, 10240], bits, 'right')  # 0..3
    P = Pool(config.CFG())
    new = []
    chk = []
    for j, (r, d) in enumerate(zip(recs, cands['pidx'])):
        rr = dict(r)
        z = Z[j]
        for s in 'ABC':
            p = r[s]
            if p is None:
                continue
            pid = d[s]
            F = P.fps(np.maximum(pid, 0)).astype(np.float32)
            raw = F @ z
            if s == 'B' and len(raw) > 2:
                chk.append(np.corrcoef(_rank_norm(raw), p['X'][:, 16])[0, 1])
            ex = []
            for b in range(4):
                m = blk == b
                x = F[:, m] @ z[m]
                ex += [_z(x), _rank_norm(x), x - x.max()]
            var = (F.mean(0) > 0) & (F.mean(0) < 1)
            pr = 1 / (1 + np.exp(-z[var])); Fv = F[:, var]
            nv = max(int(var.sum()), 1)
            miss = ((1 - Fv) * (pr > 0.9)).sum(1); extra = (Fv * (pr < 0.1)).sum(1)
            cos = (Fv @ pr) / (np.linalg.norm(Fv, axis=1) * np.linalg.norm(pr) + 1e-9)
            ex += [miss, extra, _rank_norm(-(miss + extra)), _z(cos), _rank_norm(cos), np.full(len(raw), np.log(nv))]
            rr[s] = dict(p, X=np.hstack([p['X'], np.column_stack(ex).astype(np.float32)]))
        new.append(rr)
    print(f'fp_block: rank corr with cached fz_rank (split B) median={np.median(chk):.4f} min={np.min(chk):.4f}')
    return new



def frag2_feats(recs, cands, tag='exp000', workers=4):
    """MetFrag-lite v2 features (frag2.py): max over the molecule's spectra + window z/rank of key columns."""
    import multiprocessing as mp
    from .features import _rank_norm, _z
    from .frag2 import fragment_masses2, spec_scores
    from .spectra import _clean
    cfg = config.CFG()
    cp = OUT / tag / 'frag2_masses.pkl'
    FM = pickle.load(open(cp, 'rb')) if cp.exists() else {}
    need = sorted({cands['smiles'][int(i)] for d in cands['pidx'] for v in d.values() if v is not None for i in v} - set(FM))
    if need:
        with mp.get_context('spawn').Pool(workers) as p:
            FM.update(zip(need, p.map(fragment_masses2, need, chunksize=64)))
        pickle.dump(FM, open(cp, 'wb'))
    raw = pickle.load(open(OUT / tag / 'query_raw.pkl', 'rb'))
    new = []
    for r, d in zip(recs, cands['pidx']):
        rr = dict(r)
        specs = []
        for q in r['h']['q_rows']:
            sp = raw[q]
            m2, i2 = _clean(np.asarray(sp['mz'], np.float32), np.asarray(sp['it'], np.float32), cfg.INT_FLOOR,
                            cfg.MAX_PEAKS, 1.0, False)
            specs.append((np.asarray(m2, float), np.asarray(i2, float), sp['mode'], sp['adduct'], sp['prec']))
        cache = {}
        for s in 'ABC':
            p = r[s]
            if p is None:
                continue
            rows = []
            for i in d[s]:
                i = int(i)
                if i not in cache:
                    fm, fk = FM[cands['smiles'][i]]
                    v = np.stack([spec_scores(fm, fk, *sp) for sp in specs]) if specs else np.zeros((1, 8), np.float32)
                    cache[i] = np.concatenate([v.max(0), v.mean(0)[[1, 3]]])
                rows.append(cache[i])
            F = np.stack(rows).astype(np.float32)
            ex = [F]
            for c in (1, 3, 4):
                ex += [_z(F[:, c])[:, None], _rank_norm(F[:, c])[:, None], (F[:, c] - F[:, c].max())[:, None]]
            rr[s] = dict(p, X=np.hstack([p['X']] + ex).astype(np.float32))
        new.append(rr)
    return new


# --------------------------------------------------------------------------------------------- rankers
class LGBRank:
    """LightGBM LambdaRank grouped by (molecule, split); the class prior w1 enters as per-group weights."""

    def __init__(self, cfg, priors=None, seeds=(0, 1), params=None):
        self.priors = priors or cfg.W1_PRIORS; self.seeds = seeds
        self.params = dict(objective='lambdarank', learning_rate=0.03, num_leaves=31, min_data_in_leaf=50,
                           lambda_l2=1.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                           lambdarank_truncation_level=25, eval_at=[25], verbose=-1, num_threads=0)
        if params:
            self.params.update(params)
        self.n_iter = self.params.pop('n_iter', 500)

    def fit(self, X, Y, M, G):
        import lightgbm as lgb
        o = np.argsort(G, kind='stable'); X, Y, M, G = X[o], Y[o], M[o], G[o]
        _, cnt = np.unique(G, return_counts=True)
        gm = M[np.cumsum(cnt) - 1]
        self.models = []
        for w1 in self.priors:
            gw = np.where(gm == 0, w1, 1 - w1)
            for sd in self.seeds:
                ds = lgb.Dataset(X, Y, group=cnt, weight=np.repeat(gw, cnt))
                self.models.append(lgb.train(dict(self.params, seed=sd), ds, self.n_iter))
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


class Blend:
    """Mean of within-window rank scores of the HistGBM (binary) and LightGBM (lambdarank) rankers."""

    def __init__(self, a, b, wa=0.5):
        self.a, self.b, self.wa = a, b, wa

    def predict(self, X):
        from .features import _rank_norm
        return self.wa * (1 - _rank_norm(self.a.predict(X))) + (1 - self.wa) * (1 - _rank_norm(self.b.predict(X)))


def _stack(recs, split, gbase):
    X, Y, M, G = [], [], [], []
    for j, r in enumerate(recs):
        p = r[split]
        if p is None or p['n'] == 0:
            continue
        X.append(p['X']); Y.append(p['Y']); M.append(np.full(p['n'], 0 if split == 'A' else 1))
        G.append(np.full(p['n'], gbase + j))
    return X, Y, M, G


def evaluate(recs, cfg, ranker='hgb', n_folds=5, seed=0, drop_cols=None, lgb_params=None):
    from .ranker import Ranker
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(recs)) % n_folds
    rr = {s: np.full(len(recs), np.nan) for s in 'ABC'}
    keep = None
    for f in range(n_folds):
        tr = [recs[i] for i in range(len(recs)) if fold[i] != f]
        parts = [_stack(tr, 'A', 0), _stack(tr, 'B', 10 ** 6)]
        X = np.vstack([x for p in parts for x in p[0]]); Y = np.concatenate([x for p in parts for x in p[1]])
        M = np.concatenate([x for p in parts for x in p[2]]); G = np.concatenate([x for p in parts for x in p[3]])
        if drop_cols is not None:
            keep = np.setdiff1d(np.arange(X.shape[1]), drop_cols); X = X[:, keep]
        if ranker == 'hgb':
            rk = Ranker(cfg).fit(X, Y, M)
        elif ranker == 'blend':
            rk = Blend(Ranker(cfg).fit(X, Y, M), LGBRank(cfg, params=lgb_params).fit(X, Y, M, G))
        else:
            rk = LGBRank(cfg, params=lgb_params).fit(X, Y, M, G)
        for i in np.where(fold == f)[0]:
            for s in 'ABC':
                p = recs[i][s]
                if s == 'A' and not recs[i]['h']['eligible_A']:
                    continue
                if p is None:
                    rr[s][i] = 0.0; continue
                x = p['X'] if keep is None else p['X'][:, keep]
                rr[s][i] = rr_from_scores(p, rk.predict(x))
    return rr, fold


def report(recs, rr, fold, base=None, n_boot=2000):
    libs = np.array([r['h']['src_lib'] for r in recs])
    np_ = libs == NP_LIB
    A = np.nan_to_num(rr['A']); B = rr['B']
    permol = WEIGHTS['A'] * A + WEIGHTS['B'] * B  # np molecules are all A-eligible
    line = (f"A[np]={np.nanmean(rr['A'][np_]):.4f} B[np]={np.nanmean(B[np_]):.4f} "
            f"W[np]={permol[np_].mean():.4f} B[e180]={np.nanmean(B[~np_]):.4f}")
    fw = [permol[np_ & (fold == f)].mean() for f in range(fold.max() + 1)]
    line += f"  fold W[np] std={np.std(fw):.4f}"
    if base is not None:
        bA = np.nan_to_num(base['A']); bperm = WEIGHTS['A'] * bA + WEIGHTS['B'] * base['B']
        d = (permol - bperm)[np_]
        rng = np.random.default_rng(1)
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
        dB = (B - base['B'])[np_]
        line += (f"\n  delta W[np]={d.mean():+.4f} (boot se {bs.std():.4f}, P(>0)={np.mean(bs > 0):.2f}); "
                 f"dA[np]={np.nanmean((rr['A'] - base['A'])[np_]):+.4f} dB[np]={dB.mean():+.4f} "
                 f"dB[e180]={np.nanmean((B - base['B'])[~np_]):+.4f}; np mols up/down {int((d > 1e-9).sum())}/{int((d < -1e-9).sum())}")
    return line


def main(argv=None):
    ap = argparse.ArgumentParser('casmi.cvx')
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cands'); c.add_argument('--tag', default='exp000')
    z = sub.add_parser('zlog'); z.add_argument('--tag', default='exp000')
    k = sub.add_parser('cmp'); k.add_argument('--tag', default='exp000'); k.add_argument('--base', default='base'); k.add_argument('exps', nargs='+')
    r = sub.add_parser('run')
    r.add_argument('--tag', default='exp000'); r.add_argument('--exp', required=True)
    r.add_argument('--groups', default='', help='prior groups: desc,desc_rel,formula,coco')
    r.add_argument('--grouprel', action='store_true', help='add within-window gap/softmax features')
    r.add_argument('--fpblock', action='store_true', help='add per-family FPNet agreement features')
    r.add_argument('--frag2', action='store_true', help='add MetFrag-lite v2 features (frag2.py)')
    r.add_argument('--ranker', default='hgb', choices=['hgb', 'lgb_rank', 'blend'])
    r.add_argument('--lgb', default='', help='k=v,... LightGBM param overrides')
    r.add_argument('--base', default='base', help='exp name of baseline run for paired bootstrap')
    r.add_argument('--fold-seed', type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == 'cands':
        recover_candidates(a.tag); return
    if a.cmd == 'zlog':
        cache_logits(a.tag); return
    if a.cmd == 'cmp':
        recs = pickle.load(open(OUT / a.tag / 'feats.pkl', 'rb'))
        base = dict(np.load(OUT / a.tag / f'cvx_{a.base}.npz'))
        for e in a.exps:
            rr = dict(np.load(OUT / a.tag / f'cvx_{e}.npz'))
            print(f'[{e}]  ' + report(recs, rr, rr['fold'], None if e == a.base else base))
        return
    cfg = config.CFG()
    t0 = time.time()
    recs = pickle.load(open(OUT / a.tag / 'feats.pkl', 'rb'))
    names = []
    if a.groups:
        cands = pickle.load(open(OUT / a.tag / 'cands.pkl', 'rb'))
        recs, names = augment(recs, cands, tuple(a.groups.split(',')))
    if a.grouprel:
        recs = group_rel(recs)
    if a.frag2:
        recs = frag2_feats(recs, pickle.load(open(OUT / a.tag / 'cands.pkl', 'rb')), a.tag)
    if a.fpblock:
        recs = fp_block(recs, pickle.load(open(OUT / a.tag / 'cands.pkl', 'rb')), a.tag)
    lp = {}
    for kv in filter(None, a.lgb.split(',')):
        k, v = kv.split('='); lp[k] = type(LGBRank(cfg).params.get(k, 1.0))(float(v)) if k != 'n_iter' else int(v)
    rr, fold = evaluate(recs, cfg, a.ranker, seed=a.fold_seed, lgb_params=lp)
    np.savez(OUT / a.tag / f'cvx_{a.exp}.npz', fold=fold, **rr)
    bp = OUT / a.tag / f'cvx_{a.base}.npz'
    base = dict(np.load(bp)) if bp.exists() and a.base != a.exp else None
    print(f'[{a.exp}] +{len(names)} prior cols grouprel={a.grouprel} fpblock={a.fpblock} frag2={a.frag2} ranker={a.ranker} {lp} ({time.time()-t0:.0f}s)')
    print('  ' + report(recs, rr, fold, base), flush=True)




def cache_logits(tag='exp000'):
    """FPNet z per CV molecule (same queries as feats.pkl) -> outputs/cv/<tag>/zlog.npy (NaN rows if no query)."""
    import glob
    from .cv import load_query_rows, make_query
    from .fpnet import ModelBank
    recs = pickle.load(open(OUT / tag / 'feats.pkl', 'rb'))
    bank = ModelBank(sorted(glob.glob(str(config.FP_MODEL_DIR / 'fp_*.pt'))))
    raw = load_query_rows([i for r in recs for i in r['h']['q_rows']])
    Z = np.full((len(recs), bank.nbits), np.nan, np.float32)
    for j, r in enumerate(recs):
        q = make_query(None, r['h'], raw)
        if q is not None:
            z = bank.logits(q.spectra)
            if z is not None:
                Z[j] = z
    np.save(OUT / tag / 'zlog.npy', Z)
    pickle.dump(raw, open(OUT / tag / 'query_raw.pkl', 'wb'))
    print('zlog', Z.shape, np.isnan(Z[:, 0]).sum(), 'missing')


if __name__ == '__main__':
    main()
