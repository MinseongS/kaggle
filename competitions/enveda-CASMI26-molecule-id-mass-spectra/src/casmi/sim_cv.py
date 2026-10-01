"""CV of ICEBERG isomer re-scoring on the exp000 leak-free holdout (cached features / candidates; no channel recompute).

  uv run python -m casmi.sim_cv leak                 # MassSpecGym overlap of CV truths (+ top-K decoys)
  uv run python -m casmi.sim_cv oof --name base      # per-candidate out-of-fold ranker scores (cvx folds)
  uv run python -m casmi.sim_cv oof --name best --groups desc,desc_rel,formula --fpblock --ranker blend
  uv run python -m casmi.sim_cv jobs --k 60 --shards 6
  (run .venv-ice/bin/python data/ext/casmi26-iceberg/ice_runner.py per shard; see research/06-sim-rescoring.md)
  uv run python -m casmi.sim_cv eval --name best

Outputs under outputs/cv/exp000/sim/:
  msg_overlap.json    per CV molecule: truth metric key in MassSpecGym (any fold) / plain InChIKey14 match
  oof_<name>.pkl      [{split: scores array aligned with feats.pkl candidates}], fold
  formulas.pkl        {pool idx: formula}
  ice_in_<i>.json     runner shards;  ice_out_<i>.json runner outputs
  ice_scores.pkl      {mol index: {smiles: score|None}}   <- the per-(molecule, candidate) similarity cache
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np

from . import config
from .cv import NP_LIB, WEIGHTS
from .sim import fuse_formula_groups, fuse_rank_blend, items_from_specs

OUT = config.OUTPUTS / 'cv'
MSG_TSV = config.EXT / 'massspecgym' / 'MassSpecGym.tsv'


def sdir(tag):
    d = OUT / tag / 'sim'
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------------------------- leakage
def _ik14_plain(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m)[:14] if m is not None else None


def leak(tag='exp000'):
    import polars as pl
    from .metric import score_keys_parallel
    msg = pl.read_csv(MSG_TSV, separator='\t', columns=['smiles', 'inchikey', 'fold'])
    usmi = msg.unique('smiles')
    cache = sdir(tag) / 'msg_keys.pkl'
    if cache.exists():
        k_of = pickle.load(open(cache, 'rb'))
    else:
        sm = usmi['smiles'].to_list()
        k_of = dict(zip(sm, score_keys_parallel(sm, workers=8)))
        pickle.dump(k_of, open(cache, 'wb'))
    msg_mkeys = set(v for v in k_of.values() if v)
    msg_ik = set(msg['inchikey'].str.slice(0, 14).to_list())
    fold_of = {}
    for s, f in zip(usmi['smiles'].to_list(), usmi['fold'].to_list()):
        fold_of.setdefault(k_of.get(s), f)
    recs = pickle.load(open(OUT / tag / 'feats.pkl', 'rb'))
    cands = pickle.load(open(OUT / tag / 'cands.pkl', 'rb'))
    rows = []
    for j, r in enumerate(recs):
        h = r['h']
        ik = _ik14_plain(h['smiles'])
        rows.append(dict(j=j, lib=h['src_lib'], mkey=h['mkey'], in_msg=h['mkey'] in msg_mkeys,
                         in_msg_ik=ik in msg_ik, msg_fold=fold_of.get(h['mkey']), eligible_A=h['eligible_A']))
    # decoys: share of window candidates (split B) whose metric key is in MSG
    dec = []
    for r in recs:
        p = r['B']
        if p is None:
            continue
        ks = [k for k, y in zip(p['keys'], p['Y']) if y == 0]
        if ks:
            dec.append(np.mean([k in msg_mkeys for k in ks]))
    json.dump(rows, open(sdir(tag) / 'msg_overlap.json', 'w'), indent=0)
    for lib in (NP_LIB, 'enveda-180'):
        rr = [x for x in rows if x['lib'] == lib]
        print(f'{lib}: n={len(rr)} truth in MSG (metric key)={sum(x["in_msg"] for x in rr)} '
              f'(plain ik14 {sum(x["in_msg_ik"] for x in rr)}); among A-eligible '
              f'{sum(x["in_msg"] for x in rr if x["eligible_A"])}/{sum(x["eligible_A"] for x in rr)}; folds '
              f'{dict(zip(*np.unique([x["msg_fold"] for x in rr if x["in_msg"]], return_counts=True)))}')
    print(f'decoy share in MSG (split B window, mean over mols) = {np.mean(dec):.3f}; MSG unique keys {len(msg_mkeys)}')


# --------------------------------------------------------------------------------------------- OOF ranker scores
def oof(tag='exp000', name='base', groups='', fpblock=False, ranker='hgb', fold_seed=0):
    """Same folds / training rows as cvx.evaluate, but keeps per-candidate scores (higher = better)."""
    from . import cvx
    from .cv import rr_from_scores
    from .ranker import Ranker
    cfg = config.CFG()
    t0 = time.time()
    recs = pickle.load(open(OUT / tag / 'feats.pkl', 'rb'))
    if groups:
        recs, _ = cvx.augment(recs, pickle.load(open(OUT / tag / 'cands.pkl', 'rb')), tuple(groups.split(',')))
    if fpblock:
        recs = cvx.fp_block(recs, pickle.load(open(OUT / tag / 'cands.pkl', 'rb')), tag)
    rng = np.random.default_rng(fold_seed)
    fold = rng.permutation(len(recs)) % 5
    S = [dict() for _ in recs]
    rr = {s: np.full(len(recs), np.nan) for s in 'ABC'}
    for f in range(5):
        tr = [recs[i] for i in range(len(recs)) if fold[i] != f]
        parts = [cvx._stack(tr, 'A', 0), cvx._stack(tr, 'B', 10 ** 6)]
        X = np.vstack([x for p in parts for x in p[0]]); Y = np.concatenate([x for p in parts for x in p[1]])
        M = np.concatenate([x for p in parts for x in p[2]]); G = np.concatenate([x for p in parts for x in p[3]])
        if ranker == 'hgb':
            rk = Ranker(cfg).fit(X, Y, M)
        elif ranker == 'blend':
            rk = cvx.Blend(Ranker(cfg).fit(X, Y, M), cvx.LGBRank(cfg).fit(X, Y, M, G))
        else:
            rk = cvx.LGBRank(cfg).fit(X, Y, M, G)
        for i in np.where(fold == f)[0]:
            for s in 'ABC':
                p = recs[i][s]
                if p is None or (s == 'A' and not recs[i]['h']['eligible_A']):
                    continue
                S[i][s] = rk.predict(p['X']).astype(np.float64)
                rr[s][i] = rr_from_scores(p, S[i][s])
        print(f'  fold {f} done {time.time()-t0:.0f}s', flush=True)
    pickle.dump(dict(scores=S, fold=fold, rr=rr), open(sdir(tag) / f'oof_{name}.pkl', 'wb'))
    np_ = np.array([r['h']['src_lib'] == NP_LIB for r in recs])
    print(f'[{name}] A[np]={np.nanmean(rr["A"][np_]):.4f} B[np]={np.nanmean(rr["B"][np_]):.4f} '
          f'W[np]={(WEIGHTS["A"]*np.nan_to_num(rr["A"])+WEIGHTS["B"]*rr["B"])[np_].mean():.4f} '
          f'B[e180]={np.nanmean(rr["B"][~np_]):.4f} ({time.time()-t0:.0f}s)')


# --------------------------------------------------------------------------------------------- jobs
def _formulas(cands, tag):
    fp = sdir(tag) / 'formulas.pkl'
    if fp.exists():
        return pickle.load(open(fp, 'rb'))
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    F = {}
    for i, s in cands['smiles'].items():
        m = Chem.MolFromSmiles(s)
        F[i] = CalcMolFormula(m) if m is not None else f'?{i}'
    pickle.dump(F, open(fp, 'wb'))
    return F


def ranked(p, pid, score, k):
    """Ranked, metric-key-deduped list of (feats row index) -- exactly the submit order; first k."""
    order = np.argsort(-score, kind='stable')
    seen, out = set(), []
    for i in order:
        key = p['keys'][i]
        if key is None or key in seen or pid[i] < 0:
            continue
        seen.add(key); out.append(int(i))
        if len(out) >= k:
            break
    return out


def jobs(tag='exp000', k=60, shards=6, names=('base', 'best')):
    recs = pickle.load(open(OUT / tag / 'feats.pkl', 'rb'))
    cands = pickle.load(open(OUT / tag / 'cands.pkl', 'rb'))
    raw = pickle.load(open(OUT / tag / 'query_raw.pkl', 'rb'))
    O = [pickle.load(open(sdir(tag) / f'oof_{n}.pkl', 'rb')) for n in names if (sdir(tag) / f'oof_{n}.pkl').exists()]
    specs, mc = {}, {}
    for j, r in enumerate(recs):
        specs[j] = [raw[q] for q in r['h']['q_rows']]
        sm = []
        for o in O:
            for s in 'ABC':
                if s not in o['scores'][j]:
                    continue
                p = r[s]; pid = cands['pidx'][j][s]
                sm += [cands['smiles'][int(pid[i])] for i in ranked(p, pid, o['scores'][j][s], k)]
        mc[j] = list(dict.fromkeys(sm))
    items = items_from_specs(specs, mc)
    # balance shards by job count (candidates x distinct CE conditions)
    cost = [len(it['cands']) * len({(s['adduct'], tuple(s['ce_ev'])) for s in it['spectra']}) for it in items]
    buckets = [[] for _ in range(shards)]; load = np.zeros(shards)
    for i in np.argsort(-np.array(cost)):
        b = int(np.argmin(load)); buckets[b].append(items[i]); load[b] += cost[i]
    for b, its in enumerate(buckets):
        json.dump(its, open(sdir(tag) / f'ice_in_{b}.json', 'w'))
    print(f'{len(items)} covered molecules, {sum(len(it["cands"]) for it in items)} candidates, ~{sum(cost)} jobs '
          f'-> {shards} shards (load {load.astype(int).tolist()})')


def resume(tag='exp000'):
    """After an interrupted run: move finished molecules of ice_out_<i>.json into ice_done_<i>.json and rewrite
    ice_in_<i>.json with the molecules that still have no score (then relaunch the runners)."""
    import glob
    for fin in sorted(glob.glob(str(sdir(tag) / 'ice_in_*.json'))):
        i = fin.rsplit('_', 1)[1][:-5]
        fout = sdir(tag) / f'ice_out_{i}.json'; fdone = sdir(tag) / f'ice_done_{i}.json'
        done = json.load(open(fdone)) if fdone.exists() else {}
        if fout.exists():
            for mid, d in json.load(open(fout)).items():
                if any(v is not None for v in d.values()):
                    done[mid] = d
        json.dump(done, open(fdone, 'w'))
        todo = [it for it in json.load(open(fin)) if it['mid'] not in done]
        json.dump(todo, open(fin, 'w'))
        print(f'shard {i}: done {len(done)} todo {len(todo)}')


def collect(tag='exp000'):
    import glob
    S = {}
    files = sorted(glob.glob(str(sdir(tag) / 'ice_done_*.json'))) + sorted(glob.glob(str(sdir(tag) / 'ice_out_*.json')))
    for f in [f for f in files if not f.endswith('.meta.json')]:   # runner writes ice_out_<i>.json.meta.json too
        for mid, d in json.load(open(f)).items():
            if int(mid) not in S or any(v is not None for v in d.values()):
                S[int(mid)] = d
    pickle.dump(S, open(sdir(tag) / 'ice_scores.pkl', 'wb'))
    n = sum(len(v) for v in S.values()); nn = sum(v is not None for d in S.values() for v in d.values())
    print(f'collected {len(S)} molecules, {nn}/{n} candidate scores')
    return S


# --------------------------------------------------------------------------------------------- evaluation
def _rr_from_order(keys_ranked, truth, topn=25):
    for r, key in enumerate(keys_ranked[:topn]):
        if key == truth:
            return 1.0 / (r + 1)
    return 0.0


def method_rr(recs, cands, F, S, O, j, split, method, param, k=60):
    """Reciprocal rank of molecule j / split after re-ranking the ranker's top-k with the sim scores."""
    r = recs[j]; p = r[split]
    if p is None:
        return 0.0
    pid = cands['pidx'][j][split]
    sc = O['scores'][j][split]
    lst = ranked(p, pid, sc, 10 ** 9)                    # full deduped ranked list
    keys = [p['keys'][i] for i in lst]
    ice = S.get(j) or {}
    if method != 'ranker' and ice:
        top = lst[:k]
        smis = [cands['smiles'][int(pid[i])] for i in top]
        sims = [ice.get(s) for s in smis]
        if method == 'formula':
            order = fuse_formula_groups([sc[i] for i in top], sims, [F[int(pid[i])] for i in top], lam=param, top_n=k)
        elif method == 'blend':
            order = fuse_rank_blend([sc[i] for i in top], sims, w=param, top_n=k)
        elif method == 'sim':                              # sim alone inside top-k (uncovered keep ranker order after)
            order = sorted(range(len(top)), key=lambda q: (-(sims[q] if sims[q] is not None else -1.0), q))
        keys = [keys[q] for q in order] + keys[k:]
    return _rr_from_order(keys, r['h']['mkey'])


def _perf(rrA, rrB):
    return WEIGHTS['A'] * np.nan_to_num(rrA) + WEIGHTS['B'] * rrB


def evaluate(tag='exp000', name='best', k=60, n_boot=2000):
    recs = pickle.load(open(OUT / tag / 'feats.pkl', 'rb'))
    cands = pickle.load(open(OUT / tag / 'cands.pkl', 'rb'))
    F = _formulas(cands, tag)
    O = pickle.load(open(sdir(tag) / f'oof_{name}.pkl', 'rb'))
    S = pickle.load(open(sdir(tag) / 'ice_scores.pkl', 'rb'))
    ov = {x['j']: x for x in json.load(open(sdir(tag) / 'msg_overlap.json'))}
    fold = O['fold']
    n = len(recs)
    lib = np.array([r['h']['src_lib'] for r in recs])
    inmsg = np.array([ov[j]['in_msg'] for j in range(n)])
    intr = np.array([ov[j]['in_msg'] and ov[j]['msg_fold'] == 'train' for j in range(n)])   # seen by msg_all ICEBERG
    cov = np.array([bool(S.get(j)) and any(v is not None for v in S[j].values()) for j in range(n)])
    eligA = np.array([r['h']['eligible_A'] for r in recs])

    def grid_rr(method, param):
        out = {}
        for s in 'AB':
            v = np.full(n, np.nan)
            for j in range(n):
                if s == 'A' and not eligA[j]:
                    continue
                if s not in O['scores'][j]:
                    v[j] = 0.0 if recs[j][s] is None else np.nan
                    continue
                v[j] = method_rr(recs, cands, F, S, O, j, s, method, param, k)
            out[s] = v
        return out

    base = grid_rr('ranker', 0)
    # sanity: equals the stored OOF rr
    assert np.allclose(np.nan_to_num(base['B']), np.nan_to_num(O['rr']['B'])), 'rank reproduction mismatch'
    grids = {'formula': [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], 'blend': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], 'sim': [0]}
    R = {(m, prm): grid_rr(m, prm) for m, ps in grids.items() for prm in ps}
    R[('ranker', 0)] = base
    pickle.dump(dict(R=R, fold=fold), open(sdir(tag) / f'eval_{name}_k{k}.pkl', 'wb'))

    def boot(d, rng=np.random.default_rng(1)):
        return np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]).std()

    def nested(method, mask_fn):
        """Pick the param on the other 4 folds (W[np] over np molecules with sim coverage, excluding MSG-overlap
        molecules when mask_fn does), apply to the held-out fold."""
        ps = grids[method]
        out = {s: np.full(n, np.nan) for s in 'AB'}
        chosen = []
        for f in range(5):
            tr = (fold != f) & mask_fn
            best = max(ps, key=lambda q: np.nanmean(_perf(R[(method, q)]['A'], R[(method, q)]['B'])[tr]))
            chosen.append(best)
            te = fold == f
            for s in 'AB':
                out[s][te] = R[(method, best)][s][te]
        return out, chosen

    lines = [f'# ICEBERG re-rank on top-{k} of ranker "{name}" (exp000 holdout)']
    subsets = {
        'np all (n=250)': lib == NP_LIB,
        'np sim-covered': (lib == NP_LIB) & cov,
        'np not-in-MSG-train (clean)': (lib == NP_LIB) & ~intr,
        'np not-in-MSG-train covered': (lib == NP_LIB) & ~intr & cov,
        'np in-MSG-train (leaky)': (lib == NP_LIB) & intr,
        'non-np (e180 or msgfree) all': lib != NP_LIB,
        'non-np sim-covered': (lib != NP_LIB) & cov,
    }
    rows = [('ranker', 0)] + [(m, q) for m in ('formula', 'blend', 'sim') for q in grids[m]]
    for sname, msk in subsets.items():
        if msk.sum() == 0:
            continue
        lines.append(f'\n## {sname}: n={int(msk.sum())}')
        lines.append('| method | param | A | B | W(.16A+.45B) | dW vs ranker (se) | dB (se) |')
        lines.append('|---|---|---|---|---|---|---|')
        bw = _perf(base['A'], base['B'])
        for m, q in rows:
            rr = R[(m, q)]
            w = _perf(rr['A'], rr['B'])
            d = (w - bw)[msk]; dB = (rr['B'] - base['B'])[msk]
            lines.append(f'| {m} | {q} | {np.nanmean(rr["A"][msk]):.4f} | {np.nanmean(rr["B"][msk]):.4f} | '
                         f'{np.nanmean(w[msk]):.4f} | {np.nanmean(d):+.4f} ({boot(np.nan_to_num(d)):.4f}) | '
                         f'{np.nanmean(dB):+.4f} ({boot(np.nan_to_num(dB)):.4f}) |')
        for m in ('formula', 'blend'):
            for tune_name, tmask in (('tune np', lib == NP_LIB), ('tune non-np', lib != NP_LIB)):
                if tmask.sum() == 0:
                    continue
                nr, ch = nested(m, tmask)
                w = _perf(nr['A'], nr['B']); d = (w - bw)[msk]; dB = (nr['B'] - base['B'])[msk]
                lines.append(f'| nested {m} ({tune_name}) | {ch} | {np.nanmean(nr["A"][msk]):.4f} | '
                             f'{np.nanmean(nr["B"][msk]):.4f} | {np.nanmean(w[msk]):.4f} | {np.nanmean(d):+.4f} '
                             f'({boot(np.nan_to_num(d)):.4f}) | {np.nanmean(dB):+.4f} ({boot(np.nan_to_num(dB)):.4f}) |')
    txt = '\n'.join(lines)
    (sdir(tag) / f'eval_{name}_k{k}.md').write_text(txt + '\n')
    print(txt)


# --------------------------------------------------------------------------------------------- MSG-free holdout
def msgfree_holdout(n=300, seed=7, tag='exp000'):
    """Class-2-like holdout that neither simulator nor FPNet(h1) has seen: metric keys from holdout_v1 'fpval'
    (excluded from h1 FPNet training), NOT in MassSpecGym (any fold), with no timsTOF-library spectra; queried with
    k of its own spectra from one source library (k ~ test spectra-per-molecule). Split B only."""
    import polars as pl
    from .library import Library
    from .train_fpnet import load_holdout
    k_of = pickle.load(open(sdir(tag) / 'msg_keys.pkl', 'rb'))
    msg = set(v for v in k_of.values() if v)
    H = load_holdout(); fv = set(H['fpval']) - msg
    L = Library(config.CFG())
    te = pl.read_parquet(config.TEST, columns=['molecule_id'])
    k_dist = te.group_by('molecule_id').len()['len'].to_numpy()
    rng = np.random.default_rng(seed)
    keys = sorted(k for k in fv if k in L.mkey2sids)
    rng.shuffle(keys)
    out = []
    for mk in keys:
        G = np.array(sorted(L.mkey2sids[mk]), np.int64)
        sp = np.concatenate([L.spectra_of(g) for g in G])
        libs = set(L.lib[sp].tolist())
        if libs & {'enveda-180', NP_LIB}:
            continue
        pos = sp[(L.meta['adduct'].to_numpy()[sp] == '[M+H]+') & np.isfinite(L.nm[sp])]
        if not len(pos):
            continue
        lib = L.lib[pos[0]]
        q = sp[(L.lib[sp] == lib) & np.isfinite(L.nm[sp])]
        k = int(min(len(q), rng.choice(k_dist)))
        q = np.sort(rng.choice(q, size=k, replace=False))
        s0 = int(G[0])
        out.append(dict(sid=s0, mkey=mk, smiles=L.s_smiles[s0], src_lib=str(lib), G=G.tolist(), q_rows=q.tolist(),
                        eligible_A=False))
        if len(out) >= n:
            break
    d = OUT / 'msgfree'; d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / 'holdout.json', 'w'))
    from collections import Counter
    print(f'msgfree holdout {len(out)}: {Counter(h["src_lib"] for h in out)}')


def msgfree_feats():
    """Channel features for the msgfree holdout with the h1 FPNet (leak-free for these keys)."""
    from pathlib import Path
    from .cli import _engine
    from .cv import compute_features, load_query_rows
    config.FP_MODEL_DIR = config.EXT / 'casmi26-fp-models-h1'
    cfg = config.CFG()
    d = OUT / 'msgfree'
    hold = json.load(open(d / 'holdout.json'))
    E = _engine(cfg)
    raw = load_query_rows([i for h in hold for i in h['q_rows']])
    pickle.dump(raw, open(d / 'query_raw.pkl', 'wb'))
    recs, tim = compute_features(E, hold, raw)
    pickle.dump(recs, open(d / 'feats.pkl', 'wb'))
    if E.frag is not None:
        E.frag.close()


def msgfree_scores(train_tag='exp001-fph1', variant='full'):
    """Base HGB ranker (31 cols) fitted on ALL rows of train_tag (A as M=0, B as M=1) -> scores for msgfree split B
    (independent molecules, so no folds needed). variant noFP zeroes the FP columns in train and test."""
    from .cv import FP_COLS, cv_ranker, rr_from_scores
    cfg = config.CFG()
    cz = FP_COLS if variant == 'noFP' else None
    rk = cv_ranker(cfg, train_tag, cz)
    recs = pickle.load(open(OUT / 'msgfree' / 'feats.pkl', 'rb'))
    S, rr = [], {s: np.full(len(recs), np.nan) for s in 'ABC'}
    for j, r in enumerate(recs):
        d = {}
        for s in 'BC':
            p = r[s]
            if p is None:
                continue
            x = p['X'].copy()
            if cz:
                x[:, cz] = 0.0
            d[s] = rk.predict(x).astype(np.float64); rr[s][j] = rr_from_scores(p, d[s])
        S.append(d)
    fold = np.random.default_rng(0).permutation(len(recs)) % 5
    pickle.dump(dict(scores=S, fold=fold, rr=rr), open(sdir('msgfree') / f'oof_{variant}.pkl', 'wb'))
    print(f'[msgfree {variant}] B={np.nanmean(rr["B"]):.4f} n={int(np.isfinite(rr["B"]).sum())}')


def main(argv=None):
    ap = argparse.ArgumentParser('casmi.sim_cv')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('leak').add_argument('--tag', default='exp000')
    o = sub.add_parser('oof')
    o.add_argument('--tag', default='exp000'); o.add_argument('--name', required=True)
    o.add_argument('--groups', default=''); o.add_argument('--fpblock', action='store_true')
    o.add_argument('--ranker', default='hgb', choices=['hgb', 'lgb_rank', 'blend'])
    j = sub.add_parser('jobs'); j.add_argument('--tag', default='exp000'); j.add_argument('--k', type=int, default=60)
    j.add_argument('--shards', type=int, default=6)
    j.add_argument('--names', default='base,best')
    sub.add_parser('collect').add_argument('--tag', default='exp000')
    sub.add_parser('resume').add_argument('--tag', default='exp000')
    e = sub.add_parser('eval'); e.add_argument('--tag', default='exp000'); e.add_argument('--name', default='best')
    e.add_argument('--k', type=int, default=60)
    mh = sub.add_parser('msgfree'); mh.add_argument('step', choices=['holdout', 'feats', 'scores'])
    mh.add_argument('--variant', default='full')
    a = ap.parse_args(argv)
    if a.cmd == 'msgfree':
        {'holdout': msgfree_holdout, 'feats': msgfree_feats,
         'scores': lambda: msgfree_scores(variant=a.variant)}[a.step]()
    elif a.cmd == 'leak':
        leak(a.tag)
    elif a.cmd == 'oof':
        oof(a.tag, a.name, a.groups, a.fpblock, a.ranker)
    elif a.cmd == 'jobs':
        jobs(a.tag, a.k, a.shards, tuple(a.names.split(',')))
    elif a.cmd == 'resume':
        resume(a.tag)
    elif a.cmd == 'collect':
        collect(a.tag)
    elif a.cmd == 'eval':
        evaluate(a.tag, a.name, a.k)


if __name__ == '__main__':
    main()
