"""Leak-free CV on timsTOF held-out molecules.

Held-out molecules = structures sampled from enveda-np-examples and enveda-180 (the only timsTOF libraries),
deduplicated by the METRIC key (tautomer-canonical InChIKey14). Each is queried with k of its own timsTOF spectra
(k drawn from the test's spectra-per-molecule distribution). The "target group" G = every library structure
(inchikey14) that shares the target's metric key, so tautomer/stereo duplicates cannot leak.

  split A (class-1-like): hide G's spectra in the query's source library only; other libraries keep theirs.
                          Only molecules whose G has spectra in another library are eligible.
  split B (class-2-like): hide ALL spectra of G (library search + analog representatives); G stays in the pool.
  split C (class-3-like): as B, and also remove every pool member with the target's metric key.

The ranker is retrained inside CV (5 folds by molecule; rows from A (M=0) + B (M=1) of the training folds).
rank_train.npz is NOT used for CV numbers: it stores no molecule identities (only 819 group ids) and was built from
a 250-NP + 569-obscure holdout, i.e. almost certainly contains enveda-np-examples. It is reported only as a
"leaky reference" row.

Known remaining leak: the attached FPNet weights (casmi26-fp-models-v2) were trained on train.parquet with an
unknown structure split, so held-out structures may have been seen by the fingerprint model. cv reports a no-FP
variant and the FP-alone MRR per source library so the size of that leak is visible.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl

from . import config
from .engine import Mask, Query
from .features import FEATURE_NAMES
from .metric import score_key
from .ranker import Ranker

NP_LIB, E180_LIB = 'enveda-np-examples', 'enveda-180'
WEIGHTS = dict(A=0.16, B=0.45, C=0.39)
FP_COLS = list(range(15, 21)) + list(range(25, 31))


# --------------------------------------------------------------------------------------------- holdout
def select_holdout(L, n_np=200, n_e180=300, seed=0):
    rng = np.random.default_rng(seed)
    te = pl.read_parquet(config.TEST, columns=['molecule_id'])
    k_dist = te.group_by('molecule_id').len()['len'].to_numpy()
    chosen, used_mkeys = [], set()
    for lib, n in ((NP_LIB, n_np), (E180_LIB, n_e180)):
        sids = np.unique(L.sid[L.lib == lib])
        rng.shuffle(sids)
        got = 0
        for s in sids:
            mk = L.s_mkey[s]
            if mk is None or mk in used_mkeys:
                continue
            G = np.array(sorted(L.mkey2sids[mk]), np.int64)
            sp = np.concatenate([L.spectra_of(g) for g in G])
            q_sp = sp[(L.lib[sp] == lib) & np.isfinite(L.nm[sp])]
            if len(q_sp) == 0:
                continue
            k = int(min(len(q_sp), rng.choice(k_dist)))
            q_sp = np.sort(rng.choice(q_sp, size=k, replace=False))
            eligible_A = bool((L.lib[sp] != lib).any())
            used_mkeys.add(mk)
            chosen.append(dict(sid=int(s), mkey=mk, smiles=L.s_smiles[s], src_lib=lib, G=G.tolist(),
                               q_rows=q_sp.tolist(), eligible_A=eligible_A))
            got += 1
            if got >= n:
                break
    return chosen


def load_query_rows(rows):
    """Raw (uncleaned) spectra for library row indices, read from train.parquet (row index == library index)."""
    rows = sorted(set(rows))
    df = (pl.scan_parquet(config.TRAIN).with_row_index('row')
            .filter(pl.col('row').is_in(rows))
            .select('row', 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct', 'instrument_type',
                    'collision_energy_ev', 'ionization_mode').collect())
    out = {}
    for r in df.iter_rows(named=True):
        out[r['row']] = dict(mz=np.asarray(r['ms2_mzs'], np.float64), it=np.asarray(r['ms2_normalized_intensities'], np.float64),
                             prec=float(r['precursor_mz']), adduct=r['adduct'], instrument=r['instrument_type'],
                             ce=r['collision_energy_ev'], mode=1.0 if r['ionization_mode'] == 'positive' else -1.0)
    return out


def make_query(L, h, raw):
    from .chem import neutral_mass
    specs = [raw[i] for i in h['q_rows']]
    nm = neutral_mass(np.array([s['prec'] for s in specs]), np.array([s['adduct'] for s in specs], dtype=object))
    nm = nm[np.isfinite(nm)]
    if not len(nm):
        return None
    return Query(specs, float(np.median(nm)))


# --------------------------------------------------------------------------------------------- features
def _pack(c, truth):
    if c is None:
        return None
    keys = [score_key(s) for s in c['smiles']]
    Y = np.array([k == truth for k in keys], np.float32)
    return dict(X=c['X'], Y=Y, keys=keys, n=len(keys))


def compute_features(E, holdout, raw, log_every=25):
    L = E.L
    out, t0 = [], time.time()
    tim = dict(logits=0.0, lib=0.0, analog=0.0, cand=0.0)
    for i, h in enumerate(holdout):
        q = make_query(L, h, raw)
        rec = dict(h=h, A=None, B=None, C=None)
        if q is not None:
            truth = h['mkey']; G = np.array(h['G'], np.int64)
            ta = time.time(); z = E.logits(q); tim['logits'] += time.time() - ta
            mB = Mask(excl_sids=G)
            ta = time.time(); libB = E.lib_sim(q, mB); tim['lib'] += time.time() - ta
            ta = time.time(); anB = E.analog_sim(q, mB); tim['analog'] += time.time() - ta
            ta = time.time()
            rec['B'] = _pack(E.candidates(q, libB, anB, z), truth)
            rec['C'] = _pack(E.candidates(q, libB, anB, z, pool_excl_mkeys=frozenset([truth])), truth)
            tim['cand'] += time.time() - ta
            if h['eligible_A']:
                mA = Mask(src_sids=G, src_lib=h['src_lib'])
                bad = lambda idx: mA.spectra_bad(L, idx)
                ov = {int(s): L.rep_excluding(int(s), bad) for s in G}
                ta = time.time(); libA = E.lib_sim(q, mA); tim['lib'] += time.time() - ta
                ta = time.time(); anA = E.analog_sim(q, mA, ov); tim['analog'] += time.time() - ta
                ta = time.time(); rec['A'] = _pack(E.candidates(q, libA, anA, z), truth); tim['cand'] += time.time() - ta
            rec['target'] = q.target; rec['n_spec'] = len(q.spectra)
            rec['lib_max_B'] = float(max(libB.values())) if libB else 0.0
        out.append(rec)
        if (i + 1) % log_every == 0 or i + 1 == len(holdout):
            el = time.time() - t0
            print(f'  cv feats {i+1}/{len(holdout)}  {el:.0f}s ({el/(i+1):.2f}s/mol)  '
                  + ' '.join(f'{k}={v:.0f}s' for k, v in tim.items()), flush=True)
    return out, tim


# --------------------------------------------------------------------------------------------- evaluation
def rr_from_scores(pack, score, topn=25):
    """Reciprocal rank after metric-key dedupe + invalid filtering (exactly what submit.select does)."""
    if pack is None:
        return 0.0
    order = np.argsort(-score, kind='stable')
    seen, r = set(), 0
    for i in order:
        k = pack['keys'][i]
        if k is None or k in seen:
            continue
        seen.add(k); r += 1
        if pack['Y'][i] > 0:
            return 1.0 / r
        if r >= topn:
            break
    return 0.0


def _stack(recs, split, cols_zero=None):
    X, Y, M = [], [], []
    for r in recs:
        p = r[split]
        if p is None or p['n'] == 0:
            continue
        x = p['X'].copy()
        if cols_zero:
            x[:, cols_zero] = 0.0
        X.append(x); Y.append(p['Y']); M.append(np.full(p['n'], 0 if split == 'A' else 1))
    if not X:
        return None
    return np.vstack(X), np.concatenate(Y), np.concatenate(M)


def evaluate(recs, cfg, n_folds=5, seed=0, variants=('full', 'noFP'), use_rank_train=True):
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(recs)) % n_folds
    res = {}
    t0 = time.time()
    for var in variants:
        cz = FP_COLS if var == 'noFP' else None
        rr = {s: np.full(len(recs), np.nan) for s in 'ABC'}
        for f in range(n_folds):
            tr = [recs[i] for i in range(len(recs)) if fold[i] != f]
            parts = [p for p in (_stack(tr, 'A', cz), _stack(tr, 'B', cz)) if p is not None]
            X = np.vstack([p[0] for p in parts]); Y = np.concatenate([p[1] for p in parts])
            M = np.concatenate([p[2] for p in parts])
            rk = Ranker(cfg).fit(X, Y, M)
            for i in np.where(fold == f)[0]:
                for s in 'ABC':
                    p = recs[i][s]
                    if s == 'A' and not recs[i]['h']['eligible_A']:
                        continue
                    if p is None:
                        rr[s][i] = 0.0; continue
                    x = p['X'].copy()
                    if cz: x[:, cz] = 0.0
                    rr[s][i] = rr_from_scores(p, rk.predict(x))
        res[var] = rr
        print(f'  ranker CV [{var}] done ({time.time()-t0:.0f}s)', flush=True)
    if use_rank_train and config.RANK_TRAIN.exists():
        rk = Ranker.from_rank_train(cfg)
        rr = {s: np.full(len(recs), np.nan) for s in 'ABC'}
        for i, r in enumerate(recs):
            for s in 'ABC':
                if s == 'A' and not r['h']['eligible_A']:
                    continue
                rr[s][i] = rr_from_scores(r[s], rk.predict(r[s]['X'])) if r[s] is not None else 0.0
        res['rank_train(leaky)'] = rr
    # single-channel baselines on split B (no ranker => no ranker leak); FP-alone exposes the FPNet leak
    single = {}
    for name, col in (('lib', 0), ('analog_ap', 5), ('fpnet_fz', 17), ('frag', 21)):
        v = np.full(len(recs), np.nan)
        for i, r in enumerate(recs):
            v[i] = rr_from_scores(r['B'], r['B']['X'][:, col]) if r['B'] is not None else 0.0
        single[name] = v
    return res, single, fold


def summarize(recs, res, single):
    libs = np.array([r['h']['src_lib'] for r in recs])
    lines = []

    def m(v, mask=None):
        v = v if mask is None else v[mask]
        v = v[~np.isnan(v)]
        return (float(v.mean()) if len(v) else float('nan')), len(v)

    table = {}
    for var, rr in res.items():
        row = {}
        for s in 'ABC':
            row[s] = m(rr[s])
            for lib in (NP_LIB, E180_LIB):
                row[f'{s}@{lib}'] = m(rr[s], libs == lib)
        row['total'] = sum(WEIGHTS[s] * row[s][0] for s in 'ABC')
        # per-source weighted totals (A taken from the same source when available, else the pooled A)
        for lib in (NP_LIB, E180_LIB):
            a = row[f'A@{lib}'][0] if row[f'A@{lib}'][1] >= 20 else row['A'][0]
            row[f'total@{lib}'] = WEIGHTS['A'] * a + WEIGHTS['B'] * row[f'B@{lib}'][0] + WEIGHTS['C'] * row[f'C@{lib}'][0]
        table[var] = row
        lines.append(f"{var:>18}: A={row['A'][0]:.4f} (n={row['A'][1]})  B={row['B'][0]:.4f} (n={row['B'][1]})  "
                     f"C={row['C'][0]:.4f}  weighted(16/45/39)={row['total']:.4f}  "
                     f"weighted[np]={row['total@' + NP_LIB]:.4f}  weighted[e180]={row['total@' + E180_LIB]:.4f}")
        lines.append(' ' * 20 + '  '.join(f"{s}[{'np' if lib == NP_LIB else 'e180'}]={row[f'{s}@{lib}'][0]:.4f}(n={row[f'{s}@{lib}'][1]})"
                                            for s in 'AB' for lib in (NP_LIB, E180_LIB)))
    lines.append('single-channel MRR on split B (no ranker):')
    for name, v in single.items():
        lines.append(f"   {name:>10}: all={m(v)[0]:.4f}  np={m(v, libs == NP_LIB)[0]:.4f}  e180={m(v, libs == E180_LIB)[0]:.4f}")
    recall = np.mean([r['B'] is not None and r['B']['Y'].any() for r in recs])
    ncand = np.median([r['B']['n'] for r in recs if r['B'] is not None])
    lines.append(f'split-B window recall={recall:.4f}  median candidates={ncand:.0f}  '
                 f"A-eligible={np.mean([r['h']['eligible_A'] for r in recs]):.3f}")
    return table, '\n'.join(lines)


def run_cv(E, cfg, n_np=200, n_e180=300, seed=0, tag='exp000', reuse=True):
    outdir = config.OUTPUTS / 'cv' / tag
    outdir.mkdir(parents=True, exist_ok=True)
    fpath = outdir / 'feats.pkl'
    T = {}
    t0 = time.time()
    if reuse and fpath.exists():
        recs = pickle.load(open(fpath, 'rb')); tim = {}
        print(f'reusing {fpath} ({len(recs)} molecules)')
    else:
        hold = select_holdout(E.L, n_np, n_e180, seed)
        print(f'holdout: {len(hold)} molecules ({sum(h["src_lib"] == NP_LIB for h in hold)} np-examples, '
              f'{sum(h["eligible_A"] for h in hold)} A-eligible)', flush=True)
        raw = load_query_rows([i for h in hold for i in h['q_rows']])
        T['load_queries'] = time.time() - t0
        recs, tim = compute_features(E, hold, raw)
        T['features'] = time.time() - t0 - T['load_queries']
        pickle.dump(recs, open(fpath, 'wb'))
    t1 = time.time()
    res, single, fold = evaluate(recs, cfg)
    T['ranker_cv'] = time.time() - t1
    table, text = summarize(recs, res, single)
    print(text, flush=True)
    json.dump(dict(table=table, timings=T, channel_timings=tim, n=len(recs)), open(outdir / 'summary.json', 'w'),
              indent=1, default=str)
    (outdir / 'summary.txt').write_text(text + '\n' + json.dumps(dict(timings=T, channel_timings=tim), default=str) + '\n')
    return table, text, T


def cv_ranker(cfg, tag='exp000', cols_zero=None):
    """Ranker fitted on ALL CV rows (A as M=0, B as M=1) -- for test submissions with a leak-free-trained ranker."""
    recs = pickle.load(open(config.OUTPUTS / 'cv' / tag / 'feats.pkl', 'rb'))
    parts = [p for p in (_stack(recs, 'A', cols_zero), _stack(recs, 'B', cols_zero)) if p is not None]
    X = np.vstack([p[0] for p in parts]); Y = np.concatenate([p[1] for p in parts]); M = np.concatenate([p[2] for p in parts])
    return Ranker(cfg).fit(X, Y, M)


__all__ = ['run_cv', 'select_holdout', 'evaluate', 'summarize', 'cv_ranker', 'FEATURE_NAMES']
