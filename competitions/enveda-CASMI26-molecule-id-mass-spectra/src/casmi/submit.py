"""Final list selection + submission writing/validation.

Rules (one invalid SMILES reportedly zeroes the whole submission on Kaggle):
  * every output SMILES must parse AND tautomer-canonicalize with RDKit 2026.03.3 (score_key not None)
  * dedupe by the metric key (tautomer-canonical InChIKey14) so no slot is wasted on a duplicate
  * <= 25 per molecule, no empty entries, no ';' inside a SMILES, fallback 'CCO' if nothing survives
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl

from . import config
from .engine import Query
from .metric import parse_guesses, score_key

FALLBACK = 'CCO'


def select(smiles_ranked, n=25):
    """Walk a ranked SMILES list, keep valid & metric-unique entries. Returns (smiles, keys)."""
    out, keys, seen = [], [], set()
    for s in smiles_ranked:
        if not isinstance(s, str) or not s or ';' in s or ',' in s or s != s.strip():
            continue
        k = score_key(s)
        if k is None or k in seen:
            continue
        seen.add(k); out.append(s); keys.append(k)
        if len(out) >= n:
            break
    return out, keys


def validate_submission(df: pl.DataFrame, sample: pl.DataFrame, max_rank=25):
    assert df.columns[:2] == ['molecule_id', 'smiles'], df.columns
    assert len(df) == len(sample) and set(df['molecule_id']) == set(sample['molecule_id'])
    assert df['molecule_id'].is_unique().all(), 'duplicate molecule_id'
    assert df['smiles'].null_count() == 0
    n_bad = n_dup = 0
    for s in df['smiles'].to_list():
        g = parse_guesses(s)
        assert 1 <= len(g) <= max_rank, (len(g), s[:80])
        assert len(g) == len(s.split(';')), 'empty entry'
        ks = [score_key(x) for x in g]
        n_bad += sum(k is None for k in ks)
        n_dup += len(ks) - len(set(ks))
    assert n_bad == 0, f'{n_bad} unparseable SMILES'
    assert n_dup == 0, f'{n_dup} metric-duplicate SMILES'
    return True


def test_queries(path=config.TEST):
    from .chem import neutral_mass
    te = pl.read_parquet(path)
    nm = neutral_mass(te['precursor_mz'].to_numpy(), te['adduct'].to_numpy())
    te = te.with_columns(pl.Series('nm', nm))
    out = {}
    for mid, sub in te.group_by('molecule_id', maintain_order=True):
        mid = mid[0]
        nms = sub['nm'].to_numpy(); nms = nms[np.isfinite(nms)]
        if not len(nms):
            out[mid] = None; continue
        specs = [dict(mz=np.asarray(r['ms2_mzs'], np.float64), it=np.asarray(r['ms2_normalized_intensities'], np.float64),
                      prec=float(r['precursor_mz']), adduct=r['adduct'], instrument=r['instrument_type'],
                      ce=r['collision_energy_ev'], mode=1.0 if r['ionization_mode'] == 'positive' else -1.0)
                 for r in sub.iter_rows(named=True)]
        out[mid] = Query(specs, float(np.median(nms)))
    instr = te['instrument_type'].drop_nulls()
    pref = instr.mode().sort()[0] if len(instr) else None
    return out, pref


def predict_test(E, ranker, queries, cfg, log_every=50):
    """Rankers with a `prepare`/`features` pair (rows_v2.RankerV2) get a two-pass run: all candidate windows
    first, then one batched descriptor pass, then ranking. Other rankers score c['X'] (the 31 columns) directly."""
    rows, diag = {}, []
    t0 = time.time()
    two_pass = hasattr(ranker, 'prepare')
    wins = []
    for gi, (mid, q) in enumerate(queries.items()):
        c = z = None
        if q is not None:
            lib = E.lib_sim(q); an = E.analog_sim(q); z = E.logits(q)
            c = E.candidates(q, lib, an, z)
        if two_pass:
            wins.append((mid, q, c, z))
        else:
            rows[mid] = _rank_one(mid, q, c, ranker.predict(c['X']) if c is not None else None, cfg, diag)
        if gi % log_every == 0:
            print(f'  {gi}/{len(queries)}  {time.time()-t0:.0f}s', flush=True)
    if two_pass:
        ranker.prepare(E, [(q, c, z) for _, q, c, z in wins], cache=config.CACHE / 'prior_desc_test.pkl')
        for mid, q, c, z in wins:
            p = ranker.predict(ranker.features(E, q, c, z)) if c is not None else None
            rows[mid] = _rank_one(mid, q, c, p, cfg, diag)
        print(f'  ranked {len(wins)} ({time.time()-t0:.0f}s)', flush=True)
    return rows, diag


def _rank_one(mid, q, c, p, cfg, diag):
    smis = []
    if c is not None:
        order = np.argsort(-p)
        smis, _ = select([c["smiles"][i] for i in order], cfg.TOPN)
        diag.append(dict(molecule_id=mid, target=q.target, n_cand=len(c['smiles']), lib_max=float(c['lv'].max()),
                         top_analog=c['top_analog'], top_p=float(p[order[0]])))
    return smis or [FALLBACK]


def write_submission(rows: dict, out_path, sample_path=config.SAMPLE):
    sample = pl.read_csv(sample_path)
    df = pl.DataFrame(dict(molecule_id=sample['molecule_id'],
                           smiles=[';'.join(rows.get(m, [FALLBACK])) for m in sample['molecule_id'].to_list()]))
    validate_submission(df, sample)
    df.write_csv(out_path)
    return df
