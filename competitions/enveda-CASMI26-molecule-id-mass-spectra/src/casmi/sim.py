"""Spectrum-simulation isomer re-scoring (ICEBERG, MassSpecGym `msg_all` weights) -- runtime helpers.

The simulator itself runs in a SEPARATE interpreter (RDKit 2025.03 is required by ICEBERG's fragment engine; the main
project pins RDKit 2026.03 for the metric). We reuse the public offline package `ahmedberatozer/casmi26-iceberg`
(data/ext/casmi26-iceberg: ice_runner.py + ms_pred MIT subset + pure-torch DGL shim + stripped MSG checkpoints):

    local : .venv-ice/bin/python data/ext/casmi26-iceberg/ice_runner.py PKG in.json out.json --device cpu --site none
    kaggle: python ice_runner.py PKG in.json out.json --device cuda --site /kaggle/working/ice_site  (installs wheels)

Score (runner definition, `ice_ex_mean`): for each covered query spectrum (positive [M+H]+ / [M+Na]+) ICEBERG predicts
the candidate at the query's adduct and 5-eV CE bucket (multi-CE / missing CE -> 20+40+60 merged), then entropy
similarity (0.01 Da / 20 ppm) query vs prediction; mean over the molecule's covered spectra. None = not covered.

This module is pure python/numpy (no torch / rdkit) so the main pipeline can import it:
  items_from_specs()  -> runner input
  run_runner()        -> {mid: {smiles: score|None}}
  fuse_formula_groups -> v4f post-ranker re-rank (z(ranker) + lam z(sim) inside same-formula groups, slots kept)
  fuse_rank_blend     -> alternative: (1-w) rank(ranker) + w rank(sim) inside the top-K
  sim_features        -> per-candidate columns for a ranker (raw, z / rank within formula group, gap to group max)
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ICE_PKG = ROOT / 'data' / 'ext' / 'casmi26-iceberg'
ICE_PY = ROOT / '.venv-ice' / 'bin' / 'python'
COV = ('[M+H]+', '[M+Na]+')


# --------------------------------------------------------------------------------------------- runner I/O
def spec_item(sp):
    """Our query-spectrum dict (cv.load_query_rows / engine.Query style: mz, it, prec, adduct, mode(+-1), instrument,
    ce (list of eV or None)) -> runner spectrum dict."""
    ce = sp.get('ce')
    ce = [] if ce is None else [float(v) for v in np.atleast_1d(np.asarray(ce, float)) if np.isfinite(v)]
    mode = sp.get('mode')
    mode = 'positive' if (mode == 1 or mode == 1.0 or mode == 'positive') else 'negative'
    return dict(mz=[float(v) for v in sp['mz']], it=[float(v) for v in sp['it']], prec=float(sp['prec']),
                adduct=sp.get('adduct'), mode=mode, instrument=sp.get('instrument'), ce_ev=ce, ce_orig=None,
                ce_units='eV')


def covered(sp):
    m = sp.get('mode')
    return (m == 1 or m == 1.0 or m == 'positive') and sp.get('adduct') in COV


def items_from_specs(mol_specs, mol_cands):
    """mol_specs: {mid: [spectrum dict]}, mol_cands: {mid: [smiles]} (priority order = dict order)."""
    out = []
    for mid, cands in mol_cands.items():
        sp = [spec_item(s) for s in mol_specs[mid] if covered(s)]
        if sp and cands:
            out.append(dict(mid=str(mid), spectra=sp, cands=list(dict.fromkeys(cands))))
    return out


def run_runner(items, workdir, device='cpu', budget_s=0.0, site='none', python=None, pkg=ICE_PKG, threads=0,
               extra=()):
    """Blocking subprocess run; returns ({mid: {smiles: score|None}}, meta)."""
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    fin, fout = workdir / 'ice_in.json', workdir / 'ice_out.json'
    json.dump(items, open(fin, 'w'))
    cmd = [str(python or ICE_PY), str(Path(pkg) / 'ice_runner.py'), str(pkg), str(fin), str(fout), '--device', device,
           '--budget', str(float(budget_s)), '--site', str(site), '--quiet']
    if threads:
        cmd += ['--threads', str(threads)]
    cmd += list(extra)
    subprocess.run(cmd, check=False)
    out = json.load(open(fout)) if fout.exists() else {}
    meta = json.load(open(str(fout) + '.meta.json')) if Path(str(fout) + '.meta.json').exists() else {}
    return out, meta


# --------------------------------------------------------------------------------------------- fusion
def _z(vals):
    """pandas-style group z (ddof=1) over finite members; missing / n<2 / sd 0 -> 0 (= fuse._z in casmi26-iceberg)."""
    v = np.asarray([np.nan if x is None else float(x) for x in vals], float)
    ok = np.isfinite(v)
    out = np.zeros(len(v))
    if ok.sum() < 2:
        return out
    mu = v[ok].mean(); sd = v[ok].std(ddof=1)
    if not sd > 0:
        return out
    out[ok] = (v[ok] - mu) / sd
    return out


def fuse_formula_groups(scores, sims, formulas, lam=0.5, top_n=60, min_covered=2):
    """v4f re-rank. scores/sims/formulas aligned with the CURRENT ranked order (index 0 = best). Returns a permutation
    (list of indices). Each same-formula group inside the first top_n positions with >= min_covered finite sims is
    re-sorted by z(score) + lam*z(sim) and put back into the same slots."""
    n = len(scores)
    order = list(range(n))
    m = min(int(top_n), n)
    groups = {}
    for i in range(m):
        groups.setdefault(formulas[i], []).append(i)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        s = [sims[i] for i in idx]
        if sum(x is not None and math.isfinite(x) for x in s) < max(2, min_covered):
            continue
        fused = _z([scores[i] for i in idx]) + lam * _z(s)
        new = [idx[k] for k in sorted(range(len(idx)), key=lambda k: (-fused[k], k))]
        for slot, src in zip(idx, new):
            order[slot] = src
    return order


def fuse_rank_blend(scores, sims, w=0.5, top_n=60):
    """Global alternative: inside the first top_n (ranked order), fused = (1-w) * rank-score(ranker) + w *
    rank-score(sim) (rank-score in [0,1], 1 = best; missing sim -> 0.5). Positions > top_n untouched."""
    n = len(scores); m = min(int(top_n), n)
    if m < 2:
        return list(range(n))
    rs = 1.0 - np.arange(m) / (m - 1)
    s = np.array([np.nan if x is None else float(x) for x in sims[:m]])
    ok = np.isfinite(s)
    rq = np.full(m, 0.5)
    if ok.sum() >= 2:
        o = np.argsort(-s[ok], kind='stable')
        r = np.empty(ok.sum()); r[o] = 1.0 - np.arange(ok.sum()) / (ok.sum() - 1)
        rq[ok] = r
    f = (1 - w) * rs + w * rq
    top = sorted(range(m), key=lambda k: (-f[k], k))
    return top + list(range(m, n))


def sim_features(sims, formulas):
    """Per-candidate ranker columns: raw sim (NaN->-1), covered flag, z within formula group, rank-norm within
    formula group (1 = best), sim - group max. For later wiring into the ranker (not done here)."""
    s = np.array([np.nan if x is None else float(x) for x in sims])
    f = np.asarray(formulas, object)
    zc = np.zeros(len(s)); rk = np.zeros(len(s)); gap = np.zeros(len(s))
    for g in set(f.tolist()):
        idx = np.where(f == g)[0]
        zc[idx] = _z(s[idx])
        v = s[idx]; ok = np.isfinite(v)
        if ok.sum():
            o = np.argsort(-v[ok], kind='stable'); r = np.empty(ok.sum())
            r[o] = 1.0 - np.arange(ok.sum()) / max(ok.sum() - 1, 1)
            tmp = np.zeros(len(idx)); tmp[ok] = r; rk[idx] = tmp
            gp = np.zeros(len(idx)); gp[ok] = v[ok] - v[ok].max(); gap[idx] = gp
    return np.column_stack([np.nan_to_num(s, nan=-1.0), np.isfinite(s), zc, rk, gap]).astype(np.float32)


SIM_FEATURE_NAMES = ['ice_sim', 'ice_cov', 'ice_z_form', 'ice_rank_form', 'ice_gap_form']
