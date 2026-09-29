"""Candidate prior features ("how natural-product-like / well-known is this structure"), additive to the 31 features.

Structure-intrinsic descriptors only (NP-likeness, SA score, complexity, rings, sugars, element counts) plus
window-relative versions (z-score / rank inside the candidate window) and same-formula group statistics.
Pool provenance (COCONUT vs train-only) is available as a SEPARATE opt-in column group ('coco') because in the CV
simulation every answer is a training-library structure -- see research/04-pipeline-analysis.md.

Used by cvx.py (CV on cached feats). Per-SMILES descriptors are cached in outputs/cv/prior_desc.pkl.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

from . import config

DESC_NAMES = ['np_score', 'sa_score', 'heavy', 'rings', 'arom_rings', 'fsp3', 'stereo', 'bertz', 'nO', 'nN',
              'nS_P_hal', 'hbd', 'logp', 'sugars', 'methoxy', 'charge_abs']
_NP = _SA = None


def _init():
    global _NP, _SA
    from rdkit import RDLogger
    from rdkit.Chem import RDConfig
    RDLogger.DisableLog('rdApp.*')
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'NP_Score'))
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import npscorer
    import sascorer
    _NP = (npscorer, npscorer.readNPModel())
    _SA = sascorer
    sascorer.readFragmentScores()


def _sugars(mol):
    ri = mol.GetRingInfo()
    n = 0
    for ring in ri.AtomRings():
        if len(ring) not in (5, 6):
            continue
        at = [mol.GetAtomWithIdx(i) for i in ring]
        if sum(a.GetSymbol() == 'O' for a in at) != 1 or sum(a.GetSymbol() == 'C' for a in at) != len(ring) - 1:
            continue
        if any(a.GetIsAromatic() for a in at):
            continue
        rs = set(ring)
        exo_o = 0
        for a in at:
            if a.GetSymbol() != 'C':
                continue
            for nb in a.GetNeighbors():
                if nb.GetIdx() not in rs and nb.GetSymbol() == 'O':
                    exo_o += 1
        n += exo_o >= 2
    return n


def describe(smi):
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, Lipinski, rdMolDescriptors
    if _NP is None:
        _init()
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return [np.nan] * len(DESC_NAMES) + [None]
        sym = [a.GetSymbol() for a in m.GetAtoms()]
        methoxy = len(m.GetSubstructMatches(Chem.MolFromSmarts('[CH3][OX2][#6]')))
        out = [
            _NP[0].scoreMol(m, _NP[1]), _SA.calculateScore(m), m.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcNumRings(m), rdMolDescriptors.CalcNumAromaticRings(m),
            rdMolDescriptors.CalcFractionCSP3(m),
            len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)),
            GraphDescriptors.BertzCT(m), sym.count('O'), sym.count('N'),
            sum(s in ('S', 'P', 'F', 'Cl', 'Br', 'I') for s in sym), Lipinski.NumHDonors(m), Crippen.MolLogP(m),
            _sugars(m), methoxy, sum(abs(a.GetFormalCharge()) for a in m.GetAtoms()),
        ]
        return [float(x) for x in out] + [rdMolDescriptors.CalcMolFormula(m)]
    except Exception:
        return [np.nan] * len(DESC_NAMES) + [None]


def describe_many(smiles, workers=None, cache=config.OUTPUTS / 'cv' / 'prior_desc.pkl'):
    """{smiles: (desc vector float32, formula)} with an on-disk cache."""
    D = pickle.load(open(cache, 'rb')) if cache and cache.exists() else {}
    todo = sorted({s for s in smiles if s not in D})
    if todo:
        workers = workers or max(1, (os.cpu_count() or 4) - 1)
        with mp.get_context('spawn').Pool(workers, initializer=_init) as p:
            res = p.map(describe, todo, chunksize=200)
        for s, r in zip(todo, res):
            D[s] = (np.asarray(r[:-1], np.float32), r[-1])
        if cache:
            pickle.dump(D, open(cache, 'wb'))
    return D


def _rank(x):
    o = np.argsort(-x, kind='stable'); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r / max(1, len(x) - 1)


def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


# column groups for ablation (names returned alongside)
GROUPS = ('desc', 'desc_rel', 'formula', 'coco')


def prior_features(smiles, mass, target, is_coco, D, groups=('desc', 'desc_rel', 'formula')):
    """Extra per-candidate features for one window. smiles/mass/is_coco are per candidate; target = neutral mass."""
    nc = len(smiles)
    desc = np.stack([D[s][0] for s in smiles]).astype(np.float32)
    desc = np.where(np.isfinite(desc), desc, 0.0)
    forms = [D[s][1] for s in smiles]
    cols, names = [], []
    if 'desc' in groups:
        cols.append(desc[:, :-1]); names += DESC_NAMES[:-1]  # charge_abs dropped (COCONUT is charge-filtered)
    if 'desc_rel' in groups:
        for j, nm in enumerate(DESC_NAMES[:2] + ['bertz', 'sugars', 'heavy']):
            x = desc[:, DESC_NAMES.index(nm)]
            cols += [_z(x)[:, None], _rank(x)[:, None]]; names += [f'{nm}_z', f'{nm}_rank']
    if 'formula' in groups:
        from collections import Counter
        cnt = Counter(forms)
        same = np.array([cnt[f] for f in forms], np.float32)
        ppm = (np.asarray(mass) - target) / target * 1e6
        cols += [np.log(same)[:, None], (same / nc)[:, None], np.abs(ppm)[:, None], _rank(-np.abs(ppm))[:, None],
                 np.full((nc, 1), len(cnt), np.float32)]
        names += ['log_same_formula', 'frac_same_formula', 'abs_ppm', 'ppm_rank', 'n_formulas']
    if 'coco' in groups:
        c = np.asarray(is_coco, np.float32)
        cols += [c[:, None], np.full((nc, 1), c.mean(), np.float32)]; names += ['is_coco', 'coco_frac']
    return np.hstack(cols).astype(np.float32), names
