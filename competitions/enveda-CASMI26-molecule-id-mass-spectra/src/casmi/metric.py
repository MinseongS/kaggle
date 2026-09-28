"""Official CASMI metric, re-implemented exactly (kaggle.com/code/metric/casmi-mean-reciprocal-rank).

key(smiles) = InChIKey(TautomerEnumerator().Canonicalize(MolFromSmiles(smiles)))[:14]  with RDKit 2026.03.3.
Score per molecule = 1/rank of the first guess whose key is in the accepted set, 0 if none within max_rank.
Empty entries (A;;B) are dropped before ranking; a trailing ';' is ignored.
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

EXPECTED_RDKIT_VERSION = '2026.03.3'
_T = None


def _taut():
    global _T
    if _T is None:
        import rdkit
        from rdkit import RDLogger
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog('rdApp.*')
        if rdkit.__version__ != EXPECTED_RDKIT_VERSION:
            warnings.warn(f'RDKit {rdkit.__version__} != metric {EXPECTED_RDKIT_VERSION}: tautomer keys may differ')
        _T = rdMolStandardize.TautomerEnumerator()
    return _T


@lru_cache(maxsize=2_000_000)
def score_key(smiles):
    """Tautomer-canonical InChIKey14, or None if RDKit cannot parse/canonicalize (== metric's canonicalize_smiles_to_ik14)."""
    if not isinstance(smiles, str):
        return None
    from rdkit import Chem
    te = _taut()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        cm = te.Canonicalize(mol)
        ik = Chem.MolToInchiKey(cm)
        return ik[:14] if ik else None
    except (ValueError, RuntimeError, TypeError):
        return None


def score_keys_parallel(smiles_list, workers=8, chunksize=256):
    """Compute score keys for many SMILES with a process pool (order preserved)."""
    import multiprocessing as mp
    if workers <= 1 or len(smiles_list) < 2000:
        return [score_key(s) for s in smiles_list]
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers) as p:
        return p.map(_score_key_plain, smiles_list, chunksize=chunksize)


def _score_key_plain(s):
    return score_key(s)


def parse_guesses(raw):
    return [c.strip() for c in str(raw).split(';') if c.strip()]


def reciprocal_rank(guesses, accepted_keys, max_rank=25):
    for r, g in enumerate(guesses[:max_rank], 1):
        if score_key(g) in accepted_keys:
            return 1.0 / r
    return 0.0


def mrr(solution: dict, submission: dict, max_rank=25):
    """solution: {molecule_id: answer smiles (comma-separated alternatives allowed)};
       submission: {molecule_id: 'smi1;smi2;...' or list}. Missing molecules score 0."""
    rr = []
    for mid, ans in solution.items():
        acc = {k for k in map(score_key, [s for s in ans.split(',') if s]) if k}
        if not acc:
            raise ValueError(f'unparseable solution for {mid}: {ans!r}')
        g = submission.get(mid, [])
        if isinstance(g, str):
            g = parse_guesses(g)
        if len(g) > max_rank:
            raise ValueError(f'{mid}: {len(g)} guesses > {max_rank}')
        rr.append(reciprocal_rank(g, acc, max_rank))
    return float(np.mean(rr)) if rr else 0.0


def rank_of_key(ranked_keys, truth_key, max_rank=25):
    """1-based rank of truth among already-keyed guesses, or 0 if absent (fast path for CV)."""
    for r, k in enumerate(ranked_keys[:max_rank], 1):
        if k == truth_key:
            return r
    return 0
