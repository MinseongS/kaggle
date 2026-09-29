"""MetFrag-lite v2 (CV experiment): fragment masses tagged by number of broken bonds, larger bond cap (the v1 port
returns only the precursor mass for molecules with > 34 bonds, i.e. most glycosides), adduct-aware ion types, and a
chance-match correction. Features are added on top of the 31 (see cvx.py --frag2).
"""
from __future__ import annotations

import numpy as np

from .frag import H, PROTON, _components, mol_graph

NA, NH4 = 22.9897692809 - 0.00054857990, 14.0030740048 + 4 * 1.00782503207 - 0.00054857990
H2O = 2 * 1.00782503207 + 15.9949146196


def fragment_masses2(smi, max_bonds=60):
    """(masses, n_breaks) unique by rounded mass keeping the fewest breaks."""
    try:
        g = mol_graph(smi)
    except Exception:
        g = None
    if g is None:
        return np.zeros(0), np.zeros(0, np.int8)
    w, bonds, n = g
    nb = len(bonds)
    best = {round(float(w.sum()), 4): 0}

    def add(m, k):
        r = round(float(m), 4)
        if best.get(r, 9) > k:
            best[r] = k

    if nb:
        for i in range(nb):
            for c in _components(n, bonds, {i}):
                add(w[c].sum(), 1)
        if nb <= max_bonds:
            for i in range(nb):
                for j in range(i + 1, nb):
                    for c in _components(n, bonds, {i, j}):
                        add(w[c].sum(), 2)
    ks = sorted(best)
    return np.array(ks), np.array([best[k] for k in ks], np.int8)


def _match(ion, mz, tol):
    if len(ion) == 0 or len(mz) == 0:
        return np.zeros(len(mz), bool)
    ion = np.sort(ion)
    idx = np.searchsorted(ion, mz)
    ok = np.zeros(len(mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - mz) <= tol
    return ok


def spec_scores(fm, fk, mz, it, mode, adduct, prec):
    """One spectrum -> vector of explanation scores for one candidate."""
    keep = mz < prec - 0.5  # fragment peaks only (precursor explained by every isomer)
    mz, it = mz[keep], it[keep]
    if len(mz) == 0 or len(fm) == 0:
        return np.zeros(8, np.float32)
    w = np.sqrt(it); tot = w.sum() + 1e-12
    ch = PROTON if mode > 0 else -PROTON
    hs = np.arange(-2, 3) * H

    def ions(m, extra=()):
        base = [m + h + ch for h in hs]
        for e in extra:
            base += [m + h + e for h in hs]
        return np.concatenate(base) if len(m) else np.zeros(0)

    extra = []
    if isinstance(adduct, str) and 'Na' in adduct:
        extra.append(NA)
    if isinstance(adduct, str) and 'NH4' in adduct:
        extra.append(NH4)
    m1 = fm[fk <= 1]
    ok1 = _match(ions(m1), mz, 0.01)
    ok2 = _match(ions(fm), mz, 0.01)
    ok2t = _match(ions(fm), mz, 0.005)
    ok2a = ok2 | _match(ions(fm, extra), mz, 0.01) | _match(ions(fm - H2O), mz, 0.01)
    # chance level: fraction of the m/z axis covered by ion windows
    span = max(float(prec), 50.0)
    chance = min(1.0, len(fm) * len(hs) * 2 * 0.01 / span)
    f2 = ok2 @ w / tot
    top = np.argsort(-it)[:10]
    return np.array([ok1 @ w / tot, f2, ok2t @ w / tot, ok2a @ w / tot, f2 - chance, chance,
                     ok2[top].mean(), ok2 @ it / (it.sum() + 1e-12)], np.float32)


FRAG2_NAMES = ['fr1', 'fr2', 'fr2_tight', 'fr2_adduct', 'fr2_minus_chance', 'fr_chance', 'fr2_top10', 'fr2_lin']
