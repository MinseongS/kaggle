"""MetFrag-lite: fraction of sqrt-intensity explained by 1- or 2-bond-break fragments (+-2 H). Verbatim port;
fragment masses are cached per SMILES and computed with a persistent worker pool."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from .spectra import _clean

AMU = {'C': 12.0, 'H': 1.00782503207, 'N': 14.0030740048, 'O': 15.9949146196, 'P': 30.97376163,
       'S': 31.97207100, 'F': 18.99840322, 'Cl': 34.96885268, 'Br': 78.9183371, 'I': 126.904473,
       'Na': 22.9897692809, 'K': 38.96370668, 'Si': 27.9769265325, 'B': 11.0093054, 'Se': 79.9165213}
H = AMU['H']; PROTON = H - 0.00054857990


def mol_graph(smi):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    n = m.GetNumAtoms()
    w = np.zeros(n)
    for a in m.GetAtoms():
        w[a.GetIdx()] = AMU.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H
    if (w == 0).any(): return None
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    return w, bonds, n


def _components(n, bonds, drop):
    adj = [[] for _ in range(n)]
    for i, (a, b) in enumerate(bonds):
        if i in drop: continue
        adj[a].append(b); adj[b].append(a)
    seen = np.zeros(n, bool); comps = []
    for s in range(n):
        if seen[s]: continue
        stack = [s]; seen[s] = True; cur = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]: seen[v] = True; stack.append(v); cur.append(v)
        comps.append(cur)
    return comps


def fragment_masses(smi, max_breaks=2, max_bonds=34):
    g = mol_graph(smi)
    if g is None: return np.zeros(0)
    w, bonds, n = g
    nb = len(bonds)
    if nb == 0 or nb > max_bonds: return np.array([w.sum()])
    out = {w.sum()}
    for i in range(nb):
        for c in _components(n, bonds, {i}):
            out.add(float(w[c].sum()))
    if max_breaks >= 2:
        for i in range(nb):
            for j in range(i + 1, nb):
                for c in _components(n, bonds, {i, j}):
                    out.add(float(w[c].sum()))
    return np.array(sorted(out))


def explain_score(frag_mass, peak_mz, peak_int, mode=1.0, tol=0.01, h_shifts=(-2, -1, 0, 1, 2)):
    if len(frag_mass) == 0 or len(peak_mz) == 0: return 0.0
    ion = []
    for dh in h_shifts:
        ion.append(frag_mass + dh * H + (PROTON if mode > 0 else -PROTON))
    ion = np.sort(np.concatenate(ion))
    w = np.sqrt(np.asarray(peak_int, float)); tot = w.sum()
    if tot <= 0: return 0.0
    idx = np.searchsorted(ion, peak_mz)
    ok = np.zeros(len(peak_mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - peak_mz) <= tol
    return float(w[ok].sum() / tot)


def _frag_masses(smi):
    try:
        return fragment_masses(smi)
    except Exception:
        return np.zeros(0)


class Fragmenter:
    def __init__(self, workers=4):
        self.cache = {}
        self.workers = workers
        self._pool = None

    def _p(self):
        if self._pool is None and self.workers > 1:
            self._pool = mp.get_context('spawn').Pool(self.workers)
        return self._pool

    def masses(self, smiles):
        need = [s for s in dict.fromkeys(smiles) if s not in self.cache]
        if need:
            p = self._p()
            res = p.map(_frag_masses, need, chunksize=4) if p is not None and len(need) > 8 else list(map(_frag_masses, need))
            self.cache.update(zip(need, res))
        return [self.cache[s] for s in smiles]

    def scores(self, cand_smiles, specs, mode, cfg):
        """Max over the molecule's spectra of the explain-score, per candidate."""
        frags = self.masses(list(cand_smiles))
        peaks = []
        for mz, it in specs:
            m2, i2 = _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32), cfg.INT_FLOOR, cfg.MAX_PEAKS, 1.0, False)
            peaks.append((np.asarray(m2, float), np.asarray(i2, float)))
        out = np.zeros(len(cand_smiles), np.float32)
        for j, f in enumerate(frags):
            out[j] = max((explain_score(f, a, b, mode=mode, tol=cfg.MZ_TOL) for a, b in peaks), default=0.0)
        return out

    def close(self):
        if self._pool is not None:
            self._pool.close(); self._pool = None
