"""Per-molecule scoring: 4 evidence channels -> 31 features for every candidate in the +-10 ppm window.

Channels (see research/04-pipeline-analysis.md):
  1. library   : entropy similarity vs library spectra in the same neutral-mass window   (class-1 evidence)
  2. analog    : mass-shifted entropy similarity vs one representative per structure within +-200 Da,
                 propagated to candidates as max_a sim(a)^p * Tanimoto(c, a)            (class-2 evidence)
  3. frag      : MetFrag-lite explained-intensity fraction
  4. fpnet     : spectrum->fingerprint logits z, candidate score f.z

`Mask` lets CV hide evidence per query (library spectra, analog representatives, pool members) without rebuilding
any index -- see cv.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config
from .features import N_FEAT, rank_features
from .metric import score_key
from .spectra import clean, search_clean, search_shift_clean


@dataclass
class Mask:
    """Evidence hidden from one query.
    excl_sids   : structures whose spectra are removed from the library side entirely (split B/C)
    src_sids/src_lib : structures whose spectra are removed only from library `src_lib` (split A)
    pool_excl_mkeys  : metric keys removed from the candidate pool (split C)"""
    excl_sids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    src_sids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    src_lib: str | None = None
    pool_excl_mkeys: frozenset = frozenset()

    def spectra_bad(self, L, idx):
        if len(idx) == 0:
            return np.zeros(0, bool)
        s = L.sid[idx]
        bad = np.isin(s, self.excl_sids) if len(self.excl_sids) else np.zeros(len(idx), bool)
        if len(self.src_sids) and self.src_lib is not None:
            bad |= np.isin(s, self.src_sids) & (L.lib[idx] == self.src_lib)
        return bad

    @property
    def empty(self):
        return not len(self.excl_sids) and not len(self.src_sids) and not self.pool_excl_mkeys


NO_MASK = Mask()


@dataclass
class Query:
    spectra: list          # dict(mz, it, prec, adduct, instrument, ce, mode)
    target: float          # neutral mass (median over spectra)

    @property
    def peaks(self):
        return [(s['mz'], s['it']) for s in self.spectra]

    @property
    def mode(self):
        return float(np.mean([s['mode'] for s in self.spectra]))


class Engine:
    def __init__(self, cfg: config.CFG, L, pool, bank=None, frag=None, prefer_instrument='timsTOF'):
        self.cfg, self.L, self.pool, self.bank, self.frag = cfg, L, pool, bank, frag
        L.build_rep(prefer_instrument if cfg.ANALOG_MATCH_INSTR else None)
        self.key2sid = {k: i for i, k in enumerate(L.s_key)}
        self.sid2pidx = np.array([pool.k2i.get(k, -1) for k in L.s_key], np.int64)
        self.pidx2sid = np.full(len(pool.keys), -1, np.int64)
        ok = self.sid2pidx >= 0
        self.pidx2sid[self.sid2pidx[ok]] = np.where(ok)[0]

    # ------------------------------------------------------------------ channel 1
    def lib_sim(self, q: Query, mask: Mask = NO_MASK):
        """{sid: max entropy similarity} over library spectra within +-PPM_WIN of the target."""
        cfg, L = self.cfg, self.L
        cand = L.window(q.target, q.target * cfg.PPM_WIN / 1e6)
        if len(cand) and not mask.empty:
            cand = cand[~mask.spectra_bad(L, cand)]
        if len(cand) == 0:
            return {}
        best = np.full(len(cand), -1.0, np.float32)
        for mz, it in q.peaks:
            qm, qp = clean(mz, it, cfg)
            if len(qm) == 0:
                continue
            best = np.maximum(best, search_clean(qm, qp, cand, L.off, L.mz, L.it, cfg.MZ_TOL))
        agg = {}
        for s, v in zip(L.sid[cand].tolist(), best.tolist()):
            if v >= 0 and v > agg.get(s, -1.0):
                agg[s] = v
        return agg

    # ------------------------------------------------------------------ channel 2
    def analog_sim(self, q: Query, mask: Mask = NO_MASK, rep_override=None):
        """[(sid, sim)] top N_ANALOG by mass-shifted entropy similarity over +-ANALOG_WIN Da."""
        cfg, L = self.cfg, self.L
        lo = np.searchsorted(L.rep_nm, q.target - cfg.ANALOG_WIN, 'left')
        hi = np.searchsorted(L.rep_nm, q.target + cfg.ANALOG_WIN, 'right')
        cand = L.rep[lo:hi].copy(); csid = L.rep_sid[lo:hi]; cnm = L.rep_nm[lo:hi].copy()
        if len(cand) == 0:
            return []
        keep = np.ones(len(cand), bool)
        if len(mask.excl_sids):
            keep &= ~np.isin(csid, mask.excl_sids)
        if rep_override:
            pos = {s: i for i, s in enumerate(csid.tolist()) if s in rep_override}
            for s, r in rep_override.items():
                if s in pos:
                    if r < 0:
                        keep[pos[s]] = False
                    else:
                        cand[pos[s]] = r; cnm[pos[s]] = L.nm[r]
        cand, csid, cnm = cand[keep], csid[keep], cnm[keep]
        if len(cand) == 0:
            return []
        shift = (q.target - cnm).astype(np.float32)
        best = np.full(len(cand), -1.0, np.float32)
        any_q = False
        for mz, it in q.peaks:
            qm, qp = clean(mz, it, cfg)
            if len(qm) == 0:
                continue
            any_q = True
            best = np.maximum(best, search_shift_clean(qm, qp, cand, L.off, L.mz, L.it, cfg.MZ_TOL, shift))
        if not any_q:
            return []
        # one rep per sid -> agg == best; library spectra with no peaks keep 0 (as in the notebook)
        best = np.maximum(best, 0.0)
        o = np.argsort(-best, kind='stable')[:cfg.N_ANALOG]
        return list(zip(csid[o].tolist(), best[o].tolist()))

    # ------------------------------------------------------------------ channel 4
    def logits(self, q: Query):
        if self.bank is None or not self.cfg.USE_FP_MODEL:
            return None
        return self.bank.logits(q.spectra)

    # ------------------------------------------------------------------ candidates + features
    def candidates(self, q: Query, lib_hits, analogs, zlog, pool_excl_mkeys=frozenset()):
        cfg, pool = self.cfg, self.pool
        cand = pool.window(q.target, cfg.PPM_WIN)
        if len(cand) == 0:
            cand = pool.window(q.target, cfg.PPM_FALLBACK)
        if len(cand) and pool_excl_mkeys:
            cand = cand[np.array([score_key(pool.smiles[c]) not in pool_excl_mkeys for c in cand], bool)]
        if len(cand) == 0:
            return None
        lv = np.array([lib_hits.get(self.pidx2sid[c], 0.0) if self.pidx2sid[c] >= 0 else 0.0 for c in cand], np.float32)
        if cfg.CAND_CAP and len(cand) > cfg.CAND_CAP:
            coarse = lv * 100.0
            if zlog is not None:
                mz = pool.fps(cand).astype(np.float32) @ zlog
                coarse = coarse + (mz - mz.mean()) / max(float(mz.std()), 1e-9)
            else:
                coarse = coarse - np.abs(pool.mass[cand] - q.target)
            keep = np.argsort(-coarse)[:cfg.CAND_CAP]
            cand, lv = cand[keep], lv[keep]
        cfp = pool.fps(cand)
        csmi = [pool.smiles[c] for c in cand]
        ids, sims = [], []
        for s, v in analogs:
            i = self.sid2pidx[s]
            if i >= 0:
                ids.append(i); sims.append(v)
        afp = pool.fps(np.array(ids)) if ids else None
        fsc = self.frag.scores(csmi, q.peaks, q.mode, cfg) if (self.frag is not None and cfg.USE_FRAG) else None
        X = rank_features(cfp, lv, afp, np.array(sims, np.float32), zlog, fsc, p_sim=cfg.SIM_POWER)[:, :N_FEAT]
        return dict(pidx=cand, smiles=csmi, X=X, lv=lv, top_analog=float(sims[0]) if sims else 0.0)
