"""Reference spectral library (train.parquet) as a compact on-disk cache + mass index.

Cache (data/cache/lib_*):
  lib_off.npy / lib_mz.npy / lib_it.npy   pre-cleaned peaks (CSR, float32) -- see spectra.clean_batch
  lib_meta.parquet                        per spectrum: sid, lib, instrument, adduct, mode, precursor_mz, nm, npk_raw, ce
  structs.parquet                         per structure (unique inchikey14): sid, inchikey14, smiles, formula, mkey
mkey = metric score key (tautomer-canonical InChIKey14) -- used for CV masking/labels, never as a feature.
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from . import chem, config
from .spectra import clean_batch

C = config.CACHE


def _lib_paths():
    return dict(off=C / 'lib_off.npy', mz=C / 'lib_mz.npy', it=C / 'lib_it.npy',
                meta=C / 'lib_meta.parquet', structs=C / 'structs.parquet')


def build_library_cache(cfg: config.CFG, force=False):
    P = _lib_paths()
    if all(p.exists() for p in P.values()) and not force:
        print('library cache exists'); return
    C.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # ---- structures
    lf = pl.scan_parquet(config.TRAIN)
    st = (lf.select('inchikey14', 'normalized_smiles', 'molecular_formula').with_row_index('row')
            .group_by('inchikey14').agg(pl.col('row').min(), pl.col('normalized_smiles').first(),
                                        pl.col('molecular_formula').first())
            .sort('row').drop('row').with_row_index('sid').collect())
    print(f'structures: {len(st):,}  ({time.time()-t0:.0f}s)', flush=True)
    from .metric import score_keys_parallel
    mk = score_keys_parallel(st['normalized_smiles'].to_list(), workers=cfg.WORKERS)
    st = st.with_columns(pl.Series('mkey', mk, dtype=pl.String))
    st.write_parquet(P['structs'])
    print(f'metric keys done: {st["mkey"].null_count()} null, {st["mkey"].n_unique():,} unique '
          f'({time.time()-t0:.0f}s)', flush=True)
    k2sid = dict(zip(st['inchikey14'].to_list(), st['sid'].to_list()))
    # ---- spectra, row group by row group
    pf = pq.ParquetFile(config.TRAIN)
    metas, offs, mzs, its = [], [], [], []
    base = 0
    fm_cache = {}
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=['ingest_lib', 'inchikey14', 'molecular_formula', 'instrument_type', 'adduct',
                                           'ionization_mode', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities',
                                           'collision_energy_ev'])
        mzc = t.column('ms2_mzs').combine_chunks(); itc = t.column('ms2_normalized_intensities').combine_chunks()
        off = mzc.offsets.to_numpy().astype(np.int64)
        off = off - off[0]
        mz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
        it = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
        noff, cmz, cit = clean_batch(off, mz, it, cfg.INT_FLOOR, cfg.MAX_PEAKS, cfg.INT_POWER, cfg.ENT_WEIGHT)
        offs.append(noff[1:] + base); base += int(noff[-1]); mzs.append(cmz); its.append(cit)
        df = pl.from_arrow(t.drop(['ms2_mzs', 'ms2_normalized_intensities']))
        fo = df['molecular_formula'].to_list()
        for f in set(fo):
            if f not in fm_cache:
                fm_cache[f] = chem.formula_mass(f)
        nmf = np.array([fm_cache[f] for f in fo], np.float64)
        nmp = chem.neutral_mass(df['precursor_mz'].to_numpy(), df['adduct'].to_numpy())
        ce = df['collision_energy_ev'].list.mean().fill_null(-1.0)
        metas.append(pl.DataFrame(dict(
            sid=np.array([k2sid[k] for k in df['inchikey14'].to_list()], np.int32),
            lib=df['ingest_lib'], instrument=df['instrument_type'], adduct=df['adduct'], mode=df['ionization_mode'],
            precursor_mz=df['precursor_mz'], nm_formula=nmf, nm_prec=nmp, npk_raw=np.diff(off).astype(np.int32),
            ce=ce)))
        print(f'  row group {rg+1}/{pf.num_row_groups}: {len(off)-1:,} spectra, {len(cmz):,} cleaned peaks '
              f'({time.time()-t0:.0f}s)', flush=True)
    np.save(P['off'], np.concatenate([np.zeros(1, np.int64)] + offs))
    np.save(P['mz'], np.concatenate(mzs)); np.save(P['it'], np.concatenate(its))
    pl.concat(metas).write_parquet(P['meta'])
    print(f'library cache written ({time.time()-t0:.0f}s)', flush=True)


class Library:
    def __init__(self, cfg: config.CFG):
        P = _lib_paths()
        t0 = time.time()
        self.cfg = cfg
        self.off = np.load(P['off']); self.mz = np.load(P['mz']); self.it = np.load(P['it'])
        meta = pl.read_parquet(P['meta'])
        self.meta = meta
        self.sid = meta['sid'].to_numpy()
        self.lib = meta['lib'].to_numpy()
        self.instrument = meta['instrument'].to_numpy()
        nmf = meta['nm_formula'].to_numpy(); nmp = meta['nm_prec'].to_numpy()
        self.nm = np.where(np.isfinite(nmf), nmf, nmp) if cfg.LIB_MASS_FORMULA else nmp
        ok = np.isfinite(self.nm)
        self.order = np.argsort(np.where(ok, self.nm, 1e18), kind='mergesort')
        self.n_ok = int(ok.sum())
        self.snm = self.nm[self.order][:self.n_ok]
        self.npk = meta['npk_raw'].to_numpy()
        st = pl.read_parquet(P['structs'])
        self.structs = st
        self.s_key = st['inchikey14'].to_numpy()
        self.s_smiles = st['normalized_smiles'].to_numpy()
        self.s_mkey = st['mkey'].to_numpy()
        # spectra grouped by structure (for per-query masking / representative re-selection)
        o = np.argsort(self.sid, kind='mergesort')
        self._by_sid = o
        self._sid_bounds = np.searchsorted(self.sid[o], np.arange(len(st) + 1))
        # mkey -> sids (tautomer/stereo duplicates share one metric key)
        self.mkey2sids = {}
        for s, k in enumerate(self.s_mkey):
            if k is not None:
                self.mkey2sids.setdefault(k, []).append(s)
        print(f'library: {len(self.sid):,} spectra / {len(st):,} structures, {len(self.mz):,} cleaned peaks '
              f'({time.time()-t0:.0f}s)', flush=True)

    def spectra_of(self, sid):
        return self._by_sid[self._sid_bounds[sid]:self._sid_bounds[sid + 1]]

    def window(self, target, tol):
        lo = np.searchsorted(self.snm, target - tol, 'left')
        hi = np.searchsorted(self.snm, target + tol, 'right')
        return self.order[lo:hi]

    # ------------------------------------------------------------------ analog representatives
    def build_rep(self, prefer_instrument=None):
        """One representative spectrum per structure (prefer the test instrument, then the richest raw spectrum)."""
        pref = None
        if prefer_instrument:
            pref = str(prefer_instrument).lower().replace('-', '').replace(' ', '')
        self._pref = pref
        match = self._instr_match()
        self._match = match
        n = len(self.structs)
        # vectorised: sort by (sid, match desc, npk desc, index asc) and take first of each sid
        idx = np.arange(len(self.sid))
        keys = np.lexsort((idx, -self.npk.astype(np.int64), -match.astype(np.int64), self.sid))
        first = np.ones(len(keys), bool)
        ss = self.sid[keys]
        first[1:] = ss[1:] != ss[:-1]
        rep = np.sort(keys[first])
        nm = self.nm[rep]; ok = np.isfinite(nm)
        rep = rep[ok]; nm = nm[ok]
        o = np.argsort(nm)
        self.rep, self.rep_sid, self.rep_nm = rep[o], self.sid[rep[o]], nm[o]
        self.rep_of_sid = np.full(n, -1, np.int64)
        self.rep_of_sid[self.rep_sid] = self.rep
        print(f'  analog representatives: {len(self.rep):,} (instrument pref {prefer_instrument!r}: '
              f'{match.mean():.1%} of spectra match)', flush=True)

    def _instr_match(self):
        if self._pref is None:
            return np.zeros(len(self.sid), bool)
        u, inv = np.unique(np.asarray([x if isinstance(x, str) else '' for x in self.instrument], dtype=object),
                           return_inverse=True)
        um = np.array([x.lower().replace('-', '').replace(' ', '') == self._pref for x in u])
        return um[inv]

    def rep_excluding(self, sid, bad_spectra_mask_fn):
        """Best representative of `sid` among its spectra that survive a mask (same rule as build_rep)."""
        sp = self.spectra_of(sid)
        sp = sp[~bad_spectra_mask_fn(sp)]
        sp = sp[np.isfinite(self.nm[sp])]
        if len(sp) == 0:
            return -1
        o = np.lexsort((sp, -self.npk[sp].astype(np.int64), -self._match[sp].astype(np.int64)))
        return int(sp[o[0]])
