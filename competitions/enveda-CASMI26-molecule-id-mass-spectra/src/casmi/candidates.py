"""Candidate pool = COCONUT (attached, precomputed fps) U train structures (fingerprinted here), InChIKey14-deduped,
sorted by exact mass. Optional ChEBI+LIPID MAPS (CFG.USE_BIO_DB, off by default as in the notebook).

Cache: data/cache/pool_train_fp.npy / pool_train_meta.parquet (train-structure half, ~5 min once).
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
import time

import numpy as np
import polars as pl

from . import chem, config

C = config.CACHE
_BITS = None


def _init_bits(bits_path):
    global _BITS
    _BITS = np.load(bits_path)


def _fp_and_mass(smi):
    r = chem.raw_fp(smi)
    if r is None:
        return None
    return r[0][_BITS], r[1]


def build_train_fp_cache(cfg: config.CFG, force=False):
    fpp, mp_ = C / 'pool_train_fp.npy', C / 'pool_train_meta.parquet'
    if fpp.exists() and mp_.exists() and not force:
        print('train-structure fingerprint cache exists'); return
    t0 = time.time()
    cm = pickle.load(open(config.COCO_DIR / 'coco_meta.pkl', 'rb'))
    co = [str(k) for k in cm['keys']]
    st = pl.read_parquet(C / 'structs.parquet')
    tr = st.filter(~pl.col('inchikey14').is_in(co))
    smis = tr['normalized_smiles'].to_list()
    print(f'fingerprinting {len(smis):,} training structures not in COCONUT', flush=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(cfg.WORKERS, initializer=_init_bits, initargs=(str(config.COCO_DIR / 'fp_bits.npy'),)) as p:
        res = p.map(_fp_and_mass, smis, chunksize=500)
    ok = [i for i, r in enumerate(res) if r is not None]
    fp = np.packbits(np.stack([res[i][0] for i in ok]), axis=1)
    mass = np.array([res[i][1] for i in ok])
    np.save(fpp, fp)
    pl.DataFrame(dict(key=tr['inchikey14'].to_numpy()[ok], smiles=np.asarray(smis, dtype=object)[ok].tolist(),
                      mass=mass)).write_parquet(mp_)
    print(f'train fp cache: {len(ok):,} ok / {len(smis):,} ({time.time()-t0:.0f}s)', flush=True)


class Pool:
    """Candidates + packed fingerprints, sorted by exact mass."""

    def __init__(self, cfg: config.CFG):
        t0 = time.time()
        cm = pickle.load(open(config.COCO_DIR / 'coco_meta.pkl', 'rb'))
        fp = [np.load(config.COCO_DIR / 'coco_fp.npy')]
        mass = [np.load(config.COCO_DIR / 'coco_mass.npy')]
        keys = [np.asarray(cm['keys'], dtype=object)]
        smis = [np.asarray(cm['smiles'], dtype=object)]
        src = [np.zeros(len(mass[0]), np.int8)]
        if cfg.USE_BIO_DB:
            bm = pickle.load(open(config.BIO_DIR / 'bio_meta.pkl', 'rb'))
            fp.append(np.load(config.BIO_DIR / 'bio_fp.npy')); mass.append(np.load(config.BIO_DIR / 'bio_mass.npy'))
            keys.append(np.asarray(bm['keys'], dtype=object)); smis.append(np.asarray(bm['smiles'], dtype=object))
            src.append(np.full(len(mass[-1]), 2, np.int8))
        tm = pl.read_parquet(C / 'pool_train_meta.parquet')
        fp.append(np.load(C / 'pool_train_fp.npy')); mass.append(tm['mass'].to_numpy())
        keys.append(tm['key'].to_numpy().astype(object)); smis.append(tm['smiles'].to_numpy().astype(object))
        src.append(np.ones(len(tm), np.int8))
        fp = np.vstack(fp); mass = np.concatenate(mass); keys = np.concatenate(keys)
        smis = np.concatenate(smis); src = np.concatenate(src)
        good = np.isfinite(mass)
        fp, mass, keys, smis, src = fp[good], mass[good], keys[good], smis[good], src[good]
        o = np.argsort(mass)
        self._fp = fp[o]; self.mass = mass[o]; self.keys = keys[o]; self.smiles = smis[o]
        self.src = src[o]  # provenance: NEVER a ranker feature (leaks in the class-2 simulation)
        self.nbits = cm['nbits']
        self.k2i = {k: i for i, k in enumerate(self.keys)}
        print(f'pool: {len(self.mass):,} structures, {self.nbits} bits ({time.time()-t0:.0f}s)', flush=True)

    def window(self, t, ppm):
        a = np.searchsorted(self.mass, t * (1 - ppm / 1e6), 'left')
        b = np.searchsorted(self.mass, t * (1 + ppm / 1e6), 'right')
        return np.arange(a, b)

    def fps(self, idx):
        return np.unpackbits(np.asarray(self._fp[idx]), axis=1)[:, :self.nbits]
