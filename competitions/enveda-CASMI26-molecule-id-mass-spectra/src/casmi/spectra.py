"""Similarity kernels: entropy-weighted spectral entropy similarity (Li et al. 2021) + mass-shifted variant.

Verbatim numba port of the notebook kernels. One change for speed that preserves results exactly:
library spectra are cleaned ONCE at cache-build time (clean_batch) instead of on every comparison,
so `search*` here take pre-cleaned library arrays.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _clean(mz, it, floor, topk, power, ent_weight):
    n = len(mz)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx:
            mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    c = 0
    for i in range(n):
        if it[i] >= thr:
            c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if it[i] >= thr:
            idx[j] = i
            j += 1
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c):
            v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk):
            k2[i] = idx[o[i]]
        k2.sort()
        idx = k2
        c = topk
    om = np.empty(c, np.float32)
    oi = np.empty(c, np.float32)
    s = 0.0
    for i in range(c):
        om[i] = mz[idx[i]]
        v = it[idx[i]] ** power
        oi[i] = v
        s += v
    if s > 0:
        for i in range(c):
            oi[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(c):
            if oi[i] > 0:
                S -= oi[i] * np.log(oi[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S
            s2 = 0.0
            for i in range(c):
                oi[i] = oi[i] ** w
                s2 += oi[i]
            if s2 > 0:
                for i in range(c):
                    oi[i] /= s2
    return om, oi


@njit(cache=True)
def _clean_counts(off, mz, it, floor, topk, power, ent_weight):
    n = len(off) - 1
    cnt = np.zeros(n, np.int64)
    for k in range(n):
        a = off[k]; b = off[k + 1]
        if b > a:
            cm, cp = _clean(mz[a:b], it[a:b], floor, topk, power, ent_weight)
            cnt[k] = len(cm)
    return cnt


@njit(cache=True, parallel=True)
def _clean_fill(off, mz, it, floor, topk, power, ent_weight, noff, omz, oit):
    n = len(off) - 1
    for k in prange(n):
        a = off[k]; b = off[k + 1]
        if b > a:
            cm, cp = _clean(mz[a:b], it[a:b], floor, topk, power, ent_weight)
            o = noff[k]
            for i in range(len(cm)):
                omz[o + i] = cm[i]
                oit[o + i] = cp[i]


def clean_batch(off, mz, it, floor, topk, power, ent_weight):
    """Clean a CSR batch of spectra. Returns (new_off int64, mz float32, it float32)."""
    mz = np.ascontiguousarray(mz, np.float32); it = np.ascontiguousarray(it, np.float32)
    off = np.ascontiguousarray(off, np.int64)
    cnt = _clean_counts_par(off, mz, it, floor, topk, power, ent_weight)
    noff = np.zeros(len(cnt) + 1, np.int64)
    np.cumsum(cnt, out=noff[1:])
    omz = np.empty(noff[-1], np.float32); oit = np.empty(noff[-1], np.float32)
    _clean_fill(off, mz, it, floor, topk, power, ent_weight, noff, omz, oit)
    return noff, omz, oit


@njit(cache=True, parallel=True)
def _clean_counts_par(off, mz, it, floor, topk, power, ent_weight):
    n = len(off) - 1
    cnt = np.zeros(n, np.int64)
    for k in prange(n):
        a = off[k]; b = off[k + 1]
        if b > a:
            cm, cp = _clean(mz[a:b], it[a:b], floor, topk, power, ent_weight)
            cnt[k] = len(cm)
    return cnt


@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol):
    i = 0; j = 0; n = len(qmz); m = len(cmz)
    SA = 0.0
    for x in range(n):
        if qp[x] > 0:
            SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0:
            SB -= cp[x] * np.log(cp[x])
    SAB = 0.0; tot = 0.0
    buf = np.empty(n + m, np.float64); b = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol:
            buf[b] = qp[i]; i += 1; b += 1
        elif d > tol:
            buf[b] = cp[j]; j += 1; b += 1
        else:
            buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1
    while i < n:
        buf[b] = qp[i]; i += 1; b += 1
    while j < m:
        buf[b] = cp[j]; j += 1; b += 1
    for x in range(b):
        tot += buf[x]
    if tot <= 0:
        return 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0:
            SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)


@njit(cache=True, fastmath=True)
def entropy_sim_shift(qmz, qp, cmz, cp, tol, shift):
    a = entropy_sim(qmz, qp, cmz, cp, tol)
    if shift > -0.001 and shift < 0.001:
        return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)):
        sm[i] = cmz[i] + shift
    b = entropy_sim(qmz, qp, sm, cp, tol)
    return a if a > b else b


@njit(cache=True, fastmath=True, parallel=True)
def search_clean(qmz, qp, cand, off, allmz, allin, tol):
    """Entropy similarity of one cleaned query vs pre-cleaned library spectra `cand`."""
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]; a = off[c]; b = off[c + 1]
        if b <= a:
            continue
        out[k] = entropy_sim(qmz, qp, allmz[a:b], allin[a:b], tol)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def search_shift_clean(qmz, qp, cand, off, allmz, allin, tol, shift):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]; a = off[c]; b = off[c + 1]
        if b <= a:
            continue
        out[k] = entropy_sim_shift(qmz, qp, allmz[a:b], allin[a:b], tol, shift[k])
    return out


def clean(mz, it, cfg):
    return _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                  cfg.INT_FLOOR, cfg.MAX_PEAKS, cfg.INT_POWER, cfg.ENT_WEIGHT)
