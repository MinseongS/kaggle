"""Spectrum -> 6,930-bit fingerprint Transformer (CSI:FingerID-style), ranked by f.z. Verbatim port of the notebook."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_PEAKS = 128
ADDUCT_LIST = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+",
               "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",
               "[M+2H]2+", "[M-2H]-", "[2M+H]+", "[2M+Na]+", "[2M+NH4]+", "[2M-H]-", "[2M+K]+",
               "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[M+Na-2H]-", "[M-H2O]+", "<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}
INSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]


def instr_family(s):
    if s is None:
        return 4
    t = str(s).lower()
    if 'timstof' in t: return 0
    if 'orbitrap' in t or 'qft' in t or 'ftms' in t or 'hybrid ft' in t or 'itft' in t or 'exactive' in t: return 1
    if 'tof' in t: return 2
    if 'trap' in t or 'qq' in t: return 3
    return 4


def prep_peaks(mz, inten, prec_mz, max_peaks=MAX_PEAKS, floor=1e-3, win=50.0, per_win=8):
    """Filter -> window-diversified top-N -> sort by m/z. Returns (mz, sqrt-intensity)."""
    mz = np.asarray(mz, np.float64); it = np.asarray(inten, np.float64)
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = (mz <= prec_mz + 1.5)
    mz, it = mz[keep], it[keep]
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mx = it.max()
    if mx <= 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = it >= floor * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        cnt = {}; sel = []
        for i in order:
            b = bucket[i]; c = cnt.get(b, 0)
            if c < per_win: cnt[b] = c + 1; sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            ss = set(sel.tolist())
            rest = np.array([i for i in order if i not in ss])
            need = max_peaks - len(sel)
            if len(rest): sel = np.concatenate([sel, rest[:need]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    v = np.sqrt(it / it.max())
    return mz.astype(np.float32), v.astype(np.float32)


class SinEmb(nn.Module):
    def __init__(self, dim, lo=-2.0, hi=3.2, power=1.0):
        super().__init__()
        n = dim // 2
        wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)
        self.register_buffer('inv', (2 * math.pi) / wav)

    def forward(self, x):
        a = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__(); self.h = h
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, pad):
        B, N, D = x.shape; y = self.n1(x)
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        m = (~pad)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))


class FPNet(nn.Module):
    def __init__(self, nbits, d=512, layers=6, heads=8, drop=0.1):
        super().__init__()
        self.d = d
        self.mz_emb = SinEmb(d); self.nl_emb = SinEmb(d)
        self.pk = nn.Linear(2 * d + 1, d)
        self.prec_emb = SinEmb(d)
        self.ad = nn.Embedding(len(ADDUCT_LIST), d)
        self.ins = nn.Embedding(len(INSTR_LIST), d)
        self.gl = nn.Linear(d + 3, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))

    def forward(self, mz, it, pad, prec, ad, ins, ce, mode):
        B, N = mz.shape
        nl = (prec[:, None] - mz).clamp(min=0)
        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
        g = self.gl(torch.cat([self.prec_emb(prec), (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),
                               torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for b in self.blocks: x = b(x, pad)
        x = self.norm(x)
        cls = x[:, 0]
        msk = (~pad[:, 1:]).float().unsqueeze(-1)
        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
        return self.head(torch.cat([cls, mean], -1))


class ModelBank:
    """fp_merged_* models get one merged peak list; fp_single_* get each spectrum. Final z = mean of the two views."""

    def __init__(self, paths, device='cpu'):
        self.dev = device
        self.single, self.merged = [], []
        for pth in sorted(paths):
            ck = torch.load(pth, map_location='cpu', weights_only=False)
            net = FPNet(ck['nbits'], d=ck['d'], layers=ck['layers']).to(device).eval()
            net.load_state_dict(ck['model'])
            (self.merged if 'merged' in str(pth).split('/')[-1] else self.single).append(net)
            self.nbits = ck['nbits']
            print(f'  loaded {str(pth).split("/")[-1]}: d={ck["d"]} layers={ck["layers"]} step={ck.get("step")}')
        torch.set_num_threads(max(1, torch.get_num_threads()))

    @torch.no_grad()
    def logits(self, spectra):
        """spectra: list of dict(mz, it, prec, adduct, instrument, ce(list|float|None), mode(+1/-1)).
        Exactly the notebook's model_logits(sub)."""
        out = []
        if self.single:
            za = self._logits_from(spectra, self.single)
            if za is not None: out.append(za)
        if self.merged:
            mz, it = merge_peaks(spectra)
            r0 = spectra[0]
            zb = self._logits_raw([(mz, it)], self.merged, float(np.median([s['prec'] for s in spectra])),
                                  r0['adduct'], r0['instrument'], 25.0, float(np.mean([s['mode'] for s in spectra])))
            if zb is not None: out.append(zb)
        return np.mean(out, axis=0).astype(np.float32) if out else None

    def _run(self, nets, mz, it, pad, prec, ad, ins, ce, mode):
        T = lambda x: torch.as_tensor(x, device=self.dev)
        args = (T(mz), T(it), T(pad), T(prec), T(ad), T(ins), T(ce), T(mode))
        return np.mean([n(*args).float().mean(0).cpu().numpy() for n in nets], axis=0)

    def _logits_from(self, spectra, nets):
        P, rows = [], []
        for r in spectra:
            a, b = prep_peaks(r['mz'], r['it'], float(r['prec']))
            if len(a): P.append((a, b)); rows.append(r)
        if not P: return None
        B = len(P); N = max(len(a) for a, _ in P)
        mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)
        for i, (a, b) in enumerate(P):
            mz[i, :len(a)] = a; it[i, :len(b)] = b; pad[i, :len(a)] = False

        def ce_of(r):
            v = r.get('ce')
            try:
                return float(np.mean(np.atleast_1d(v))) if v is not None and len(np.atleast_1d(v)) else 25.0
            except Exception:
                return 25.0
        return self._run(nets, mz, it, pad,
                         np.array([float(r['prec']) for r in rows], np.float32),
                         np.array([ADDUCT_IX.get(r['adduct'], ADDUCT_IX['<unk>']) for r in rows]),
                         np.array([instr_family(r['instrument']) for r in rows]),
                         np.array([ce_of(r) for r in rows], np.float32),
                         np.array([float(r['mode']) for r in rows], np.float32))

    def _logits_raw(self, pairs, nets, prec, adduct, instrument, ce, mode):
        P = [prep_peaks(mz, it, prec) for mz, it in pairs]
        P = [(a, b) for a, b in P if len(a)]
        if not P: return None
        B = len(P); N = max(len(a) for a, _ in P)
        mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)
        for i, (a, b) in enumerate(P):
            mz[i, :len(a)] = a; it[i, :len(b)] = b; pad[i, :len(a)] = False
        return self._run(nets, mz, it, pad, np.full(B, prec, np.float32),
                         np.full(B, ADDUCT_IX.get(adduct, ADDUCT_IX['<unk>'])), np.full(B, instr_family(instrument)),
                         np.full(B, ce, np.float32), np.full(B, mode, np.float32))


def merge_peaks(spectra):
    """All of a molecule's peaks collapsed into one pseudo-spectrum (near-duplicate m/z merged, stronger kept)."""
    mz = np.concatenate([np.asarray(s['mz'], float) for s in spectra])
    it = np.concatenate([np.asarray(s['it'], float) / max(float(np.asarray(s['it'], float).max()), 1e-9)
                         if len(s['it']) else np.zeros(0) for s in spectra])
    o = np.argsort(mz); mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    for j in range(1, len(mz)):
        if mz[j] - mz[j - 1] < 0.005:
            if it[j] >= it[j - 1]: keep[j - 1] = False
            else: keep[j] = False
    return mz[keep], it[keep]
