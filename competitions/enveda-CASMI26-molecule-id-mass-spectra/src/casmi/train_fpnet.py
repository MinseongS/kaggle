"""Leak-free retraining of the spectrum -> fingerprint model (FPNet), prvsiyan's recipe.

Recipe (analog-propagation notebook, appendix / TRAIN_SCRIPT_SRC):
  loss   = BCE(z, f_true) + lam * softmax-CE over [positive + K=63 decoys from the same +-10 ppm pool window],
           score = (f.z - mean_c f.z) / sqrt(nbits), computed in fp32
  optim  = AdamW(lr 3e-4, wd 0.01, betas (0.9, 0.98)), warmup 2k, cosine to 2% of lr, bs 256, fp16 autocast, clip 1
  aug    = peak dropout U(0, 0.3), intensity log-normal sigma 0.25, m/z +-5 ppm
  merge  = with prob merge_p the input is 2-4 spectra of the same structure merged (fp_merged_*: 0.6, fp_single_*: 0)
  select = keep only the best checkpoint by validation hard-negative top-1 (the model overfits after ~12-20k steps)
Changes vs the notebook:
  * ALL structures of the fixed held-out list (holdout_v1.json: every metric key used by CV) are removed from the
    training spectra AND from the decoy pool, so CV on those molecules no longer leaks through the FP channel.
  * validation = a fixed random 1.5% of the remaining metric keys (never the CV molecules), deterministic negatives;
    the selection score is the mean of all-val and timsTOF-val hard-negative top-1.
  * timsTOF spectra (the test instrument) are over-sampled to `tims_frac` of every batch.
  * adduct / CE / instrument / polarity conditioning is the architecture's global token (unchanged, fpnet.FPNet).

Commands
  uv run python -m casmi.train_fpnet holdout            # writes src/casmi/holdout_v1.json (needs data/cache)
  uv run python -m casmi.train_fpnet prep --out DIR     # spec.npz + pool.npz for training (CPU, ~5 min)
  python -m casmi.train_fpnet train --data DIR --out fp_merged_h1.pt --merge_p 0.6     # GPU
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

HOLDOUT_FILE = Path(__file__).with_name('holdout_v1.json')


# ============================================================================================ held-out list
def make_holdout(n_extra_e180=500, val_frac=0.015, seed=1):
    """cv   = the exp000 CV molecules (250 np-examples = all of them, 250 enveda-180), copied from
              outputs/cv/exp000/feats.pkl (cv.select_holdout(L, 250, 250, seed=0) at the time of exp000);
       extra= n_extra_e180 more enveda-180 molecules (never shown to FPNet; available for a bigger CV);
       fpval= metric keys used only for FPNet checkpoint selection."""
    from . import config
    from .cv import select_holdout
    from .library import Library
    cfg = config.CFG()
    L = Library(cfg)
    feats = config.OUTPUTS / 'cv' / 'exp000' / 'feats.pkl'
    if feats.exists():   # the exp000 records verbatim (query spectra included) -> exp000 features stay comparable
        cv = [r['h'] for r in pickle.load(open(feats, 'rb'))]
        print(f'cv part = exp000 holdout from {feats}')
    else:
        cv = select_holdout(L, 250, 250, 0)
    used = {h['mkey'] for h in cv}
    extra = select_holdout(L, 0, n_extra_e180, seed, exclude_mkeys=used)
    used |= {h['mkey'] for h in extra}
    # every structure's metric key that is not held out -> sample the FPNet validation keys
    rng = np.random.default_rng(seed + 1)
    rest = sorted({k for k in L.s_mkey if k is not None and k not in used})
    fpval = sorted(rng.choice(rest, size=int(len(rest) * val_frac), replace=False).tolist())
    out = dict(version=1, note='cv = exp000 CV holdout; extra = more enveda-180 held out of FPNet training; '
                               'held-out mkeys = cv + extra (excluded from FPNet spectra and decoy pool); '
                               'fpval = FPNet checkpoint-selection keys',
               cv=cv, extra=extra, fpval=fpval)
    HOLDOUT_FILE.write_text(json.dumps(out))
    print(f'wrote {HOLDOUT_FILE}: cv={len(cv)} extra={len(extra)} fpval={len(fpval)}')


def load_holdout():
    return json.loads(HOLDOUT_FILE.read_text())


def held_mkeys(H=None):
    H = H or load_holdout()
    return {h['mkey'] for h in H['cv']} | {h['mkey'] for h in H['extra']}


# ============================================================================================ data prep (CPU)
def prep(out_dir, workers=8):
    import polars as pl
    import pyarrow.parquet as pq
    from multiprocessing import Pool as MPool

    from . import config
    from .fpnet import ADDUCT_IX, instr_family
    from .metric import score_keys_parallel
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    H = load_holdout()
    held = held_mkeys(H); fpval = set(H['fpval'])
    st = pl.read_parquet(config.CACHE / 'structs.parquet')
    s_key = st['inchikey14'].to_numpy(); s_mkey = st['mkey'].to_numpy()
    s_held = np.array([k in held for k in s_mkey]); s_val = np.array([k in fpval for k in s_mkey])
    print(f'structures {len(st):,}: held {s_held.sum()} val {s_val.sum()}', flush=True)

    # ---- decoy/target pool = COCONUT U train structures (same as candidates.Pool), minus held-out metric keys
    cm = pickle.load(open(config.COCO_DIR / 'coco_meta.pkl', 'rb'))
    tm = pl.read_parquet(config.CACHE / 'pool_train_meta.parquet')
    fp = np.vstack([np.load(config.COCO_DIR / 'coco_fp.npy'), np.load(config.CACHE / 'pool_train_fp.npy')])
    mass = np.concatenate([np.load(config.COCO_DIR / 'coco_mass.npy'), tm['mass'].to_numpy()])
    keys = np.concatenate([np.asarray(cm['keys'], dtype=object), tm['key'].to_numpy().astype(object)])
    smis = np.concatenate([np.asarray(cm['smiles'], dtype=object), tm['smiles'].to_numpy().astype(object)])
    good = np.isfinite(mass)
    held_keys14 = set(s_key[s_held].tolist())
    drop = np.array([k in held_keys14 for k in keys])
    # COCONUT tautomer variants of held-out molecules: metric-key the COCONUT entries near a held-out mass
    hm = np.sort(mass[drop & good])
    near = np.zeros(len(mass), bool)
    if len(hm):
        j = np.clip(np.searchsorted(hm, mass), 1, len(hm) - 1)
        d = np.minimum(np.abs(mass - hm[j - 1]), np.abs(mass - hm[j]))
        near = good & ~drop & (d <= 1e-5 * mass)
    idx = np.where(near)[0]
    mk = score_keys_parallel(smis[idx].tolist(), workers=workers)
    extra_drop = idx[np.array([k in held for k in mk], bool)] if len(idx) else idx
    drop[extra_drop] = True
    print(f'pool: dropped {drop.sum()} held-out entries ({len(extra_drop)} via metric key among {len(idx)} '
          f'same-mass) ({time.time()-t0:.0f}s)', flush=True)
    keep = good & ~drop
    fp, mass, keys = fp[keep], mass[keep], keys[keep]
    o = np.argsort(mass, kind='mergesort'); fp, mass, keys = fp[o], mass[o], keys[o]
    k2p = {k: i for i, k in enumerate(keys)}
    s2p = np.array([k2p.get(k, -1) for k in s_key], np.int64)
    s2p[s_held] = -1
    np.save(out / 'pool_fp.npy', np.ascontiguousarray(fp)); np.save(out / 'pool_mass.npy', mass)
    print(f'pool {len(mass):,} x {fp.shape[1]} bytes; train structures with fp {int((s2p >= 0).sum()):,}', flush=True)

    # ---- spectra (row order == library cache order)
    meta = pl.read_parquet(config.CACHE / 'lib_meta.parquet', columns=['sid', 'instrument', 'adduct', 'mode', 'ce'])
    sid = meta['sid'].to_numpy()
    pf = pq.ParquetFile(config.TRAIN)
    offs, mzs, its, keep_rows = [np.zeros(1, np.int64)], [], [], []
    base, row0 = 0, 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz'])
        n = t.num_rows
        rows = np.arange(row0, row0 + n); row0 += n
        ok = s2p[sid[rows]] >= 0
        args = [(r['ms2_mzs'], r['ms2_normalized_intensities'], r['precursor_mz'])
                for r, k in zip(t.to_pylist(), ok) if k]
        with MPool(workers) as p:
            res = p.starmap(_prep_one, args, chunksize=2000)
        lens = np.array([len(a) for a, _ in res], np.int64)
        good_r = lens > 0
        kr = rows[ok][good_r]
        keep_rows.append(kr)
        mzs.append(np.concatenate([a for a, _ in res if len(a)]) if good_r.any() else np.zeros(0, np.float32))
        its.append(np.concatenate([b for _, b in res if len(b)]).astype(np.float16) if good_r.any() else np.zeros(0, np.float16))
        offs.append(base + np.cumsum(lens[good_r])); base += int(lens.sum())
        print(f'  rg {rg+1}/{pf.num_row_groups}: kept {len(kr):,}/{n:,} ({time.time()-t0:.0f}s)', flush=True)
    rows = np.concatenate(keep_rows)
    m = meta[rows]
    prec = pl.read_parquet(config.TRAIN, columns=['precursor_mz'])['precursor_mz'].to_numpy()[rows]
    ins_u = {}
    ins = np.array([ins_u.setdefault(x, instr_family(x)) for x in m['instrument'].to_list()], np.int8)
    ad = np.array([ADDUCT_IX.get(a, ADDUCT_IX['<unk>']) for a in m['adduct'].to_list()], np.int8)
    mode = np.where(m['mode'].to_numpy() == 'positive', 1.0, -1.0).astype(np.float32)
    ce = m['ce'].to_numpy().astype(np.float32)
    ssid = sid[rows]
    np.savez(out / 'spec.npz', off=np.concatenate(offs), mz=np.concatenate(mzs), it=np.concatenate(its),
             prec=prec.astype(np.float32), ad=ad, ins=ins, ce=ce, mode=mode, pidx=s2p[ssid].astype(np.int32),
             is_val=s_val[ssid], row=rows.astype(np.int64))
    print(f'spec.npz: {len(rows):,} spectra ({(ins == 0).sum():,} timsTOF, {s_val[ssid].sum():,} val), '
          f'{base:,} peaks ({time.time()-t0:.0f}s)', flush=True)
    (out / 'prep_meta.json').write_text(json.dumps(dict(nbits=int(cm['nbits']), n_spec=int(len(rows)),
                                                         n_pool=int(len(mass)), n_held=len(held))))


def _prep_one(mz, it, prec):
    from .fpnet import prep_peaks
    return prep_peaks(mz, it, float(prec))


# ============================================================================================ training (GPU)
class Spec:
    def __init__(self, d):
        z = np.load(Path(d) / 'spec.npz')
        for k in ('off', 'mz', 'it', 'prec', 'ad', 'ins', 'ce', 'mode', 'pidx', 'is_val'):
            setattr(self, k, z[k])
        self.mass = np.load(Path(d) / 'pool_mass.npy')
        o = np.argsort(self.pidx, kind='mergesort')
        self.g_order = o
        self.g_bounds = np.searchsorted(self.pidx[o], np.arange(len(self.mass) + 1))

    def peers(self, i, rng, kmax=4):
        p = self.pidx[i]; a, b = self.g_bounds[p], self.g_bounds[p + 1]
        if b - a <= 1: return [i]
        k = int(rng.integers(2, min(kmax, b - a) + 1))
        pick = rng.choice(self.g_order[a:b], size=k, replace=False)
        if i not in pick: pick = np.concatenate([[i], pick[:-1]])
        return list(pick)

    def merged(self, idxs, maxlen):
        mz = np.concatenate([self.mz[self.off[j]:min(self.off[j] + maxlen, self.off[j + 1])] for j in idxs])
        it = np.concatenate([self.it[self.off[j]:min(self.off[j] + maxlen, self.off[j + 1])] for j in idxs]).astype(np.float32)
        o = np.argsort(mz, kind='stable'); mz, it = mz[o], it[o]
        # near-duplicate m/z: drop the weaker of each close adjacent pair (== the notebook's sequential loop)
        close = np.diff(mz) < 0.005
        keep = np.ones(len(mz), bool)
        j = np.where(close)[0] + 1
        w = it[j] >= it[j - 1]
        keep[j[w] - 1] = False; keep[j[~w]] = False
        mz, it = mz[keep], it[keep]
        if len(mz) > maxlen:
            top = np.sort(np.argsort(-it)[:maxlen]); mz, it = mz[top], it[top]
        return mz, it

    def negs(self, pos, K, rng, ppm=10.0):
        m = self.mass[pos]; tol = m * ppm / 1e6
        lo = np.searchsorted(self.mass, m - tol, 'left'); hi = np.searchsorted(self.mass, m + tol, 'right')
        n = hi - lo
        c = lo[:, None] + np.floor(rng.random((len(pos), K)) * np.maximum(n, 1)[:, None]).astype(np.int64)
        bad = c == pos[:, None]
        c = np.where(bad, np.where(c + 1 < hi[:, None], c + 1, lo[:, None]), c)
        few = n <= 1
        if few.any(): c[few] = rng.integers(0, len(self.mass), (few.sum(), K))
        return c

    def batch(self, ids, K, rng, aug, merge_p, max_peaks=128):
        B = len(ids)
        lens = np.minimum(self.off[ids + 1] - self.off[ids], max_peaks); N = max(int(lens.max()), 1)
        mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)
        for r, (i, l) in enumerate(zip(ids, lens)):
            a = self.off[i]; mz[r, :l] = self.mz[a:a + l]; it[r, :l] = self.it[a:a + l]; pad[r, :l] = False
        if merge_p > 0:
            for r, i in enumerate(ids):
                if rng.random() >= merge_p: continue
                pk = self.peers(i, rng)
                if len(pk) < 2: continue
                m2, i2 = self.merged(pk, N)
                mz[r] = 0; it[r] = 0; pad[r] = True
                mz[r, :len(m2)] = m2; it[r, :len(m2)] = i2; pad[r, :len(m2)] = False
        if aug:
            keep = rng.random((B, N)) > rng.uniform(0.0, 0.30, size=(B, 1))
            pad = pad | ~keep
            it = it * np.exp(rng.normal(0.0, 0.25, size=(B, N))).astype(np.float32)
            mz = mz * (1.0 + rng.normal(0.0, 5e-6, size=(B, N))).astype(np.float32)
            allpad = pad.all(1)
            if allpad.any(): pad[allpad, 0] = False
        ce = np.where(self.ce[ids] < 0, 25.0, self.ce[ids]).astype(np.float32)
        pos = self.pidx[ids].astype(np.int64)
        cand = np.concatenate([pos[:, None], self.negs(pos, K, rng)], 1) if K > 0 else pos[:, None]
        return (mz, it, pad, self.prec[ids].astype(np.float32), self.ad[ids].astype(np.int64),
                self.ins[ids].astype(np.int64), ce, self.mode[ids].astype(np.float32), cand)


def _train_stream(data_dir, seed, bs, K, merge_p, tims_frac):
    import torch
    wi = torch.utils.data.get_worker_info()
    wid = wi.id if wi else 0
    D = Spec(data_dir)
    rng = np.random.default_rng([seed, wid])
    tr = np.where(~D.is_val)[0]
    tims = tr[D.ins[tr] == 0]; other = tr[D.ins[tr] != 0]
    while True:
        nt = rng.binomial(bs, tims_frac) if tims_frac >= 0 else None
        if nt is None:
            ids = rng.choice(tr, size=bs, replace=False)
        else:
            ids = np.concatenate([rng.choice(tims, nt, replace=False), rng.choice(other, bs - nt, replace=False)])
        yield D.batch(ids, K, rng, True, merge_p)


class StreamDS(torch.utils.data.IterableDataset):
    def __init__(self, *args): self.args = args

    def __iter__(self): return _train_stream(*self.args)


def train(a):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from .fpnet import FPNet

    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    meta = json.loads((Path(a.data) / 'prep_meta.json').read_text()); nbits = meta['nbits']
    PFP = torch.from_numpy(np.load(Path(a.data) / 'pool_fp.npy')).to(dev)       # packed (n_pool, ceil(nbits/8))
    shifts = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)

    def unpack(idx):  # (B, C) -> (B, C, nbits) float, np.packbits big-endian order
        p = PFP[idx]
        return ((p.unsqueeze(-1) >> shifts) & 1).flatten(-2)[..., :nbits].float()

    D = Spec(a.data)
    va_all = np.where(D.is_val)[0]
    vr = np.random.default_rng(12345)
    va_t = va_all[D.ins[va_all] == 0]; va_o = va_all[D.ins[va_all] != 0]
    VAL = dict(all=np.sort(vr.choice(va_all, min(a.n_val, len(va_all)), replace=False)),
               tims=np.sort(vr.choice(va_t, min(a.n_val, len(va_t)), replace=False)))
    print(f'device {dev}; nbits {nbits}; spectra {len(D.pidx):,} (val {len(va_all):,}, val timsTOF {len(va_t):,}); '
          f'pool {len(D.mass):,}', flush=True)

    torch.manual_seed(a.seed)
    model = FPNet(nbits, d=a.d, layers=a.layers).to(dev)
    print('params %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler('cuda') if dev == 'cuda' else None

    def lr_at(s):
        if s < a.warm: return a.lr * s / max(1, a.warm)
        p = (s - a.warm) / max(1, a.steps - a.warm)
        return a.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    T = lambda x: torch.as_tensor(x, device=dev)

    def fwd(b):
        inp = tuple(T(x) for x in b[:8])
        ctx = torch.autocast('cuda', dtype=torch.float16) if dev == 'cuda' else torch.autocast('cpu', enabled=False)
        with ctx:
            z = model(*inp)
        z = z.float()
        cand = T(b[8])
        ypos = unpack(cand[:, 0])
        FPc = unpack(cand)
        raw = torch.bmm(FPc, z.unsqueeze(-1)).squeeze(-1)
        sc = (raw - raw.mean(1, keepdim=True)) / math.sqrt(nbits)
        return z, ypos, sc

    def evaluate():
        model.eval(); out = {}
        with torch.no_grad():
            for name, ids in VAL.items():
                rng = np.random.default_rng(777)          # identical negatives / merges every evaluation
                acc, bce = [], []
                for s in range(0, len(ids), 128):
                    b = D.batch(ids[s:s + 128], a.K, rng, False, a.merge_p)
                    z, ypos, sc = fwd(b)
                    bce.append(F.binary_cross_entropy_with_logits(z, ypos).item())
                    acc.append((sc.argmax(1) == 0).float().sum().item())
                out[name] = (float(np.sum(acc) / len(ids)), float(np.mean(bce)))
        model.train()
        return out

    def save(step, path, val):
        torch.save({'model': model.state_dict(), 'step': step, 'nbits': nbits, 'd': a.d, 'layers': a.layers,
                    'val': val, 'args': vars(a)}, path)

    dl = DataLoader(StreamDS(a.data, a.seed, a.bs, a.K, a.merge_p, a.tims_frac), batch_size=None, num_workers=a.workers, persistent_workers=a.workers > 0,
                    prefetch_factor=4 if a.workers > 0 else None)
    it_dl = iter(dl)
    t0 = time.time(); rb = rc = racc = 0.0; nr = 0; best = -1.0; hist = []
    for step in range(a.steps):
        for g in opt.param_groups: g['lr'] = lr_at(step)
        b = next(it_dl)
        b = tuple(x.numpy() if hasattr(x, 'numpy') else x for x in b)
        z, ypos, sc = fwd(b)
        lb = F.binary_cross_entropy_with_logits(z, ypos)
        lc = F.cross_entropy(sc, torch.zeros(len(sc), dtype=torch.long, device=dev))
        loss = lb + a.lam * lc
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print(f'  non-finite loss at step {step}; skipping', flush=True); continue
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        rb += lb.item(); rc += lc.item(); racc += (sc.argmax(1) == 0).float().mean().item(); nr += 1
        if step % 200 == 0:
            el = time.time() - t0
            print(f'step {step} bce {rb/nr:.4f} ctr {rc/nr:.4f} top1 {racc/nr:.3f} lr {lr_at(step):.2e} '
                  f'{el:.0f}s {(step+1)/max(el,1):.2f} it/s', flush=True)
            rb = rc = racc = 0.0; nr = 0
        last = (step + 1 == a.steps) or (time.time() - t0) / 60 > a.max_minutes
        if (step + 1) % a.val_every == 0 or last:
            v = evaluate()
            score = 0.5 * (v['all'][0] + v['tims'][0])
            hist.append(dict(step=step + 1, score=score, **{k: x[0] for k, x in v.items()}))
            print(f'  VAL step {step+1} top1 all {v["all"][0]:.4f} tims {v["tims"][0]:.4f} bce {v["all"][1]:.4f} '
                  f'score {score:.4f}' + ('  <- best' if score > best else ''), flush=True)
            if score > best:
                best = score; save(step + 1, a.out, v)
            Path(a.out).with_suffix('.hist.json').write_text(json.dumps(hist, indent=0))
        if last and (time.time() - t0) / 60 > a.max_minutes:
            print('time budget reached', flush=True); break
    print(f'done. best score {best:.4f}; {(time.time()-t0)/60:.1f} min', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser('train_fpnet')
    sub = ap.add_subparsers(dest='cmd', required=True)
    h = sub.add_parser('holdout'); h.add_argument('--n-extra-e180', type=int, default=500)
    p = sub.add_parser('prep'); p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=8)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True); t.add_argument('--out', default='fp_merged_h1.pt')
    t.add_argument('--steps', type=int, default=30000); t.add_argument('--bs', type=int, default=256)
    t.add_argument('--lr', type=float, default=3e-4); t.add_argument('--d', type=int, default=512)
    t.add_argument('--layers', type=int, default=6); t.add_argument('--K', type=int, default=63)
    t.add_argument('--lam', type=float, default=1.0); t.add_argument('--warm', type=int, default=2000)
    t.add_argument('--val_every', type=int, default=1000); t.add_argument('--n_val', type=int, default=4096)
    t.add_argument('--max_minutes', type=float, default=1e9); t.add_argument('--seed', type=int, default=7)
    t.add_argument('--merge_p', type=float, default=0.6)
    t.add_argument('--tims_frac', type=float, default=0.6, help='-1 = uniform sampling (the notebook)')
    t.add_argument('--workers', type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == 'holdout':
        make_holdout(a.n_extra_e180)
    elif a.cmd == 'prep':
        prep(a.out, a.workers)
    else:
        train(a)


if __name__ == '__main__':
    main()
