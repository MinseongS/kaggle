"""CASMI26 kind-2 simulation: train molecules treated as 'no public spectrum, present in PubChem'.

For each selected molecule (selection.json, made by select.py):
  * out-of-fold FPNet logits: fold-safe model k for a molecule of identity fold k (single + merged-by-mode views, as the
    public PubChem channel does);
  * PubChem tier 10 ppm window around the precursor-derived neutral mass;
  * pass 1 = ECFP4 sub-bits . z on the whole window (public channel), pass 2 = full raw fingerprint . z on
    top-PASS1 by pass 1 + top-POPU by prior + every truth row;
  * every prior array of casmi26-pubchem-popularity-prior for the scored rows.
Writes /kaggle/working/sim2_rows.parquet (rows of pass 2 + window-level stats) and sim2_mols.parquet.
CPU only. Time-boxed: stops starting new molecules after TIME_BUDGET seconds.
"""
import os, subprocess, sys, importlib.metadata
from pathlib import Path
try:
    _v = importlib.metadata.version('rdkit')
except importlib.metadata.PackageNotFoundError:
    _v = ''
if _v.replace('.03.', '.3.') != '2026.3.3':
    _tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    _w = sorted(Path('/kaggle/input').rglob(f'rdkit-2026.3.3-{_tag}-*.whl'))
    assert _w, 'rdkit wheel missing for ' + _tag
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q', str(_w[0])])
import base64, glob, json, os, sys, time, zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

T0 = time.time()
TIME_BUDGET = float(os.environ.get('SIM2_BUDGET', 10.5 * 3600))
PPM = 10.0
PASS1 = 2000
POPU = 300
KEEP_FZ = 600
KEEP_POP = 100
MAX_SPEC = 8
NP_LIBS = {'riken', 'gnps', 'msdial', 'massbank', 'mona', 'spectraverse', 'enveda-np-examples'}
TEST_ADDUCTS = ['[M+H]+', '[M-H]-', '[M+CH2O2-H]-', '[M+Na]+', '[M+NH4]+', '[M+K]+', '[M+Cl]-']
SELECTION = json.loads(zlib.decompress(base64.b64decode('__SELECTION__')))


def find(pattern):
    hits = sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True))
    assert hits, pattern
    return hits[0]


CODE = os.path.dirname(os.path.dirname(find('casmi26-attributed-v4g-reproduction-assets/casmi26-v3-models/code/casmi/fpnet.py')))
FOLD_DIR = os.path.dirname(find('casmi26-fold-safe-fpnet/fpnet_fold0.pt'))
POP_DIR = os.path.dirname(find('casmi26-pubchem-popularity-prior/pc_lsid.npy'))
TIER_DIR = os.path.dirname(find('casmi26-pubchem-popularity-prior/casmi26-pubchem-tier/pc_mass.npy'))
TRAIN = find('enveda-CASMI26-molecule-id-mass-spectra/train.parquet')
sys.path.insert(0, CODE)
print('code', CODE, '\nfold', FOLD_DIR, '\npop', POP_DIR, '\ntier', TIER_DIR, '\ntrain', TRAIN, flush=True)

# ---------------------------------------------------------------- 1. spectra of the selected molecules
import pyarrow.parquet as pq
import pyarrow.dataset as ds

sel = pd.DataFrame(SELECTION)
want = set(sel.ik14)
cols = ['ingest_lib', 'inchikey14', 'adduct', 'ionization_mode', 'precursor_mz', 'ms2_mzs',
        'ms2_normalized_intensities', 'collision_energy_ev']
tb = ds.dataset(TRAIN).to_table(columns=cols, filter=ds.field('inchikey14').isin(list(want)))
sp = tb.to_pandas()
sp = sp[sp.ingest_lib.isin(NP_LIBS) & sp.adduct.isin(TEST_ADDUCTS)].reset_index(drop=True)
print('spectra', len(sp), 'molecules with spectra', sp.inchikey14.nunique(), f'{time.time()-T0:.0f}s', flush=True)

import torch
from casmi import chem, fpnet
from casmi.spectra import merge_spectra

torch.set_num_threads(4)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def views(g, mono):
    """Public-channel views for one molecule: single spectra of the chosen mode + one merged spectrum.
    Spectra whose labelled adduct does not reproduce the true mass within 10 ppm (library mislabels) are dropped,
    then one adduct is kept (most frequent, [M+H]+/[M-H]- first on ties) so the merged view is consistent."""
    nm = chem.neutral_mass(g.precursor_mz.values, g.adduct.values)
    g = g[np.abs(nm - mono) <= mono * 10e-6]
    if not len(g):
        return None
    pos = g[g.ionization_mode == 'positive']
    g = pos if len(pos) else g
    cnt = g.adduct.value_counts()
    best = sorted(cnt.index, key=lambda a: (-cnt[a], a not in ('[M+H]+', '[M-H]-')))[0]
    g = g[g.adduct == best].head(MAX_SPEC)
    md = 1 if g.ionization_mode.iloc[0] == 'positive' else -1
    items_s, raw = [], []
    for r in g.itertuples():
        mz = np.asarray(r.ms2_mzs, np.float64); it = np.asarray(r.ms2_normalized_intensities, np.float64)
        ce = r.collision_energy_ev
        cev = float(np.mean(ce)) if ce is not None and len(ce) else np.nan
        cen = len(ce) if ce is not None else 0
        pm, pi = fpnet.prep_peaks(mz, it, r.precursor_mz)
        items_s.append(dict(mz=pm, it=pi, prec=float(r.precursor_mz), adduct_ix=chem.adduct_index(r.adduct),
                            ce=0.0 if np.isnan(cev) else cev, ce_known=0.0 if np.isnan(cev) else 1.0,
                            n_merged=min(max(1, cen), 8), mode=float(md)))
        raw.append((mz, it, cev, cen, r.adduct, float(r.precursor_mz)))
    mm, ii = merge_spectra([(a[0], a[1]) for a in raw])
    prec = float(np.median([a[5] for a in raw]))
    ces = [a[2] for a in raw if not np.isnan(a[2])]
    ads = [a[4] for a in raw]; ad = max(ads, key=ads.count)
    pm, pi = fpnet.prep_peaks(mm, ii, prec)
    item_m = dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad), ce=float(np.mean(ces)) if ces else 0.0,
                  ce_known=1.0 if ces else 0.0, n_merged=min(sum(max(1, a[3]) for a in raw), 8), mode=float(md))
    target = float(chem.neutral_mass(np.array([prec]), np.array([ad]))[0])
    return items_s, item_m, target, ad, md, len(raw)


mols = []
groups = dict(tuple(sp.groupby('inchikey14')))
for r in sel.itertuples():
    g = groups.get(r.ik14)
    if g is None or not len(g):
        continue
    v = views(g, r.mass)
    if v is None:
        continue
    items_s, item_m, target, ad, md, n = v
    mols.append(dict(ik14=r.ik14, fold=int(r.fold), mass=float(r.mass), src=r.src, smiles=r.smiles, target=target,
                     adduct=ad, mode=md, n_spec=n, items_s=items_s, item_m=item_m))
print('molecules ready', len(mols), f'{time.time()-T0:.0f}s', flush=True)

Z = {}
with torch.no_grad():
    for k in range(5):
        model = fpnet.load_model(os.path.join(FOLD_DIR, f'fpnet_fold{k}.pt'), dev)
        mk = [m for m in mols if m['fold'] == k]
        for s in range(0, len(mk), 32):
            chunk = mk[s:s + 32]
            flat = [it for m in chunk for it in m['items_s']] + [m['item_m'] for m in chunk]
            z = model(*fpnet.collate(flat, dev))[0].float().cpu().numpy()
            pos = 0
            for m in chunk:
                n = len(m['items_s'])
                Z[m['ik14']] = [z[pos:pos + n].mean(0), None]
                pos += n
            for j, m in enumerate(chunk):
                Z[m['ik14']][1] = z[pos + j]
        del model
        print('fold', k, 'scored', len(mk), f'{time.time()-T0:.0f}s', flush=True)
for m in mols:
    zs, zm = Z[m['ik14']]
    m['z'] = (0.5 * (zs + zm)).astype(np.float32)
    del m['items_s'], m['item_m']

# ---------------------------------------------------------------- 2. PubChem windows (workers)
BITS = np.load(os.path.join(FOLD_DIR, 'fp_bits.npy'))
PRI = ['lsid', 'lpmid', 'lsynonyms', 'lmesh', 'lpatent', 'n_depositors', 'n_vendors', 'np', 'src_mask', 'lotus_refs',
       'formula_isomers', 'cid']
_W = {}


def init():
    sys.path.insert(0, CODE)
    from casmi import chem as c
    _W['chem'] = c
    r = c._rdkit()
    _W['Chem'] = r['Chem']; _W['gen'] = r['gen'].GetMorganGenerator(radius=2, fpSize=4096)
    _W['mass'] = np.load(os.path.join(TIER_DIR, 'pc_mass.npy'), mmap_mode='r')
    _W['off'] = np.load(os.path.join(TIER_DIR, 'pc_off.npy'), mmap_mode='r')
    _W['buf'] = np.load(os.path.join(TIER_DIR, 'pc_smiles.npy'), mmap_mode='r')
    _W['ik14'] = np.load(os.path.join(POP_DIR, 'pc_ik14.npy'), mmap_mode='r')
    for p in PRI:
        _W[p] = np.load(os.path.join(POP_DIR, f'pc_{p}.npy'), mmap_mode='r')
    _W['sel_e'] = np.where(BITS < 4096)[0]; _W['raw_e'] = BITS[_W['sel_e']]


def probe(m):
    t0 = time.time()
    c = _W['chem']; Chem = _W['Chem']; gen = _W['gen']
    tol = m['target'] * PPM * 1e-6
    a = int(np.searchsorted(_W['mass'], m['target'] - tol, 'left')); b = int(np.searchsorted(_W['mass'], m['target'] + tol, 'right'))
    n = b - a
    info = dict(ik14=m['ik14'], n_window=n, truth_rows=0, truth_pass1_rank=-1, secs=0.0)
    if n <= 0:
        return info, None
    off = np.asarray(_W['off'][a:b + 1], np.int64)
    raw = bytes(_W['buf'][off[0]:off[-1]]); base = off[0]
    smis = [raw[off[i] - base:off[i + 1] - base].decode('ascii') for i in range(n)]
    ik = np.asarray(_W['ik14'][a:b])
    truth = ik == m['ik14'].encode()
    info['truth_rows'] = int(truth.sum())
    z = m['z']; ze = z[_W['sel_e']]
    E = np.zeros((n, len(ze)), np.float32); ok = np.zeros(n, bool)
    for i, s in enumerate(smis):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        E[i] = gen.GetFingerprintAsNumPy(mol)[_W['raw_e']]; ok[i] = True
    sc1 = E @ ze; sc1[~ok] = -np.inf
    del E
    rank1 = np.empty(n, np.int64); rank1[np.argsort(-sc1, kind='stable')] = np.arange(1, n + 1)
    if truth.any():
        info['truth_pass1_rank'] = int(rank1[truth].min())
    pr = {p: np.asarray(_W[p][a:b]) for p in PRI}
    pop = pr['lsid'].astype(np.float32) + pr['lpmid'].astype(np.float32)
    keep = (rank1 <= PASS1) & ok
    if POPU and ok.any():
        pp = np.where(ok, pop, -np.inf); ku = min(POPU, int(ok.sum()))
        keep[np.argpartition(-pp, ku - 1)[:ku]] = True
    keep |= truth & ok
    idx = np.where(keep)[0]
    fz = np.full(len(idx), np.nan, np.float32)
    for j, i in enumerate(idx):
        fp = c.raw_fingerprint(smis[i])
        if fp is not None:
            fz[j] = float(fp[BITS].astype(np.float32) @ z)
    good = ~np.isnan(fz)
    # export: top KEEP_FZ by fz, top KEEP_POP by pop, truth rows
    exp = np.zeros(len(idx), bool)
    o = np.argsort(-np.where(good, fz, -np.inf), kind='stable'); exp[o[:KEEP_FZ]] = True
    o = np.argsort(-np.where(good, pop[idx], -np.inf), kind='stable'); exp[o[:KEEP_POP]] = True
    exp |= truth[idx]
    exp &= good
    # window-level standardisation over all pass-2 rows (public channel z-scores over its pass-2 set)
    mu, sd = float(fz[good].mean()), float(fz[good].std() + 1e-9)
    info.update(n_pass2=int(good.sum()), fz_mu=mu, fz_sd=sd, n_ok=int(ok.sum()))
    sel_i = idx[exp]
    rows = dict(ik14=np.repeat(m['ik14'], len(sel_i)), row=(a + sel_i).astype(np.int64),
                cand_ik14=ik[sel_i].astype('U14'), smiles=np.array([smis[i] for i in sel_i], object),
                sc1=sc1[sel_i].astype(np.float32), rank1=rank1[sel_i].astype(np.int32), fz=fz[exp],
                in_pass1=(rank1[sel_i] <= PASS1), is_truth=truth[sel_i])
    for p in PRI:
        rows[p] = pr[p][sel_i]
    info['secs'] = time.time() - t0
    return info, pd.DataFrame(rows)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    order = rng.permutation(len(mols))
    tasks = [mols[i] for i in order]
    infos, parts = [], []
    done = 0
    with Pool(4, initializer=init) as pool:
        it = pool.imap_unordered(probe, tasks, chunksize=1)
        for info, df in it:
            infos.append(info)
            if df is not None:
                parts.append(df)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                el = time.time() - T0
                print(f'{done}/{len(tasks)} {el:.0f}s last window {info["n_window"]} {info["secs"]:.1f}s', flush=True)
                pd.DataFrame(infos).to_parquet('/kaggle/working/sim2_progress.parquet', index=False)
            if time.time() - T0 > TIME_BUDGET:
                print('time budget reached, stopping', flush=True)
                pool.terminate()
                break
    meta = pd.DataFrame([{k: v for k, v in m.items() if k != 'z'} for m in mols])
    meta = meta.merge(pd.DataFrame(infos), on='ik14', how='left')
    meta.to_parquet('/kaggle/working/sim2_mols.parquet', index=False)
    np.save('/kaggle/working/sim2_z.npy', np.stack([m['z'] for m in mols]).astype(np.float16))
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet('/kaggle/working/sim2_rows.parquet', index=False)
    print('molecules probed', len(infos), 'with truth in tier', int((meta.truth_rows > 0).sum()),
          'rows', sum(len(p) for p in parts), f'{time.time()-T0:.0f}s', flush=True)
