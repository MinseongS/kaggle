"""Pick ~2000 train molecules for the kind-2 simulation (no public spectrum, in PubChem).

NP libraries + enveda-np-examples, identity folds 0-4 balanced, neutral mass matched to visible test.
Writes kernels/sim2_gen/selection.json: [{ik14, fold, mass, smiles, src}]
usage: uv run python kernels/sim2_gen/select.py
"""
import json
import numpy as np, pandas as pd

ROOT = 'data'
AD = {'[M+H]+': 1.007276, '[M-H]-': -1.007276, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823,
      '[M+K]+': 38.963158, '[M+CH2O2-H]-': 44.998201, '[M+Cl]-': 34.969402}
NP_LIBS = ['riken', 'gnps', 'msdial', 'massbank', 'mona', 'spectraverse']
N_LIB, SEED = 1750, 20261105

te = pd.read_parquet(f'{ROOT}/comp/test.parquet', columns=['molecule_id', 'adduct', 'precursor_mz'])
te = te[te.adduct.isin(AD)]
te['nm'] = te.precursor_mz - te.adduct.map(AD)
tm = te.groupby('molecule_id').nm.median().values
bins = np.quantile(tm, np.linspace(0, 1, 11)); bins[0], bins[-1] = 157.0, 1160.0

tr = pd.read_parquet(f'{ROOT}/comp/train.parquet', columns=['ingest_lib', 'inchikey14', 'adduct', 'normalized_smiles'])
tr = tr[tr.adduct.isin(AD)]
st = pd.read_parquet(f'{ROOT}/ext/panel/casmi26-identity-disjoint-validation-folds/structures.parquet',
                     columns=['inchikey14', 'mono_mass', 'fold_identity', 'smiles', 'fingerprint_valid'] if False else
                     ['inchikey14', 'mono_mass', 'fold_identity', 'smiles'])
st = st.drop_duplicates('inchikey14').set_index('inchikey14')

npx = set(tr.loc[tr.ingest_lib == 'enveda-np-examples', 'inchikey14'])
lib = set(tr.loc[tr.ingest_lib.isin(NP_LIBS), 'inchikey14']) - npx
cand = pd.DataFrame({'ik14': sorted(lib)})
cand = cand[cand.ik14.isin(st.index)]
cand['mass'] = st.loc[cand.ik14, 'mono_mass'].values
cand['fold'] = st.loc[cand.ik14, 'fold_identity'].values
cand = cand[(cand.mass >= 157) & (cand.mass <= 1160) & cand.fold.between(0, 4)]
cand['bin'] = np.clip(np.searchsorted(bins, cand.mass, 'right') - 1, 0, 9)
rng = np.random.default_rng(SEED)
per = N_LIB // 50   # 10 mass bins x 5 folds
pick = []
for (b, f), g in cand.groupby(['bin', 'fold']):
    pick.append(g.sample(min(per, len(g)), random_state=int(rng.integers(1 << 31))))
pick = pd.concat(pick)
pick['src'] = 'lib'
ex = pd.DataFrame({'ik14': sorted(npx & set(st.index))})
ex['mass'] = st.loc[ex.ik14, 'mono_mass'].values
ex['fold'] = st.loc[ex.ik14, 'fold_identity'].values
ex['src'] = 'npx'
out = pd.concat([pick[['ik14', 'mass', 'fold', 'src']], ex])
out['smiles'] = st.loc[out.ik14, 'smiles'].values
print('candidates', len(cand), 'picked lib', len(pick), 'npx', len(ex))
print(out.groupby(['src', 'fold']).size().unstack())
print('mass median test %.0f sim %.0f' % (np.median(tm), out.mass.median()))
json.dump(out.to_dict('records'), open('kernels/sim2_gen/selection.json', 'w'))
