"""Masses, adducts, neutral-mass derivation and RDKit fingerprints (identical constants to the notebook)."""
from __future__ import annotations

import re

import numpy as np

MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163,
            S=31.97207100, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473,
            Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213)
E = 0.00054857990
PROTON = MASS['H'] - E
H2O = 2 * MASS['H'] + MASS['O']
NH4 = MASS['N'] + 4 * MASS['H']
FORMATE = MASS['C'] + 2 * MASS['H'] + 2 * MASS['O']
ACETATE = 2 * MASS['C'] + 4 * MASS['H'] + 2 * MASS['O']
ADDUCTS = {
    "[M+H]+": (1, 1, PROTON), "[M+NH4]+": (1, 1, NH4 - E), "[M+Na]+": (1, 1, MASS['Na'] - E),
    "[M+K]+": (1, 1, MASS['K'] - E), "[M-H2O+H]+": (1, 1, PROTON - H2O), "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
    "[M+2H]2+": (1, 2, 2 * PROTON), "[M]+": (1, 1, -E), "[M-H2O]+": (1, 1, -E - H2O),
    "[M+CH3OH+H]+": (1, 1, PROTON + MASS['C'] + 4 * MASS['H'] + MASS['O']),
    "[M+CH3CN+H]+": (1, 1, PROTON + 2 * MASS['C'] + 3 * MASS['H'] + MASS['N']),
    "[M-H]-": (1, 1, -PROTON), "[M-H2O-H]-": (1, 1, -PROTON - H2O), "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),
    "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON), "[M+Cl]-": (1, 1, MASS['Cl'] + E), "[M]-": (1, 1, E),
    "[M-2H]-": (1, 2, -2 * PROTON), "[M+Na-2H]-": (1, 1, MASS['Na'] - 2 * PROTON),
    "[2M+H]+": (2, 1, PROTON), "[2M+Na]+": (2, 1, MASS['Na'] - E), "[2M+NH4]+": (2, 1, NH4 - E),
    "[2M+K]+": (2, 1, MASS['K'] - E), "[2M-H]-": (2, 1, -PROTON), "[2M+CH2O2-H]-": (2, 1, FORMATE - PROTON),
    "[2M+C2H4O2-H]-": (2, 1, ACETATE - PROTON), "[2M+Na-2H]-": (2, 1, MASS['Na'] - 2 * PROTON),
    "[3M+H]+": (3, 1, PROTON), "[3M-H]-": (3, 1, -PROTON),
}
MONO = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, S=31.97207100,
            P=30.97376163, Cl=34.96885268, Br=78.9183371, F=18.99840322, I=126.904473,
            Si=27.9769265325, Se=79.9165213, Na=22.9897692809, K=38.96370668, B=11.0093054,
            As=74.9215965, Fe=55.9349375, D=2.0141017778)
_TOK = re.compile(r'([A-Z][a-z]?)(\d*)')


def formula_mass(f):
    """Neutral monoisotopic mass by arithmetic on a molecular formula (NaN if unknown element)."""
    if not isinstance(f, str) or not f:
        return np.nan
    m = 0.0
    for el, n in _TOK.findall(f):
        if not el:
            continue
        if el not in MONO:
            return np.nan
        m += MONO[el] * (int(n) if n else 1)
    return m


def neutral_mass(mz, adduct):
    mz = np.asarray(mz, np.float64)
    out = np.full(len(mz), np.nan)
    ad = np.asarray(adduct, dtype=object)
    for a, (n, z, d) in ADDUCTS.items():
        m = (ad == a)
        if m.any():
            out[m] = (mz[m] * z - d) / n
    return out


# ---------------------------------------------------------------- fingerprints (ECFP4|ECFP6|RDKit|MACCS)[BITS]
_g: dict = {}


def _fp_init():
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    _g['m2'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _g['m3'] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _g['rk'] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)


def raw_fp(smi):
    """Full 10,407-bit fingerprint (uint8) and exact mass, or None."""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem.Descriptors import ExactMolWt
    if not _g:
        _fp_init()
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        fp = np.concatenate([_g['m2'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g['m3'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g['rk'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])
        return fp, float(ExactMolWt(m))
    except Exception:
        return None
