"""Configuration. Defaults reproduce prvsiyan/analog-propagation-casmi-2026-baseline (the 0.335/0.328 config).

Every constant carries the notebook's own measurement comment in research/04-pipeline-analysis.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CASMI_DATA", ROOT / "data"))
EXT = Path(os.environ.get("CASMI_EXT", DATA / "ext"))
CACHE = Path(os.environ.get("CASMI_CACHE", DATA / "cache"))
OUTPUTS = Path(os.environ.get("CASMI_OUTPUTS", ROOT / "outputs"))

TRAIN = DATA / "train.parquet"
TEST = DATA / "test.parquet"
SAMPLE = DATA / "sample_submission.csv"

COCO_DIR = EXT / "coconut-casmi26-candidates"
BIO_DIR = EXT / "chebi-lipidmaps-casmi26"
FP_MODEL_DIR = EXT / "casmi26-fp-models-v2"
RANK_TRAIN = EXT / "casmi26-ranker-features" / "rank_train.npz"


@dataclass
class CFG:
    # candidate generation
    PPM_WIN: float = 10.0
    PPM_FALLBACK: float = 30.0
    # spectrum cleaning (library + query side of the similarity channels)
    INT_FLOOR: float = 0.002
    MAX_PEAKS: int = 256
    MZ_TOL: float = 0.01
    INT_POWER: float = 1.0
    ENT_WEIGHT: bool = True
    # analog propagation
    ANALOG_WIN: float = 200.0
    N_ANALOG: int = 200
    SIM_POWER: float = 3.0  # used inside rank_features (P_SIM)
    # ranker
    W1_PRIORS: tuple = (0.30, 0.60)
    SEEDS: tuple = (0, 1, 2, 3)
    GBM: dict = field(default_factory=lambda: dict(max_depth=6, max_iter=500, learning_rate=0.03,
                                                   min_samples_leaf=80, l2_regularization=1.0))
    USE_BIO_DB: bool = False
    CAND_CAP: int = 500
    LIB_MASS_FORMULA: bool = True
    ANALOG_MATCH_INSTR: bool = True
    TOPN: int = 25
    # channels on/off (for ablations)
    USE_FP_MODEL: bool = True
    USE_FRAG: bool = True
    # runtime
    WORKERS: int = max(1, (os.cpu_count() or 4) - 1)
    DEVICE: str = "cpu"  # 'cpu' | 'mps' | 'cuda' for FPNet
