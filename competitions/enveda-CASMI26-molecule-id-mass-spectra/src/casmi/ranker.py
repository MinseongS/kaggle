"""Calibrated ranker: HistGradientBoosting P(candidate is the answer), bagged over class priors x seeds.

Training rows come from a class-1 simulation (M=0) and a class-2 simulation (M=1); sample weight w1 on M=0 rows and
1-w1 on M=1 rows, averaged over CFG.W1_PRIORS x CFG.SEEDS (the notebook's (0.30, 0.60) x (0,1,2,3)).
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config


class Ranker:
    def __init__(self, cfg: config.CFG, priors=None, seeds=None):
        self.cfg = cfg
        self.priors = tuple(priors if priors is not None else cfg.W1_PRIORS)
        self.seeds = tuple(seeds if seeds is not None else cfg.SEEDS)
        self.models = []

    def fit(self, X, Y, M):
        t0 = time.time()
        self.models = []
        for w1 in self.priors:
            W = np.where(M == 0, w1, 1.0 - w1)
            for sd in self.seeds:
                m = HistGradientBoostingClassifier(random_state=sd, **self.cfg.GBM)
                m.fit(X, Y, sample_weight=W)
                self.models.append(m)
        self.fit_secs = time.time() - t0
        return self

    def predict(self, X):
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)

    @classmethod
    def from_rank_train(cls, cfg: config.CFG, path=config.RANK_TRAIN):
        """The notebook's shipped simulation rows (819 query groups; identities unknown -> not usable inside CV)."""
        z = np.load(path)
        r = cls(cfg).fit(z['X'], z['Y'], z['M'])
        print(f'ranker (rank_train.npz): {len(r.models)} GBMs on {len(z["Y"]):,} rows ({r.fit_secs:.0f}s)', flush=True)
        return r
