"""CLI:  uv run casmi build | cv | predict

  casmi build                               # one-time caches (library, metric keys, train-structure fingerprints)
  casmi cv --n-np 200 --n-e180 300 --tag exp000
  casmi predict --ranker rank_train --out outputs/submission.csv
  casmi predict --ranker cv --cv-tag exp000 --out outputs/submission_cvranker.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import time

from . import config


def _engine(cfg, prefer_instrument='timsTOF'):
    from .candidates import Pool
    from .engine import Engine
    from .frag import Fragmenter
    from .library import Library
    bank = None
    if cfg.USE_FP_MODEL:
        from .fpnet import ModelBank
        paths = sorted(glob.glob(str(config.FP_MODEL_DIR / 'fp_*.pt')))
        bank = ModelBank(paths, device=cfg.DEVICE) if paths else None
    L = Library(cfg)
    P = Pool(cfg)
    fr = Fragmenter(cfg.WORKERS) if cfg.USE_FRAG else None
    return Engine(cfg, L, P, bank, fr, prefer_instrument)


def main(argv=None):
    ap = argparse.ArgumentParser('casmi')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('build')
    c = sub.add_parser('cv')
    c.add_argument('--n-np', type=int, default=200); c.add_argument('--n-e180', type=int, default=300)
    c.add_argument('--seed', type=int, default=0); c.add_argument('--tag', default='exp000')
    c.add_argument('--no-reuse', action='store_true')
    c.add_argument('--holdout', choices=['cv', 'all'], default=None,
                   help='fixed list from holdout_v1.json (cv == exp000 molecules) instead of sampling')
    p = sub.add_parser('predict')
    p.add_argument('--ranker', choices=['rank_train', 'cv'], default='rank_train')
    p.add_argument('--cv-tag', default='exp000')
    p.add_argument('--out', default=str(config.OUTPUTS / 'submission.csv'))
    for s in (c, p):
        s.add_argument('--device', default='cpu'); s.add_argument('--no-fp', action='store_true')
        s.add_argument('--no-frag', action='store_true'); s.add_argument('--workers', type=int, default=None)
        s.add_argument('--fp-dir', default=None, help='directory with fp_*.pt (default: data/ext/casmi26-fp-models-v2)')
    a = ap.parse_args(argv)
    cfg = config.CFG()
    if getattr(a, 'workers', None): cfg.WORKERS = a.workers
    if getattr(a, 'device', None): cfg.DEVICE = a.device
    if getattr(a, 'no_fp', False): cfg.USE_FP_MODEL = False
    if getattr(a, 'no_frag', False): cfg.USE_FRAG = False
    if getattr(a, 'fp_dir', None):
        from pathlib import Path
        config.FP_MODEL_DIR = Path(a.fp_dir)
    T0 = time.time()
    if a.cmd == 'build':
        from .candidates import build_train_fp_cache
        from .library import build_library_cache
        build_library_cache(cfg); build_train_fp_cache(cfg)
    elif a.cmd == 'cv':
        from .cv import run_cv
        E = _engine(cfg)
        print(f'engine ready {time.time()-T0:.0f}s', flush=True)
        run_cv(E, cfg, a.n_np, a.n_e180, a.seed, a.tag, reuse=not a.no_reuse, holdout=a.holdout)
    elif a.cmd == 'predict':
        from . import submit
        from .ranker import Ranker
        queries, pref = submit.test_queries()
        E = _engine(cfg, pref)
        t_eng = time.time() - T0
        if a.ranker == 'rank_train':
            rk = Ranker.from_rank_train(cfg)
        else:
            from .cv import FP_COLS, cv_ranker
            rk = cv_ranker(cfg, a.cv_tag, FP_COLS if not cfg.USE_FP_MODEL else None)
        t1 = time.time()
        rows, diag = submit.predict_test(E, rk, queries, cfg)
        t_pred = time.time() - t1
        df = submit.write_submission(rows, a.out)
        meta = dict(out=a.out, ranker=a.ranker, secs_total=round(time.time() - T0), secs_setup=round(t_eng),
                    secs_predict=round(t_pred), n=len(df), mean_len=float(df['smiles'].str.split(';').list.len().mean()))
        json.dump(meta, open(a.out.replace('.csv', '.meta.json'), 'w'), indent=1)
        print(f'wrote {a.out} {df.shape}; {meta}', flush=True)
    if getattr(locals().get('E', None), 'frag', None) is not None:
        E.frag.close()


if __name__ == '__main__':
    main()
