"""Cross-model scoring study on an eval run with ARC_CO_SCORE=1 (v7+): inference_outputs.tar + inference_extra.tar.

    uv run python src/coscore.py outputs/arc26-nvarc-v7-eval [--min-prob 0.2]

Every candidate is normally judged only by the TTT adapter that produced it. The sidecar gives, per candidate grid:
  base      mean NLL under the SFT model with the adapter disabled (8 views)
  adapterN  (pass-N tasks) mean NLL of pass-1 candidates under pass N's adapter
Rules below rank the SAME candidates, so differences are paired.
"""
import argparse
import bz2
import json
import math
import os
import pickle
import sys
import tarfile
import tempfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rescore import load_candidates, evaluate, pick_single, DATA  # noqa: E402


def load_extra(path):
    """-> {(bk, shape, bytes): {kind: mean nll}}"""
    out = defaultdict(dict)
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(path) as tar:
            tar.extractall(tmp, filter="data")
        root = os.path.join(tmp, "inference_extra")
        for name in os.listdir(root):
            with bz2.BZ2File(os.path.join(root, name)) as f:
                for e in pickle.load(f):
                    g = np.asarray(e["solution"])
                    out[(e["bk"], g.shape, g.tobytes())][e["kind"]] = float(np.mean(e["score_aug"]))
    return out


def attach(cands, extra):
    n_base = n_x = 0
    for bk, lst in cands.items():
        for c in lst:
            e = extra.get((bk, c["grid"].shape, c["grid"].tobytes()), {})
            c["base"] = e.get("base")
            # cross-adapter: own-adapter mean (c["aug_mean"], pooled over the passes that found it) + other adapters
            xs = [v for k, v in e.items() if k.startswith("adapter")]
            c["x_mean"] = float(np.mean([c["aug_mean"]] + xs)) if xs else c["aug_mean"]
            n_base += c["base"] is not None
            n_x += bool(xs)
    return n_base, n_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--min-prob", type=float, default=None)
    a = ap.parse_args()
    sol = json.load(open(os.path.join(DATA, "arc-agi_evaluation_solutions.json")))
    cands = load_candidates(os.path.join(a.run_dir, "inference_outputs.tar"), None, None,
                            -math.log(a.min_prob) if a.min_prob else None)
    n_base, n_x = attach(cands, load_extra(os.path.join(a.run_dir, "inference_extra.tar")))
    n_all = sum(map(len, cands.values()))
    print(f"candidates {n_all}, with base score {n_base}, with cross-adapter score {n_x}")
    s = {t: sol[t] for t in {bk.split('_')[0] for bk in cands}}
    BIG = 1e3

    def b(c):  # missing base score -> neutral (own score)
        return c["base"] if c["base"] is not None else c["aug_mean"]
    rules = {"kgmon": lambda c: c["n"] - c["aug_mean"],
             "v6 aug": lambda c: 0.05 * c["n_pp"] - c["aug_mean"],
             "base only": lambda c: 0.05 * c["n_pp"] - b(c),
             "cross-adapter": lambda c: 0.05 * c["n_pp"] - c["x_mean"]}
    for w in (0.25, 0.5, 1.0, 2.0):
        rules[f"aug + {w}*base"] = (lambda w: lambda c: 0.05 * c["n_pp"] - (c["aug_mean"] + w * b(c)) / (1 + w))(w)
        rules[f"xadapt + {w}*base"] = (lambda w: lambda c: 0.05 * c["n_pp"] - (c["x_mean"] + w * b(c)) / (1 + w))(w)
    base = evaluate(cands, s, pick_single(rules["v6 aug"]))
    for name, r in rules.items():
        sc = evaluate(cands, s, pick_single(r))
        g = [t for t in s if sc[t] > base[t]]; l = [t for t in s if sc[t] < base[t]]
        print(f"{name:22s} {sum(sc.values()):6.2f}  vs v6 +{len(g)} -{len(l)}" + (f"  +{g} -{l}" if len(g) + len(l) <= 8 else ""))
    orc = evaluate(cands, s, lambda lst: lst)
    print(f"{'oracle':22s} {sum(orc.values()):6.2f}")


if __name__ == "__main__":
    main()
