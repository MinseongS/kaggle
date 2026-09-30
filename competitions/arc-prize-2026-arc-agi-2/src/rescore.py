"""Offline selection / re-scoring study on saved NVARC candidates (inference_outputs.tar from an eval commit run).

    uv run python src/rescore.py outputs/arc26-nvarc-v2-eval/inference_outputs.tar

Each pickle is one view's decoded beams: [{"beam_score", "score_aug" (8 aug NLLs), "solution"}].
A candidate = distinct grid per test output; its views = the samples that produced it.
Score = sum over tasks of (correct test outputs in top-2) / (test outputs in task), same as the LB.
"""
import argparse
import bz2
import json
import math
import os
import pickle
import re
import tarfile
import tempfile
from collections import defaultdict

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def load_candidates(path, max_pass=None):
    """-> {basekey: [ {grid, n, aug, beam, passes} ]}"""
    with tempfile.TemporaryDirectory() as tmp:
        if path.endswith(".tar"):
            with tarfile.open(path) as tar:
                tar.extractall(tmp, filter="data")
            root = os.path.join(tmp, "inference_outputs")
        else:
            root = path
        by_key = defaultdict(dict)
        for name in os.listdir(root):
            m = re.search(r"\.p(\d+)$", name)
            n_pass = int(m.group(1)) if m else 1
            if max_pass and n_pass > max_pass:
                continue
            with bz2.BZ2File(os.path.join(root, name)) as f:
                samples = pickle.load(f)
            bk = name.split(".")[0]
            for s in samples:
                grid = np.asarray(s["solution"])
                h = (grid.shape, grid.tobytes())
                c = by_key[bk].setdefault(h, {"grid": grid, "aug": [], "beam": [], "passes": set()})
                c["aug"].append(float(np.mean(s["score_aug"])))
                c["beam"].append(float(s["beam_score"]))
                c["passes"].add(n_pass)
    out = {}
    for bk, cands in by_key.items():
        lst = []
        for c in cands.values():
            c["n"] = len(c["aug"])
            c["aug_mean"] = float(np.mean(c["aug"]))
            c["beam_min"] = float(np.min(c["beam"]))
            lst.append(c)
        out[bk] = lst
    return out


def rules():
    r = {
        "kgmon (n - aug)": lambda c: c["n"] - c["aug_mean"],
        "probmul_3": lambda c: sum(3 - b for b in c["beam"]) + (3 * 8 - 8 * c["aug_mean"]),
        "aug only": lambda c: -c["aug_mean"],
        "n only (+aug tiebreak)": lambda c: c["n"] - 1e-3 * c["aug_mean"],
        "n - aug - beam": lambda c: c["n"] - c["aug_mean"] - c["beam_min"],
    }
    for a in (0.25, 0.5, 2.0, 4.0):
        r[f"{a}*n - aug"] = (lambda a: lambda c: a * c["n"] - c["aug_mean"])(a)
    for b in (0.5, 1.0, 2.0, 4.0):
        r[f"{b}*log(n) - aug"] = (lambda b: lambda c: b * math.log(c["n"]) - c["aug_mean"])(b)
    return r


def evaluate(cands, solutions, rule, k=2):
    per_task = defaultdict(list)
    for task, outs in solutions.items():
        for i, gold in enumerate(outs):
            gold = np.asarray(gold)
            ranked = sorted(cands.get(f"{task}_{i}", []), key=rule, reverse=True)[:k]
            per_task[task].append(any(c["grid"].shape == gold.shape and np.array_equal(c["grid"], gold) for c in ranked))
    return {t: sum(v) / len(v) for t, v in per_task.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="inference_outputs.tar or directory of pickles")
    ap.add_argument("--max-pass", type=int, default=None, help="only use candidates from passes <= this")
    args = ap.parse_args()

    solutions = json.load(open(os.path.join(DATA, "arc-agi_evaluation_solutions.json")))
    cands = load_candidates(args.path, args.max_pass)
    tasks = sorted({bk.split("_")[0] for bk in cands})
    solutions = {t: solutions[t] for t in tasks}
    print(f"tasks with candidates: {len(tasks)}  test outputs: {sum(map(len, solutions.values()))}")

    oracle = evaluate(cands, solutions, lambda c: 0, k=10**9)
    print(f"{'oracle (any candidate)':28s} {sum(oracle.values()):6.2f}")

    base = None
    for name, rule in rules().items():
        sc = evaluate(cands, solutions, rule)
        if base is None:
            base = sc
        gained = [t for t in tasks if sc[t] > base[t]]
        lost = [t for t in tasks if sc[t] < base[t]]
        print(f"{name:28s} {sum(sc.values()):6.2f}   vs kgmon +{len(gained)} -{len(lost)}"
              + (f"  gained={gained} lost={lost}" if gained or lost else ""))

    # Where the right answer exists but top-2 misses it: how deep is it?
    ranks = []
    kg = rules()["kgmon (n - aug)"]
    for task, outs in solutions.items():
        for i, gold in enumerate(outs):
            gold = np.asarray(gold)
            ranked = sorted(cands.get(f"{task}_{i}", []), key=kg, reverse=True)
            for r, c in enumerate(ranked):
                if c["grid"].shape == gold.shape and np.array_equal(c["grid"], gold):
                    if r >= 2:
                        ranks.append((f"{task}_{i}", r + 1, len(ranked), c["n"], round(c["aug_mean"], 3),
                                      [(x["n"], round(x["aug_mean"], 3)) for x in ranked[:2]]))
                    break
    print(f"\ncorrect but outside kgmon top-2: {len(ranks)}")
    for row in ranks:
        print("  ", row)


if __name__ == "__main__":
    main()
