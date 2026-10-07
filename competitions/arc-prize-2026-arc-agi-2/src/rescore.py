"""Offline selection / re-scoring study on saved NVARC candidates (inference_outputs.tar from an eval commit run).

    uv run python src/rescore.py outputs/arc26-nvarc-v3-eval/inference_outputs.tar [--max-pass 1] [--aug-k 4]

Each pickle is one view's decoded beams: [{"beam_score", "score_aug" (8 aug NLLs), "solution"}].
A candidate = distinct grid per test output; its views = the samples that produced it.
Score = sum over tasks of (correct test outputs in top-2) / (test outputs in task), same as the LB.

NVARC 2025 reported pass@2 30.5% vs pass@10 ~40% on eval ("our scorer missed 10%"), so selection is the
cheapest lever. The eval set leaked into NVARC's SFT data, so only low-parameter rules are trusted and the
learned ranker is scored strictly by task-grouped cross-validation.
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


def load_candidates(path, max_pass=None, aug_k=None, max_beam=None, keep_view=None):
    """-> {basekey: [candidate dict]}"""
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
            if keep_view is not None and not keep_view(name):
                continue
            with bz2.BZ2File(os.path.join(root, name)) as f:
                samples = pickle.load(f)
            bk = name.split(".")[0]
            for s in samples:
                if max_beam is not None and s["beam_score"] > max_beam:
                    continue  # e.g. -log(0.2): rebuild the p>=0.2 DFS candidate set from a lower-threshold run
                grid = np.asarray(s["solution"])
                h = (grid.shape, grid.tobytes())
                c = by_key[bk].setdefault(h, {"grid": grid, "aug": None, "views": [], "beam": [], "passes": set()})
                aug = list(s["score_aug"])[:aug_k] if aug_k else list(s["score_aug"])
                c["aug"] = aug  # identical per grid within a pass (cached by grid); last pass wins
                c["views"].append(float(np.mean(aug)))
                c["beam"].append(float(s["beam_score"]))
                c["passes"].add(n_pass)
    out = {}
    for bk, cands in by_key.items():
        lst = []
        n_passes = len(set().union(*(c["passes"] for c in cands.values())))
        for c in cands.values():
            c["n_pp"] = len(c["views"]) / n_passes  # votes per pass: pass-2 re-solves would otherwise double n
            c["n"] = len(c["views"])
            c["aug_mean"] = float(np.mean(c["views"]))
            c["aug_min"] = float(np.min(c["aug"]))
            c["aug_max"] = float(np.max(c["aug"]))
            c["aug_std"] = float(np.std(c["aug"]))
            c["beam_min"] = float(np.min(c["beam"]))
            c["beam_mean"] = float(np.mean(c["beam"]))
            lst.append(c)
        out[bk] = lst
    return out


def view_filter(n_perms, eval_seed=2, data=None):
    """Filename predicate: view (transform + colour perm, ignoring example order) is in augment(n=n_perms)."""
    import sys, types
    sys.modules.setdefault("transformers", types.SimpleNamespace(AutoTokenizer=None))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kernels", "v8", "src"))
    from arc_loader import ArcDataset
    ds = ArcDataset.from_file(os.path.join(data or DATA, "arc-agi_evaluation_challenges.json"))
    strip = lambda k: re.sub(r"\.ex\d+$", "", re.sub(r"\.p\d+$", "", k))
    keep = set()
    for key in ds.keys:
        keep |= {strip(k) for k in ds.change_keys([key]).split_multi_replies().augment(n=n_perms, seed=eval_seed).keys}
    return lambda name: strip(name) in keep


def add_priors(cands, challenges):
    """Model-free sanity checks from the task's train pairs."""
    for bk, lst in cands.items():
        task, i = bk.split("_")
        t = challenges[task]
        tin = np.asarray(t["test"][int(i)]["input"])
        ins = [np.asarray(p["input"]) for p in t["train"]]
        outs = [np.asarray(p["output"]) for p in t["train"]]
        out_colors = set().union(*(set(np.unique(o)) for o in outs))
        in_colors = set(np.unique(tin))
        same_shape = all(a.shape == b.shape for a, b in zip(ins, outs))
        fixed_shape = outs[0].shape if len({o.shape for o in outs}) == 1 else None
        for c in lst:
            g = c["grid"]
            c["colors_ok"] = float(set(np.unique(g)) <= (out_colors | in_colors))
            if same_shape:
                c["shape_ok"] = float(g.shape == tin.shape)
            elif fixed_shape is not None:
                c["shape_ok"] = float(g.shape == fixed_shape)
            else:
                c["shape_ok"] = 1.0
            c["is_input"] = float(g.shape == tin.shape and np.array_equal(g, tin))


FEATURES = ["n", "log_n", "aug_mean", "aug_min", "aug_max", "aug_std", "beam_min", "beam_mean",
            "colors_ok", "shape_ok", "is_input", "rank_n", "rank_aug", "n_frac"]


def feature_rows(lst):
    ns = np.array([c["n"] for c in lst], float)
    augs = np.array([c["aug_mean"] for c in lst])
    rank_n = (-ns).argsort().argsort()
    rank_aug = augs.argsort().argsort()
    rows = []
    for j, c in enumerate(lst):
        c["log_n"] = math.log(c["n"])
        c["rank_n"] = float(rank_n[j])
        c["rank_aug"] = float(rank_aug[j])
        c["n_frac"] = c["n"] / ns.sum()
        rows.append([c[f] for f in FEATURES])
    return np.array(rows, float)


def single_rules():
    r = {
        "kgmon (n - aug)": lambda c: c["n"] - c["aug_mean"],
        "probmul_3": lambda c: sum(3 - b for b in c["beam"]) + (3 - c["aug_mean"]) * len(c["aug"]),
        "kgmon per-pass (n/passes - aug)": lambda c: c["n_pp"] - c["aug_mean"],
        "aug only": lambda c: -c["aug_mean"],
        "v6 score_aug": lambda c: 0.05 * c["n_pp"] - c["aug_mean"],
        "n only (+aug tiebreak)": lambda c: c["n"] - 1e-3 * c["aug_mean"],
        "n - aug - beam_min": lambda c: c["n"] - c["aug_mean"] - c["beam_min"],
        # colors_ok: gold never violates it (0/172 eval, 0/1076 training outputs) -> hard penalty.
        # shape_ok: gold violates it 2.9% (eval) / 0.3% (training) -> soft penalty only.
        "kgmon + colors": lambda c: c["n"] - c["aug_mean"] - 100 * (1 - c["colors_ok"]),
        "kgmon + colors + shape(-1)": lambda c: c["n"] - c["aug_mean"] - 100 * (1 - c["colors_ok"]) - (1 - c["shape_ok"]),
        "n - aug_max": lambda c: c["n"] - c["aug_max"],
    }
    for a in (0.25, 0.5, 2.0, 4.0):
        r[f"{a}*n - aug"] = (lambda a: lambda c: a * c["n"] - c["aug_mean"])(a)
    for b in (0.5, 1.0, 2.0, 4.0):
        r[f"{b}*log(n) - aug"] = (lambda b: lambda c: b * math.log(c["n"]) - c["aug_mean"])(b)
    return r


def pick_single(rule):
    return lambda lst: sorted(lst, key=rule, reverse=True)[:2]


def pick_split(rule1, rule2):
    """attempt_1 by rule1, attempt_2 = best of the rest by rule2 (hedging between two criteria)."""
    def f(lst):
        if not lst:
            return []
        a1 = max(lst, key=rule1)
        rest = [c for c in lst if c is not a1]
        return [a1] + ([max(rest, key=rule2)] if rest else [])
    return f


def evaluate(cands, solutions, picker):
    per_task = defaultdict(list)
    for task, outs in solutions.items():
        for i, gold in enumerate(outs):
            gold = np.asarray(gold)
            chosen = picker(cands.get(f"{task}_{i}", []))
            per_task[task].append(any(c["grid"].shape == gold.shape and np.array_equal(c["grid"], gold) for c in chosen))
    return {t: sum(v) / len(v) for t, v in per_task.items()}


def is_gold(c, gold):
    return c["grid"].shape == gold.shape and np.array_equal(c["grid"], gold)


def ranker_cv(cands, solutions, n_folds=5, seed=0, C=0.3):
    """Task-grouped CV of a logistic ranker. Returns per-task scores (each task scored by a model not trained on it)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tasks = sorted(solutions)
    rng = np.random.default_rng(seed)
    fold_of = {t: f for t, f in zip(rng.permutation(tasks), np.arange(len(tasks)) % n_folds)}
    data = {}
    for task, outs in solutions.items():
        for i, gold in enumerate(outs):
            lst = cands.get(f"{task}_{i}", [])
            if lst:
                data[(task, i)] = (lst, feature_rows(lst), np.array([is_gold(c, np.asarray(gold)) for c in lst], int))
    per_task = defaultdict(list)
    coefs = []
    for f in range(n_folds):
        tr = [v for (t, _), v in data.items() if fold_of[t] != f]
        X = np.vstack([x for _, x, _ in tr])
        y = np.concatenate([y for _, _, y in tr])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
        coefs.append(m.coef_[0])
        for task, outs in solutions.items():
            if fold_of[task] != f:
                continue
            for i in range(len(outs)):
                if (task, i) not in data:
                    per_task[task].append(False)
                    continue
                lst, x, y = data[(task, i)]
                top = np.argsort(-m.decision_function(sc.transform(x)))[:2]
                per_task[task].append(bool(y[top].any()))
    coef = dict(zip(FEATURES, np.round(np.mean(coefs, 0), 2))) if coefs else {}
    return {t: sum(v) / len(v) for t, v in per_task.items()}, coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="inference_outputs.tar or directory of pickles")
    ap.add_argument("--max-pass", type=int, default=None, help="only use candidates from passes <= this")
    ap.add_argument("--aug-k", type=int, default=None, help="use only the first k of the 8 aug NLLs (value of more augs)")
    ap.add_argument("--min-prob", type=float, default=None, help="drop beams with p < this (paired check of a lower DFS threshold)")
    ap.add_argument("--data-dir", default=None, help="dir with arc-agi_evaluation_{challenges,solutions}.json (e.g. synthetic set)")
    ap.add_argument("--decode-perms", type=int, default=None,
                    help="keep only views whose (transform, colour perm) belong to the first N colour perms "
                         "(v8 decodes 3; --decode-perms 2 rebuilds the 16-view baseline of the same run)")
    args = ap.parse_args()

    data = args.data_dir or DATA
    solutions = json.load(open(os.path.join(data, "arc-agi_evaluation_solutions.json")))
    challenges = json.load(open(os.path.join(data, "arc-agi_evaluation_challenges.json")))
    keep = view_filter(args.decode_perms, data=args.data_dir) if args.decode_perms else None
    cands = load_candidates(args.path, args.max_pass, args.aug_k, -math.log(args.min_prob) if args.min_prob else None, keep)
    add_priors(cands, challenges)
    for lst in cands.values():
        feature_rows(lst)
    tasks = sorted({bk.split("_")[0] for bk in cands})
    solutions = {t: solutions[t] for t in tasks}
    print(f"tasks with candidates: {len(tasks)}  test outputs: {sum(map(len, solutions.values()))}")
    for k in (2, 5, 10, 10**9):
        sc = evaluate(cands, solutions, lambda lst, k=k: sorted(lst, key=single_rules()["kgmon (n - aug)"], reverse=True)[:k])
        print(f"{'kgmon pass@' + (str(k) if k < 10**9 else 'all (oracle)'):28s} {sum(sc.values()):6.2f}")
    print()

    rules = single_rules()
    kg = rules["kgmon (n - aug)"]
    base = evaluate(cands, solutions, pick_single(kg))

    def report(name, sc):
        gained = [t for t in tasks if sc.get(t, 0) > base[t]]
        lost = [t for t in tasks if sc.get(t, 0) < base[t]]
        print(f"{name:40s} {sum(sc.values()):6.2f}   vs kgmon +{len(gained)} -{len(lost)}"
              + (f"  +{gained} -{lost}" if (gained or lost) and len(gained) + len(lost) <= 8 else ""))

    for name, rule in rules.items():
        report(name, evaluate(cands, solutions, pick_single(rule)))
    print()
    pp = rules["kgmon per-pass (n/passes - aug)"]
    for n2 in ("aug only", "0.5*log(n) - aug", "probmul_3", "n - aug_max"):
        report(f"a1=per-pass, a2={n2}", evaluate(cands, solutions, pick_split(pp, rules[n2])))
    for n2 in ("kgmon per-pass (n/passes - aug)", "aug only", "n only (+aug tiebreak)", "kgmon + colors", "1.0*log(n) - aug", "probmul_3"):
        report(f"a1=kgmon, a2={n2}", evaluate(cands, solutions, pick_split(kg, rules[n2])))
    print()
    for C in (0.03, 0.3, 3.0):
        scs = [ranker_cv(cands, solutions, seed=s, C=C) for s in range(3)]
        tot = [sum(s.values()) for s, _ in scs]
        report(f"logistic ranker C={C} (5-fold CV, 3 seeds mean)",
               {t: np.mean([s.get(t, 0) for s, _ in scs]) for t in tasks})
        print(f"{'':40s} per-seed {np.round(tot, 2)}  coef {scs[0][1]}")

    print("\ncorrect but outside kgmon top-2 (key, rank, #cands, correct(n, aug, colors_ok, shape_ok), top2 (n, aug)):")
    for task, outs in solutions.items():
        for i, gold in enumerate(outs):
            gold = np.asarray(gold)
            ranked = sorted(cands.get(f"{task}_{i}", []), key=kg, reverse=True)
            for r, c in enumerate(ranked):
                if is_gold(c, gold):
                    if r >= 2:
                        print("  ", f"{task}_{i}", r + 1, len(ranked),
                              (c["n"], round(c["aug_mean"], 3), c["colors_ok"], c["shape_ok"]),
                              [(x["n"], round(x["aug_mean"], 3)) for x in ranked[:2]])
                    break


if __name__ == "__main__":
    main()
