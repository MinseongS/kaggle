#!/usr/bin/env python3
"""Score run_icecuber.py output against solutions (2 attempts, exact match, per-output partial credit).

Reports: task score (sum of k/N), outputs solved (top-1 / top-2 / any of 3),
fits_all precision (how often a train-verified top-1 candidate is wrong = false positive).
"""
import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--solutions", required=True)
    ap.add_argument("--dump", default=None, help="write per-output detail JSON")
    ap.add_argument("--only-fits", action="store_true", help="count only fits_all candidates as attempts")
    a = ap.parse_args()
    pred = json.load(open(a.pred))
    sol = json.load(open(a.solutions))
    task_score = 0.0
    c = Counter()
    detail = {}
    for k, tests in pred.items():
        gts = sol[k]
        per = []
        for j, t in enumerate(tests):
            gt = gts[j]
            cands = [x for x in t["cands"] if x["fits_all"]] if a.only_fits else t["cands"]
            hit = [x["grid"] == gt for x in cands]
            rank = hit.index(True) + 1 if True in hit else 0
            top = cands[0] if cands else None
            c["outputs"] += 1
            c["has_cand"] += bool(cands)
            c["top1"] += rank == 1
            c["top2"] += 1 <= rank <= 2
            c["top3"] += rank >= 1
            if top is not None and top["fits_all"]:
                c["top1_fits"] += 1
                c["top1_fits_correct"] += rank == 1
                c["top2_fits_correct"] += 1 <= rank <= 2
            any_fit = [x for x in cands if x["fits_all"]]
            c["any_fits"] += bool(any_fit)
            per.append(dict(rank=rank, n_cands=len(cands), top_fits=bool(top and top["fits_all"]),
                            top_score=top["score"] if top else None, n_train=None,
                            top_pass=top["pass"] if top else None))
            task_score += (1 <= rank <= 2) / len(tests)
        detail[k] = per
    n = len(pred)
    print(f"tasks={n} task_score={task_score:.2f} ({100*task_score/n:.2f}%)")
    print(f"outputs={c['outputs']} with_cand={c['has_cand']} top1={c['top1']} top2={c['top2']} top3={c['top3']}")
    tf = c["top1_fits"]
    print(f"top1 fits_all (train-verified): {tf} outputs; top1 correct {c['top1_fits_correct']}, "
          f"top2 correct {c['top2_fits_correct']}; FP rate (top1 wrong | fits) = "
          f"{(tf - c['top1_fits_correct'])/tf if tf else float('nan'):.3f}; "
          f"FP (top2 wrong | fits) = {(tf - c['top2_fits_correct'])/tf if tf else float('nan'):.3f}")
    solved = sorted(k for k, per in detail.items() if any(1 <= p["rank"] <= 2 for p in per))
    print("solved tasks (>=1 output in top2):", solved)
    if a.dump:
        json.dump(detail, open(a.dump, "w"), indent=0)


if __name__ == "__main__":
    main()
