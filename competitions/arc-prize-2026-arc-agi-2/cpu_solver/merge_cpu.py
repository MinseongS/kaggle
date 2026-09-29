#!/usr/bin/env python3
"""Merge CPU (icecuber) train-verified candidates into an NVARC submission.

Rule (conservative, because train-verified icecuber answers were wrong 3/6 times on local eval):
  only candidates with fits_all=True are used, and NVARC's attempt_1 is never demoted unless NVARC had
  no real candidate.
  1. NVARC attempt_1 is a fallback (no candidate: [[0]] or the test input)  -> CPU goes to attempt_1
     (NVARC's attempt_1 fallback moves to attempt_2 if attempt_2 is also empty).
  2. CPU answer already equals attempt_1 or attempt_2                        -> nothing.
  3. NVARC attempt_2 is empty/fallback/duplicate of attempt_1                 -> CPU goes to attempt_2.
  4. margins given and NVARC attempt_2 is weak (score2 < --weak2)            -> CPU replaces attempt_2.
     (score = kgmon of NVARC's 2nd candidate; to be tuned on NVARC eval-run candidates)
Optional --require-flip-agree keeps only candidates also found by a flip pass (23/33) -- fewer false
positives, fewer hits; decide on NVARC eval data.

Usage: merge_cpu.py --submission submission.json --cpu cpu.json [--nvarc-scores s.json] --out merged.json
  --nvarc-scores: {task_id: [[score_top1, score_top2 or null], ...per test]}  (kgmon scores)
"""
import argparse
import json


def is_fallback(att, test_input):
    return att is None or att == [[0]] or att == test_input


def pick_cpu(cands, require_flip_agree=False):
    fits = [c for c in cands if c["fits_all"]]
    if require_flip_agree:
        fits = [c for c in fits if c.get("n_passes", 1) >= 2]
    return fits[0]["grid"] if fits else None


def merge(sub, cpu, challenges, nvarc_scores=None, weak2=None, require_flip_agree=False):
    stats = dict(a1_fill=0, a2_fill=0, a2_replace=0, dup=0)
    for k, outs in sub.items():
        if k not in cpu:
            continue
        for i, att in enumerate(outs):
            g = pick_cpu(cpu[k][i]["cands"], require_flip_agree) if i < len(cpu[k]) else None
            if g is None:
                continue
            ti = challenges[k]["test"][i]["input"]
            a1, a2 = att.get("attempt_1"), att.get("attempt_2")
            if g == a1 or g == a2:
                stats["dup"] += 1
            elif is_fallback(a1, ti):
                att["attempt_1"] = g
                if is_fallback(a2, ti):
                    att["attempt_2"] = a1 if a1 is not None else [[0]]
                stats["a1_fill"] += 1
            elif is_fallback(a2, ti) or a2 == a1:
                att["attempt_2"] = g
                stats["a2_fill"] += 1
            elif nvarc_scores and weak2 is not None:
                s2 = nvarc_scores.get(k, [[None, None]] * len(outs))[i][1]
                if s2 is None or s2 < weak2:
                    att["attempt_2"] = g
                    stats["a2_replace"] += 1
    return sub, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--nvarc-scores", default=None)
    ap.add_argument("--weak2", type=float, default=None)
    ap.add_argument("--require-flip-agree", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sub = json.load(open(a.submission))
    cpu = json.load(open(a.cpu))
    ch = json.load(open(a.challenges))
    sc = json.load(open(a.nvarc_scores)) if a.nvarc_scores else None
    sub, stats = merge(sub, cpu, ch, sc, a.weak2, a.require_flip_agree)
    json.dump(sub, open(a.out, "w"))
    print("merge stats:", stats)


if __name__ == "__main__":
    main()
