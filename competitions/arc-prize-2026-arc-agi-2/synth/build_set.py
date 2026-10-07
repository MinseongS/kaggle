"""Build a task set in the competition's file format from synthetic generators.

    uv run python synth/build_set.py --out synth/out/val --gens 'synth/gens/*.py' --per-gen 3 --seed0 100000

Writes arc-agi_evaluation_challenges.json / _solutions.json (what the kernels read in eval mode) plus
manifest.json (task id -> generator, seed). Task ids are 8 hex chars from crc32(generator:seed).
The marker file synth_set.marker lets a kernel find the directory among its inputs.
"""
import argparse
import glob
import json
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.dirname(__file__))
from check import check_task, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gens", default=os.path.join(os.path.dirname(__file__), "gens", "*.py"))
    ap.add_argument("--per-gen", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=100000, help="validation seeds start here; FT data must use other seeds")
    a = ap.parse_args()
    paths = sorted(p for p in glob.glob(a.gens) if not os.path.basename(p).startswith("_"))
    ch, sol, man = {}, {}, {}
    for path in paths:
        m = load(path)
        name = os.path.basename(path)[:-3]
        for k in range(a.per_gen):
            seed = a.seed0 + k
            t = m.generate(random.Random(seed))
            check_task(t, f"{name}:{seed}")
            tid = f"{zlib.crc32(f'{name}:{seed}'.encode()):08x}"
            assert tid not in ch, f"id collision {tid}"
            ch[tid] = {"train": t["train"], "test": [{"input": p["input"]} for p in t["test"]]}
            sol[tid] = [p["output"] for p in t["test"]]
            man[tid] = {"generator": name, "seed": seed, "concept": m.CONCEPT}
    os.makedirs(a.out, exist_ok=True)
    for fn, obj in [("arc-agi_evaluation_challenges.json", ch), ("arc-agi_evaluation_solutions.json", sol), ("manifest.json", man)]:
        json.dump(obj, open(os.path.join(a.out, fn), "w"))
    open(os.path.join(a.out, "synth_set.marker"), "w").write("synthetic validation set\n")
    print(f"wrote {len(ch)} tasks from {len(paths)} generators to {a.out}")


if __name__ == "__main__":
    main()
